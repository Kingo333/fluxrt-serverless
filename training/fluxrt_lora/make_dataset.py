"""Turn our r2i shards (the same triples JoyAI trains on) into a FluxRT LoRA dataset.

  python3 fluxrt_lora/make_dataset.py --manifest-dir $AZ_MANIFEST_DIR --out DIR [--max 3000] [--size 288x512]

Reads <manifest-dir>/i2i.json (tar shards; r2i samples are 0.png garment, 1.png person
source, 2.png target) and writes DIR/data/train-NNNNN.parquet with image columns
person / garment / target plus caption, in the Hugging Face datasets layout, so
`--dataset_name DIR/data` loads it in the diffusers trainer. Only parquet goes in
data/ (load_dataset would also try to read stray JSON files there).

- person and target: centre-cropped and resized together to the FluxRT input size
  (default 288x512, the endpoint's FLUXRT_WIDTH/HEIGHT), so training sees what
  FluxRT sees at inference.
- garment: kept whole (long side <= 1024); the patched trainer pads it to 384x512.
- caption: fluxrt_prompt(), the Python twin of the app's buildTryOnPrompt().
- --video-frames N: also take N matching stills from each iv2v clip pair in
  <manifest-dir>/v2v.json (0.jpg garment, 0.mp4 source, 1.mp4 target), e.g. ViViD.
  Source and target frames come from the same timestamps; uses ffmpeg.
- selection.json is copied next to the data, so provenance (trained_with_vivid,
  commercial_use_cleared) travels with the LoRA.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_tools.fluxrt_prompt import fluxrt_prompt  # noqa: E402

IMAGE_FEATURE = {"_type": "Image"}
FEATURES = {"key": {"dtype": "string", "_type": "Value"}, "caption": {"dtype": "string", "_type": "Value"},
            "person": IMAGE_FEATURE, "garment": IMAGE_FEATURE, "target": IMAGE_FEATURE}


def iter_r2i(manifest_dir: Path):
    manifest = json.loads((manifest_dir / "i2i.json").read_text())
    for shard in manifest["resolved_files"]:
        groups: dict[str, dict[str, bytes]] = {}
        with tarfile.open(shard) as tar:
            for m in tar:
                if not m.isfile():
                    continue
                key, _, name = m.name.partition(".")
                groups.setdefault(key, {})[name] = tar.extractfile(m).read()
        for key, files in groups.items():
            meta = json.loads(files.get("json", b"{}"))
            names = [n for n in ("0", "1", "2") if f"{n}.png" in files or f"{n}.jpg" in files]
            if names != ["0", "1", "2"]:
                continue  # i2i without a garment: FluxRT always gets one
            pick = lambda n: files.get(f"{n}.png") or files[f"{n}.jpg"]  # noqa: E731
            yield key, pick("0"), pick("1"), pick("2"), meta


def _frames(mp4: bytes, n: int) -> list[bytes]:
    """n evenly spaced PNG frames from an mp4 (ffmpeg), same indices for any clip of equal length."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "c.mp4"
        src.write_bytes(mp4)
        count = int(subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
             "stream=nb_read_frames", "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=True).stdout.strip() or 0)
        if count <= 0:
            return []
        idx = sorted({min(count - 1, round((i + 0.5) * count / n)) for i in range(n)})
        sel = "+".join(f"eq(n\\,{i})" for i in idx)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"select='{sel}'", "-vsync", "0",
                        str(Path(d) / "f%03d.png")], check=True)
        return [p.read_bytes() for p in sorted(Path(d).glob("f*.png"))]


def iter_video_stills(manifest_dir: Path, per_clip: int):
    path = manifest_dir / "v2v.json"
    if per_clip <= 0 or not path.exists():
        return
    for shard in json.loads(path.read_text())["resolved_files"]:
        groups: dict[str, dict[str, bytes]] = {}
        with tarfile.open(shard) as tar:
            for m in tar:
                if m.isfile():
                    key, _, name = m.name.partition(".")
                    groups.setdefault(key, {})[name] = tar.extractfile(m).read()
        for key, files in groups.items():
            if not {"0.jpg", "0.mp4", "1.mp4"} <= set(files):
                continue
            meta = json.loads(files.get("json", b"{}"))
            src, tgt = _frames(files["0.mp4"], per_clip), _frames(files["1.mp4"], per_clip)
            for i, (p, t) in enumerate(zip(src, tgt)):
                yield f"{key}_f{i}", files["0.jpg"], p, t, meta


def _cover(img, w: int, h: int):
    from PIL import Image
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * scale)), max(h, round(img.height * scale))), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def _jpeg(img, quality=95) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def convert(garment: bytes, person: bytes, target: bytes, size: tuple[int, int]):
    from PIL import Image
    g = Image.open(io.BytesIO(garment)).convert("RGB")
    if max(g.size) > 1024:
        g.thumbnail((1024, 1024), Image.LANCZOS)
    p = _cover(Image.open(io.BytesIO(person)).convert("RGB"), *size)
    t = _cover(Image.open(io.BytesIO(target)).convert("RGB"), *size)
    return _jpeg(p), _jpeg(g), _jpeg(t)


def caption_for(meta: dict) -> str:
    m = meta.get("metadata", {})
    return fluxrt_prompt(m.get("category"), m.get("garment_name"), m.get("garment_description"))


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    img = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema([("key", pa.string()), ("caption", pa.string()),
                        ("person", img), ("garment", img), ("target", img)],
                       metadata={"huggingface": json.dumps({"info": {"features": FEATURES}})})
    cols = {"key": [r["key"] for r in rows], "caption": [r["caption"] for r in rows]}
    for c in ("person", "garment", "target"):
        cols[c] = [{"bytes": r[c], "path": f"{r['key']}_{c}.jpg"} for r in rows]
    pq.write_table(pa.table(cols, schema=schema), path)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--size", default="288x512", help="WIDTHxHEIGHT of person/target (multiples of 16)")
    ap.add_argument("--per-file", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-frames", type=int, default=0, help="stills per iv2v clip pair (0 = stills only)")
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    if w % 16 or h % 16:
        raise SystemExit("--size must be multiples of 16")
    mdir, out = Path(a.manifest_dir), Path(a.out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)

    items = list(iter_r2i(mdir)) + list(iter_video_stills(mdir, a.video_frames))
    random.Random(a.seed).shuffle(items)
    items = items[: a.max]
    rows, files = [], 0
    for key, g, p, t, meta in items:
        person, garment, target = convert(g, p, t, (w, h))
        rows.append({"key": key, "caption": caption_for(meta), "person": person, "garment": garment, "target": target})
        if len(rows) >= a.per_file:
            write_parquet(rows, data / f"train-{files:05d}.parquet")
            files, rows = files + 1, []
    if rows:
        write_parquet(rows, data / f"train-{files:05d}.parquet")
        files += 1
    if (mdir / "selection.json").exists():
        shutil.copy(mdir / "selection.json", out / "selection.json")
    summary = {"samples": len(items), "files": files, "size": [w, h], "dataset_name": str(data)}
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()

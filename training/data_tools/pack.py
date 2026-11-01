"""Staged pairs (with generated sources) -> JoyAI shards + catalog.

  python3 -m data_tools.pack STAGING_DIR OUT_DIR --name glance_upper [--per-shard 500]

Pairs without a source yet are skipped and counted. About 10% of training pairs are
also written as clean-input "identity" pairs (source = the real target), --identity-frac. Images become r2i samples
(0.png garment, 1.png source, 2.png target); videos become iv2v samples
(0.jpg garment, 0.mp4 source, 1.mp4 target). Writes OUT_DIR/<name>.catalog.json,
which manifests.py turns into training manifests.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

from .shards import Sample, write_shards
from .staging import find, instruction, iter_pairs


def _as_jpeg(data: bytes) -> bytes:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        out = io.BytesIO()
        im.convert("RGB").save(out, format="JPEG", quality=95)
        return out.getvalue()


def _frame_count(data: bytes) -> int:
    import av
    with av.open(io.BytesIO(data)) as c:
        return sum(1 for _ in c.decode(video=0))


def _trim_video(data: bytes, n: int) -> bytes:
    """First n frames, re-encoded at the same size and rate (crf 18)."""
    import av
    out = io.BytesIO()
    with av.open(io.BytesIO(data)) as src, av.open(out, "w", format="mp4") as dst:
        st = src.streams.video[0]
        enc = dst.add_stream("libx264", rate=round(float(st.average_rate or 24)))
        enc.width, enc.height, enc.pix_fmt = st.codec_context.width, st.codec_context.height, "yuv420p"
        enc.options = {"crf": "18"}
        for i, frame in enumerate(src.decode(video=0)):
            if i >= n:
                break
            for pkt in enc.encode(frame.reformat(format="yuv420p")):
                dst.mux(pkt)
        for pkt in enc.encode():
            dst.mux(pkt)
    return out.getvalue()


def _matched_target(src: bytes, tgt: bytes) -> bytes:
    """The loader picks the same relative window in source and target, so they must be
    the same length. A capped pseudo-source (--max-frames) gets a trimmed target copy;
    the staged original is never changed."""
    ns, nt = _frame_count(src), _frame_count(tgt)
    return _trim_video(tgt, ns) if ns < nt else tgt


def _sample(d: Path, meta: dict) -> Sample | None:
    ref, src, tgt = find(d, "ref"), find(d, "source"), find(d, "target")
    if not (ref and src and tgt):
        return None
    detailed = None
    if meta.get("garment_description"):
        detailed = instruction(meta["category"], meta.get("garment_name"), meta["garment_description"])
    common = dict(key=d.name, caption=meta.get("caption") or instruction(meta["category"]),
                  captions={"caption_detailed": detailed},
                  source=meta["source"], split=meta["split"], garment_id=meta.get("garment_id"),
                  person_id=meta.get("person_id"), consent_id=meta.get("consent_id"),
                  extra={k: meta[k] for k in ("config", "category", "pseudo_source") if k in meta})
    if tgt.suffix == ".mp4":
        src_bytes = src.read_bytes()
        return Sample(task="iv2v", files={"0.jpg": _as_jpeg(ref.read_bytes()), "0.mp4": src_bytes,
                                          "1.mp4": _matched_target(src_bytes, tgt.read_bytes())}, **common)
    files = {"0.png": ref.read_bytes(), "1.png": src.read_bytes(), "2.png": tgt.read_bytes()}
    if any(p.suffix != ".png" for p in (ref, src, tgt)):
        files = {"0.jpg": _as_jpeg(ref.read_bytes()), "1.jpg": _as_jpeg(src.read_bytes()), "2.jpg": _as_jpeg(tgt.read_bytes())}
    return Sample(task="r2i", files=files, **common)


def _picked(key: str, frac: float) -> bool:
    import hashlib
    return frac > 0 and int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def _identity(s: Sample) -> Sample:
    """Clean-input pair (review R5): source = the real target, garment unchanged, so the
    model also sees real camera-like input, not only JoyAI-generated sources."""
    files = dict(s.files)
    if s.task == "iv2v":
        files["0.mp4"] = files["1.mp4"]
    else:
        src_name = "1.png" if "1.png" in files else "1.jpg"
        tgt_name = "2.png" if "2.png" in files else "2.jpg"
        files[src_name] = files[tgt_name]
    return Sample(key=f"{s.key}_id", task=s.task, files=files, caption=s.caption, source=s.source, split=s.split,
                  garment_id=s.garment_id, person_id=s.person_id, consent_id=s.consent_id,
                  extra={**s.extra, "identity_pair": True}, captions=dict(s.captions))


def pack(staging_dir: str, out_dir: str, name: str, per_shard: int = 500, identity_frac: float = 0.1) -> dict:
    buckets: dict[tuple, list[Sample]] = defaultdict(list)
    missing = 0
    identity = 0
    for d, meta in iter_pairs(Path(staging_dir)):
        s = _sample(d, meta)
        if s is None:
            missing += 1
            continue
        buckets[(s.task, s.source, s.split)].append(s)
        if s.split == "train" and _picked(s.key, identity_frac):
            buckets[(s.task, s.source, s.split)].append(_identity(s))
            identity += 1
    catalog = []
    for (task, source, split), items in sorted(buckets.items()):
        catalog += write_shards(items, out_dir, f"{name}-{task}-{source}-{split}", per_shard)
    path = Path(out_dir) / f"{name}.catalog.json"
    path.write_text(json.dumps(catalog, indent=2))
    return {"catalog": str(path), "shards": len(catalog), "samples": sum(e["num_samples"] for e in catalog),
            "skipped_no_source": missing, "identity_pairs": identity}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-shard", type=int, default=500)
    ap.add_argument("--identity-frac", type=float, default=0.1,
                    help="share of training pairs also added as clean-input 'no change' pairs (review R5)")
    a = ap.parse_args()
    print(json.dumps(pack(a.staging_dir, a.out_dir, a.name, a.per_shard, a.identity_frac), indent=2))


if __name__ == "__main__":
    main()

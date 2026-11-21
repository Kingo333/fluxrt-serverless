"""What is on the training volume, counted (fxreal6 plan, step 3; owner, 2026-11-15).

  python3 inventory.py /workspace/azyah-vton            # prints a short table and INVENTORY_JSON={...}
  python3 inventory.py /workspace/azyah-vton --probe 30 # also measures 30 random clips with ffprobe

Read-only, standard library only (no venv, no GPU, no network): it runs on the smallest CPU Pod
with the azyah-vton-train volume attached. It answers the plan's open counts:

  - staged clips and stills per set and per garment type (dress / upper / lower), train vs held-out
  - how many ViViD videos are still inside the zip, per type (not extracted yet)
  - VITON-HD files on disk
  - datasets already built, LoRA files (with checksums), base models, test sets, own and hijab clips
  - free space
  - with --probe: frames, frames per second and seconds of a sample of clips (ffprobe, if installed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")
# ViViD's folder names -> our garment types.
ZIP_TYPES = {"upper_body": "upper", "lower_body": "lower", "dresses": "dress", "dress": "dress"}


def _count(d: Path, exts: tuple[str, ...]) -> int:
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts) if d.is_dir() else 0


def _size_gb(d: Path) -> float:
    if d.is_file():
        return round(d.stat().st_size / 1e9, 2)
    return round(sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e9, 2) if d.is_dir() else 0.0


def staged_set(d: Path) -> dict:
    """One staging folder: items (sub-folders with meta.json) by garment type and split."""
    by_type: Counter = Counter()
    by_split: Counter = Counter()
    with_video = with_source = described = 0
    items = 0
    for m in sorted(d.glob("*/meta.json")):
        try:
            meta = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        items += 1
        by_type[str(meta.get("category") or "unknown")] += 1
        by_split[str(meta.get("split") or "unknown")] += 1
        folder = m.parent
        with_video += any((folder / f"target{e}").exists() for e in VIDEO_EXT)
        with_source += any((folder / f"source{e}").exists() for e in VIDEO_EXT + IMAGE_EXT)
        described += bool(str(meta.get("garment_description") or "").strip())
    return {"items": items, "by_type": dict(sorted(by_type.items())), "by_split": dict(sorted(by_split.items())),
            "with_video": with_video, "with_input": with_source, "with_description": described}


def zip_videos(z: Path) -> dict:
    """Videos inside a dataset zip, by garment type (from the folder names)."""
    by_type: Counter = Counter()
    try:
        with zipfile.ZipFile(z) as f:
            for n in f.namelist():
                if not n.lower().endswith(VIDEO_EXT):
                    continue
                parts = [p.lower() for p in n.split("/")]
                kind = next((ZIP_TYPES[p] for p in parts if p in ZIP_TYPES), "unknown")
                by_type[kind] += 1
    except (OSError, zipfile.BadZipFile) as e:
        return {"error": str(e)}
    return {"videos": sum(by_type.values()), "by_type": dict(sorted(by_type.items())), "size_gb": _size_gb(z)}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(video: Path) -> dict | None:
    """Frames, frames per second and seconds of one clip (None without ffprobe or on a bad file)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
             "stream=nb_read_packets,r_frame_rate,width,height", "-of", "json", str(video)],
            capture_output=True, text=True, timeout=60, check=True).stdout
        s = json.loads(out)["streams"][0]
        num, den = (s.get("r_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        frames = int(s.get("nb_read_packets") or 0)
        return {"frames": frames, "fps": round(fps, 2), "seconds": round(frames / fps, 2) if fps else 0.0,
                "size": f"{s.get('width')}x{s.get('height')}"}
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
        return None


def probe_sample(staging: Path, n: int, seed: int = 7) -> dict:
    clips = sorted(p for p in staging.glob("*/*/target.mp4"))
    if not clips or n <= 0:
        return {"probed": 0}
    if not shutil.which("ffprobe"):
        return {"probed": 0, "note": "ffprobe not installed (apt-get install -y ffmpeg)"}
    picked = random.Random(seed).sample(clips, min(n, len(clips)))
    got = [r for r in (probe(p) for p in picked) if r and r["frames"] > 0]
    if not got:
        return {"probed": 0}
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    return {"probed": len(got), "frames_median": med([r["frames"] for r in got]), "frames_min": min(r["frames"] for r in got),
            "fps_median": med([r["fps"] for r in got]), "seconds_median": med([r["seconds"] for r in got]),
            "sizes": dict(Counter(r["size"] for r in got).most_common(3))}


def inventory(root: Path, probe_n: int = 0) -> dict:
    data, staging = root / "data", root / "data" / "staging"
    inv: dict = {"root": str(root), "exists": root.is_dir()}
    if not root.is_dir():
        return inv
    usage = shutil.disk_usage(root)
    inv["disk"] = {"free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1)}

    inv["staging"] = {d.name: staged_set(d) for d in sorted(staging.iterdir()) if d.is_dir()} if staging.is_dir() else {}
    inv["zips"] = {str(z.relative_to(root)): zip_videos(z) for z in sorted(data.rglob("*.zip"))} if data.is_dir() else {}

    viton = data / "raw" / "viton_hd"
    inv["viton_hd"] = {"downloaded": (viton / ".downloaded").exists(), "parquet_files": _count(viton, (".parquet",)),
                       "images": _count(viton, IMAGE_EXT), "size_gb": _size_gb(viton)}

    built = {}
    for s in sorted((data / "fluxrt_lora").glob("*/dataset_summary.json")) if (data / "fluxrt_lora").is_dir() else []:
        try:
            built[s.parent.name] = json.loads(s.read_text())
        except (OSError, ValueError):
            built[s.parent.name] = {"error": "unreadable"}
    inv["datasets_built"] = built

    loras = {}
    out = root / "outputs" / "fluxrt_lora"
    for d in sorted(out.iterdir()) if out.is_dir() else []:
        w = d / "pytorch_lora_weights.safetensors"
        if d.is_dir():
            loras[d.name] = {"weights_mb": round(w.stat().st_size / 1e6, 1) if w.exists() else None,
                             "sha256": sha256(w) if w.exists() else None,
                             "selection_json": (d / "selection.json").exists(),
                             "checkpoints": len(list(d.glob("checkpoint-*")))}
    inv["loras"] = loras

    ck = root / "checkpoints"
    inv["models"] = {d.name: _size_gb(d) for d in sorted(ck.iterdir()) if d.is_dir()} if ck.is_dir() else {}

    ev = root / "eval"
    sets = {}
    for j in sorted(ev.glob("*/*.json")) if ev.is_dir() else []:
        if j.name.endswith("_descriptions.json") or j.name in ("descriptions.json",):
            continue
        try:
            body = json.loads(j.read_text())
            n = len(body) if isinstance(body, (list, dict)) else 0
        except (OSError, ValueError):
            n = -1
        sets[str(j.relative_to(ev))] = n
    inv["eval_sets"] = sets
    inv["own_clips"] = _count(ev / "own_clips", VIDEO_EXT)
    inv["hijab_clips"] = _count(ev / "hijab_clips", VIDEO_EXT)
    if probe_n:
        inv["clip_probe"] = probe_sample(staging, probe_n)
    return inv


def summary_lines(inv: dict) -> list[str]:
    if not inv.get("exists"):
        return [f"NO_VOLUME: {inv['root']} not found"]
    out = [f"== volume {inv['root']}: {inv['disk']['free_gb']} GB free of {inv['disk']['total_gb']}"]
    out.append("== staged sets (items; by type; train/held-out; with video; with input; described)")
    for name, s in inv["staging"].items():
        out.append(f"  {name}: {s['items']}; {s['by_type']}; {s['by_split']}; video {s['with_video']}; "
                   f"input {s['with_input']}; described {s['with_description']}")
    if not inv["staging"]:
        out.append("  (none)")
    out.append("== zips (videos by type)")
    for name, z in inv["zips"].items():
        out.append(f"  {name}: {z.get('videos', '?')} videos {z.get('by_type', z.get('error'))} ({z.get('size_gb', '?')} GB)")
    v = inv["viton_hd"]
    out.append(f"== VITON-HD: downloaded={v['downloaded']} parquet={v['parquet_files']} images={v['images']} ({v['size_gb']} GB)")
    out.append("== datasets built: " + (", ".join(f"{k}={b.get('samples', '?')} samples" for k, b in inv["datasets_built"].items()) or "none"))
    out.append("== LoRA files")
    for name, l in inv["loras"].items():
        out.append(f"  {name}: {l['weights_mb']} MB sha256 {str(l['sha256'])[:16]} selection.json={l['selection_json']} checkpoints={l['checkpoints']}")
    out.append("== models (GB): " + (", ".join(f"{k} {g}" for k, g in inv["models"].items()) or "none"))
    out.append("== test sets (items): " + (", ".join(f"{k} {n}" for k, n in inv["eval_sets"].items()) or "none"))
    out.append(f"== own clips: {inv['own_clips']}  hijab clips: {inv['hijab_clips']}")
    if "clip_probe" in inv:
        out.append(f"== clip sample: {inv['clip_probe']}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the training root, /workspace/azyah-vton")
    ap.add_argument("--probe", type=int, default=0, help="measure this many random staged clips with ffprobe")
    a = ap.parse_args(argv)
    inv = inventory(a.root, a.probe)
    print("\n".join(summary_lines(inv)))
    print("INVENTORY_JSON=" + json.dumps(inv, sort_keys=True))
    return 0 if inv.get("exists") else 1


if __name__ == "__main__":
    sys.exit(main())

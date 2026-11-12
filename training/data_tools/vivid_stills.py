"""ViViD dress STILLS for the 0811 LoRA (step 3a, owner's Option A, 2026-11-05): one steady
frame per staged ViViD clip pair becomes a still pair, so data_tools.live_source can make its
input the live way (fixed garment photo) as a single steady frame.

  python3 -m data_tools.vivid_stills /workspace/azyah-vton/data/staging/vivid \
      /workspace/azyah-vton/data/staging/vivid_dress_stills --category dress --limit 400

Why stills: the P8 ViViD pseudo-source VIDEOS flicker (owner check 2026-11-04); a single frame
of the REAL clip is a steady, real target, and dresses (full length, hems, waist belts) exist in
no commercial-clean set we have. Glance has tops and bottoms only, upper-body crops.

- train split only; the frame is the middle of the first --max-frames frames (default 97)
- writes ref.<ext> (the product photo), target.png (the real frame), meta.json with
  source "vivid" (research-only licence; manifests/selection.json keep commercial_use_cleared
  false whenever these are in the data), from_clip and frame index; no source.* (live_source
  makes live_source.png)
- resumable: existing pairs are kept
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

from .staging import find, iter_pairs, stage_pair


def pick_frame(n_frames: int, max_frames: int = 97) -> int:
    """Middle frame of the first max_frames (the part P8/live edits cover)."""
    if n_frames <= 0:
        raise ValueError("clip has no frames")
    return (min(n_frames, max_frames) - 1) // 2


def read_frame(path: Path, index: int):
    import av
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i == index:
                return f.to_image().convert("RGB")
    raise ValueError(f"{path}: no frame {index}")


def count_frames(path: Path) -> int:
    import av
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        if st.frames:
            return int(st.frames)
        return sum(1 for _ in c.decode(st))


def build(staging_dir: str, out_dir: str, category: str = "dress", limit: int = 400, max_frames: int = 97) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"written": 0, "kept_existing": 0, "skipped": 0, "seen": {}}
    for d, meta in iter_pairs(Path(staging_dir)):
        if stats["written"] + stats["kept_existing"] >= limit:
            break
        tgt, ref = find(d, "target"), find(d, "ref")
        seen = f"{meta.get('split')}/{meta.get('category')}/{tgt.suffix if tgt else 'no-target'}/{'ref' if ref else 'no-ref'}"
        stats["seen"][seen] = stats["seen"].get(seen, 0) + 1
        if meta.get("split") != "train" or meta.get("category") != category or tgt is None or ref is None \
                or tgt.suffix != ".mp4":
            stats["skipped"] += 1
            continue
        key = f"{d.name}_still"
        if (out / key / "meta.json").exists():
            stats["kept_existing"] += 1
            continue
        idx = pick_frame(count_frames(tgt), max_frames)
        buf = io.BytesIO()
        read_frame(tgt, idx).save(buf, format="PNG")
        new_meta = {k: meta[k] for k in ("source", "split", "category", "garment_id", "person_id", "consent_id",
                                         "garment_name", "garment_description") if k in meta}
        new_meta.update({"from_clip": d.name, "frame": idx, "still_of_clip": True})
        pair = stage_pair(out, key, {"target.png": buf.getvalue()}, new_meta)
        shutil.copy(ref, pair / f"ref{ref.suffix}")
        stats["written"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("staging_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--category", default="dress")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max-frames", type=int, default=97)
    a = ap.parse_args()
    print(json.dumps(build(a.staging_dir, a.out_dir, a.category, a.limit, a.max_frames), indent=2))


if __name__ == "__main__":
    main()

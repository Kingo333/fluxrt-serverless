"""Frames where the garment MOVES, from every ViViD train clip of one category (fxreal5, owner 2026-11-11).

  python -m data_tools.vivid_motion STAGING/vivid OUT [--category dress] [--n 0] [--singles 2]
      [--gap-min 3] [--gap-max 8] [--every 2] [--size 432x768] [--model SEG.tflite]

fxreal3's pairs (data_tools.vivid_pairs) take two frames 3 apart at a random moment: often the model is
standing still, so the dress barely moves between them. For dresses the owner wants the drape and the
movement right, so this tool looks at the whole clip first:

- it segments every --every-th frame (MediaPipe multiclass selfie segmenter) and measures how much the
  clothes area changes between frames (1 - overlap of the two clothes masks);
- the PAIR is the two frames, --gap-min..--gap-max apart, where the dress changes MOST (a step, a turn,
  a swinging hem). fxreal3's pair term then asks the two answers' garment colours to differ only as much
  as the real frames do: steady colour while the dress moves;
- --singles more frames per clip, each as different as possible from the frames already picked
  (greedy farthest pose), so the model sees each dress in several poses.

Writes OUT/<clip>/ a_target.png, a_mask.png, b_target.png, b_mask.png (the pair, like vivid_pairs, so
build_real.vivid_pair_items reads it), s1_target.png, s1_mask.png, ... (singles), ref.jpg, and
meta.json (category, clip, frames, singles, motion score, split "train", and the clip's
garment_description when staging has one). Frames where the person fills under 8% of the picture or the
clothes area is under 2,000 px are never picked. --n 0 = every clip of the category.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLOTHES, BACKGROUND = 4, 0
MIN_PERSON_FRAC = 0.08
MIN_CLOTHES_PX = 2000  # at --size 432x768


def change(m1, m2) -> float:
    """1 - IoU of two clothes masks: 0 = identical, 1 = no overlap."""
    import numpy as np
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(1.0 - np.logical_and(m1, m2).sum() / union)


def pick_pair(masks: dict, gap_min: int, gap_max: int):
    """(t, t2, change) with gap_min <= t2 - t <= gap_max and the most change, or None."""
    idx = sorted(masks)
    best = None
    for i, t in enumerate(idx):
        for t2 in idx[i + 1:]:
            g = t2 - t
            if g < gap_min:
                continue
            if g > gap_max:
                break
            c = change(masks[t], masks[t2])
            if best is None or c > best[2]:
                best = (t, t2, c)
    return best


def pick_singles(masks: dict, chosen: list, k: int) -> list:
    """k more frames, each the one whose clothes area differs most from every frame already chosen."""
    picked, taken = [], list(chosen)
    for _ in range(max(0, k)):
        cands = [t for t in masks if t not in taken]
        if not cands:
            break
        t = max(cands, key=lambda c: (min(change(masks[c], masks[o]) for o in taken) if taken else 0.0, -c))
        picked.append(t)
        taken.append(t)
    return picked


def _cover(img, w: int, h: int):
    from PIL import Image
    s = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def usable(cat) -> bool:
    return float((cat != BACKGROUND).mean()) >= MIN_PERSON_FRAC and int((cat == CLOTHES).sum()) >= MIN_CLOTHES_PX * cat.size / (432 * 768)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vivid")
    ap.add_argument("out")
    ap.add_argument("--category", default="dress")
    ap.add_argument("--n", type=int, default=0, help="clips to write (0 = all of the category)")
    ap.add_argument("--singles", type=int, default=2)
    ap.add_argument("--gap-min", type=int, default=3)
    ap.add_argument("--gap-max", type=int, default=8)
    ap.add_argument("--every", type=int, default=2, help="score every N-th frame")
    ap.add_argument("--size", default="432x768")
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args(argv)
    import av
    import numpy as np
    from PIL import Image
    from data_tools.clothes_mask import Segmenter
    seg0 = Segmenter(Path(a.model))
    seg = lambda im: np.squeeze(seg0(im))  # noqa: E731
    w, h = (int(v) for v in a.size.lower().split("x"))
    sw, sh = w // 2, h // 2  # scoring size
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for d in sorted(p for p in Path(a.vivid).iterdir() if p.is_dir()):
        mp = d / "meta.json"
        if mp.exists() and (d / "target.mp4").exists() and (d / "ref.jpg").exists():
            meta = json.loads(mp.read_text())
            if meta.get("split") == "train" and meta.get("category") == a.category:
                clips.append((d, meta))
    random.Random(a.seed).shuffle(clips)
    stats = {"category": a.category, "clips": len(clips), "written": 0, "kept_existing": 0, "skipped": 0,
             "frames": 0, "mean_change": 0.0, "with_description": 0}
    changes = []
    for d, meta in clips:
        if a.n and stats["written"] + stats["kept_existing"] >= a.n:
            break
        dst = out / d.name
        if (dst / "meta.json").exists():
            stats["kept_existing"] += 1
            continue
        with av.open(str(d / "target.mp4")) as c:
            frames = [f.to_image().convert("RGB") for f in c.decode(video=0)]
        n = len(frames)
        lo, hi = min(4, n // 4), n - min(4, n // 4)
        masks = {}
        for t in range(lo, hi, max(1, a.every)):
            cat = seg(_cover(frames[t], sw, sh))
            if usable(cat):
                masks[t] = cat == CLOTHES
        pair = pick_pair(masks, a.gap_min, a.gap_max)
        if pair is None:
            stats["skipped"] += 1
            continue
        t, t2, score = pair
        singles = pick_singles(masks, [t, t2], a.singles)
        picks = [("a", t), ("b", t2)] + [(f"s{i}", s) for i, s in enumerate(singles, 1)]
        imgs = {tag: _cover(frames[f], w, h) for tag, f in picks}
        cats = {tag: seg(im) for tag, im in imgs.items()}
        if not (usable(cats["a"]) and usable(cats["b"])):
            stats["skipped"] += 1
            continue
        singles_kept = [s for (tag, s) in picks[2:] if usable(cats[tag])]
        dst.mkdir(parents=True, exist_ok=True)
        for tag, f in picks:
            if tag.startswith("s") and f not in singles_kept:
                continue
            imgs[tag].save(dst / f"{tag}_target.png")
            Image.fromarray(((cats[tag] == CLOTHES) * 255).astype(np.uint8), "L").save(dst / f"{tag}_mask.png")
        shutil.copy(d / "ref.jpg", dst / "ref.jpg")
        desc = meta.get("garment_description")
        stats["with_description"] += int(bool(desc))
        (dst / "meta.json").write_text(json.dumps({
            "category": a.category, "clip": d.name, "frames": [t, t2], "singles": singles_kept,
            "motion": round(score, 4), "split": "train", "source": "vivid",
            **({"garment_description": desc} if desc else {})}))
        stats["written"] += 1
        stats["frames"] += 2 + len(singles_kept)
        changes.append(score)
    stats["mean_change"] = round(sum(changes) / len(changes), 4) if changes else 0.0
    print(json.dumps(stats))
    return stats


def motion_items(root: Path):
    """vivid_motion output, for build_real: (clip, category, ref, description, pair [(t, m), (t, m)], singles [(t, m)])."""
    if not root or not Path(root).is_dir():
        return
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        mp = d / "meta.json"
        need = [d / f for f in ("a_target.png", "a_mask.png", "b_target.png", "b_mask.png", "ref.jpg")]
        if not mp.exists() or not all(f.exists() for f in need):
            continue
        meta = json.loads(mp.read_text())
        if meta.get("split", "train") != "train":
            continue
        singles = []
        for i in range(1, 10):
            t, m = d / f"s{i}_target.png", d / f"s{i}_mask.png"
            if t.exists() and m.exists():
                singles.append((t, m))
        yield (d.name, meta.get("category", "dress"), d / "ref.jpg", meta.get("garment_description"),
               [(need[0], need[1]), (need[2], need[3])], singles)


if __name__ == "__main__":
    main()

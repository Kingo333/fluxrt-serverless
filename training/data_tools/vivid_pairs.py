"""Two nearby frames per ViViD TRAIN clip, with garment masks, for fxreal3's same-clip colour term.

  python -m data_tools.vivid_pairs STAGING/vivid OUT [--n 300] [--gap 3] [--size 432x768] [--model SEG.tflite]

For up to --n train clips (target.mp4 + ref.jpg + meta.json with split "train"), writes OUT/<clip>/:
  a_target.png, b_target.png   frames t and t+gap of the real video (the answers), covered to --size
  a_mask.png, b_mask.png       the clothes area of each frame (MediaPipe multiclass selfie segmenter)
  ref.jpg, meta.json           the clip's garment photo; category, clip, frames, split
build_real.py then re-dresses both frames with the same other garment and pastes the real frame back
outside the mask. Frames where the person fills less than 8% of the picture (walk-in) are skipped.
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


def _cover(img, w: int, h: int):
    from PIL import Image
    s = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def pick_frames(n_frames: int, gap: int, rng: random.Random) -> tuple[int, int] | None:
    """A frame pair (t, t+gap) away from the first and last few frames."""
    lo, hi = min(4, n_frames // 4), n_frames - gap - min(4, n_frames // 4)
    if hi <= lo:
        return None
    t = rng.randrange(lo, hi)
    return t, t + gap


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vivid")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--size", default="432x768")
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    import av
    import numpy as np
    from PIL import Image
    from data_tools.clothes_mask import Segmenter
    seg0 = Segmenter(Path(a.model))
    seg = lambda im: np.squeeze(seg0(im))  # noqa: E731
    w, h = (int(v) for v in a.size.lower().split("x"))
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for d in sorted(p for p in Path(a.vivid).iterdir() if p.is_dir()):
        mp = d / "meta.json"
        if mp.exists() and (d / "target.mp4").exists() and (d / "ref.jpg").exists():
            meta = json.loads(mp.read_text())
            if meta.get("split") == "train":
                clips.append((d, meta))
    rng.shuffle(clips)
    stats = {"train_clips": len(clips), "written": 0, "kept_existing": 0, "skipped": 0}
    for d, meta in clips:
        if stats["written"] + stats["kept_existing"] >= a.n:
            break
        dst = out / d.name
        if (dst / "meta.json").exists():
            stats["kept_existing"] += 1
            continue
        with av.open(str(d / "target.mp4")) as c:
            frames = [f.to_image().convert("RGB") for f in c.decode(video=0)]
        fr = pick_frames(len(frames), a.gap, rng)
        if fr is None:
            stats["skipped"] += 1
            continue
        imgs = [_cover(frames[t], w, h) for t in fr]
        cats = [seg(im) for im in imgs]
        if min(float((c != BACKGROUND).mean()) for c in cats) < 0.08 or min(int((c == CLOTHES).sum()) for c in cats) < 2000:
            stats["skipped"] += 1
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for tag, im, c in zip("ab", imgs, cats):
            im.save(dst / f"{tag}_target.png")
            Image.fromarray(((c == CLOTHES) * 255).astype(np.uint8), "L").save(dst / f"{tag}_mask.png")
        shutil.copy(d / "ref.jpg", dst / "ref.jpg")
        (dst / "meta.json").write_text(json.dumps({"category": meta.get("category", "upper"), "clip": d.name,
                                                   "frames": list(fr), "split": "train", "source": "vivid"}))
        stats["written"] += 1
    print(json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()

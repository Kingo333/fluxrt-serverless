"""Frame strips for the owner: step through a clip frame by frame, every model side by side.

  python -m eval.frame_strip RESULTS_DIR OUT_DIR --models live0811 lora250 lora500 \
      [--clips vivid_test_000000 ...] [--every 6] [--height 200]

For each clip folder (input.mp4, ref.*, out_<model>.mp4) writes OUT_DIR/<clip>_frames.png:
- rows: input, then one row per model (the first model is the reference, e.g. live0811)
- columns: frame 0, every --every frames after it, then a "worst" column with that row's worst
  frame (the most change to the person vs the input: face box when a face is found, else the
  left/right background strips; the same checks as eval.metrics), framed in red. The same frame
  is also framed in red where it appears among the sampled columns.
The labels name the models (for looking, not for the blinded review).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LABEL_W = 110
HEAD_H = 22


def _frames(path: Path, h: int, max_frames: int) -> list:
    import av
    out = []
    with av.open(str(path)) as c:
        for k, f in enumerate(c.decode(c.streams.video[0])):
            if k >= max_frames:
                break
            im = f.to_image().convert("RGB")
            w = max(2, round(im.width * h / im.height))
            out.append(im.resize((w, h)))
    return out


def person_change(inp: list, out: list) -> list[float]:
    """Per-frame mean abs change of the person vs the input (face box, else background strips)."""
    from .metrics import border_mask, face_box
    a0 = np.asarray(inp[0], dtype=np.float32) / 255.0
    box = face_box(a0)
    strips = border_mask(a0.shape[:2])
    vals = []
    for i, o in zip(inp, out):
        d = np.abs(np.asarray(i, dtype=np.float32) - np.asarray(o.resize(i.size), dtype=np.float32)).mean(axis=2) / 255.0
        if box:
            x, y, w, h = box
            vals.append(float(d[y:y + h, x:x + w].mean()))
        else:
            vals.append(float(d[strips].mean()))
    return vals


def strip(d: Path, models: list[str], every: int, h: int, max_frames: int):
    from PIL import Image, ImageDraw
    inp = _frames(d / "input.mp4", h, max_frames)
    rows = [("input", inp, None)]
    for m in models:
        p = d / f"out_{m}.mp4"
        if p.exists():
            fr = _frames(p, h, max_frames)
            ch = person_change(inp, fr)
            rows.append((m, fr, int(np.argmax(ch)) if ch else None))
    if len(rows) < 2:
        return None
    n = min(len(r[1]) for r in rows)
    cols = list(range(0, n, max(1, every)))
    fw = inp[0].width
    W = LABEL_W + (len(cols) + 1) * (fw + 4)
    H = HEAD_H + len(rows) * (h + 4)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for ci, k in enumerate(cols):
        dr.text((LABEL_W + ci * (fw + 4) + 4, 5), f"frame {k}", fill=(0, 0, 0))
    dr.text((LABEL_W + len(cols) * (fw + 4) + 4, 5), "worst", fill=(200, 0, 0))
    for ri, (name, fr, worst) in enumerate(rows):
        y = HEAD_H + ri * (h + 4)
        dr.text((4, y + h // 2 - 6), name[:16], fill=(0, 0, 0))
        for ci, k in enumerate(cols):
            x = LABEL_W + ci * (fw + 4)
            sheet.paste(fr[k], (x, y))
            if worst is not None and k == worst:
                dr.rectangle([x - 2, y - 2, x + fw + 1, y + h + 1], outline=(220, 0, 0), width=3)
        if worst is not None:
            x = LABEL_W + len(cols) * (fw + 4)
            sheet.paste(fr[min(worst, n - 1)], (x, y))
            dr.rectangle([x - 2, y - 2, x + fw + 1, y + h + 1], outline=(220, 0, 0), width=3)
            dr.text((x + 3, y + 3), f"#{worst}", fill=(255, 255, 255))
    return sheet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--max-frames", type=int, default=97)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.results_dir)
    clips = [root / c for c in a.clips] if a.clips else sorted(p for p in root.iterdir() if p.is_dir())
    n = 0
    for d in clips:
        if not (d / "input.mp4").exists():
            continue
        sheet = strip(d, a.models, a.every, a.height, a.max_frames)
        if sheet is not None:
            sheet.save(out / f"{d.name}_frames.png")
            print(f"{d.name}: {out / (d.name + '_frames.png')}", flush=True)
            n += 1
    return n


if __name__ == "__main__":
    main()

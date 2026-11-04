"""Side-by-side review videos for a P9e results folder (CPU only).

  python -m eval.grid RESULTS_DIR OUT_DIR [--models base exp2] [--height 320] [--max-frames 96]

For every clip folder (input.mp4, ref.jpg, out_<model>.mp4, target.mp4) writes
OUT_DIR/<clip>.mp4 with the panels  input | garment | out_<model>... | target,
each labelled, and OUT_DIR/all.mp4 with every clip one after another.
The labels name the models, so this is for looking, not for the blinded review (ab_sheet.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

LABEL_H = 28


def _read(path: Path, max_frames: int):
    import av
    frames, fps = [], 16
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        if st.average_rate:
            fps = float(st.average_rate)
        for f in c.decode(st):
            frames.append(f.to_image().convert("RGB"))
            if len(frames) >= max_frames:
                break
    return frames, fps


def _fit(im, h):
    w = max(2, round(im.width * h / im.height / 2) * 2)
    return im.resize((w, h))


def _panel(im, label, h):
    from PIL import Image, ImageDraw
    im = _fit(im, h)
    out = Image.new("RGB", (im.width, h + LABEL_H), (255, 255, 255))
    out.paste(im, (0, LABEL_H))
    ImageDraw.Draw(out).text((6, 7), label, fill=(0, 0, 0))
    return out


def compose(panels_per_frame: list[list[tuple]], h: int):
    """panels_per_frame: per frame, a list of (PIL image, label). Returns stacked frames."""
    from PIL import Image
    frames = []
    for panels in panels_per_frame:
        parts = [_panel(im, label, h) for im, label in panels]
        w = sum(p.width for p in parts)
        canvas = Image.new("RGB", (w + w % 2, h + LABEL_H), (255, 255, 255))
        x = 0
        for p in parts:
            canvas.paste(p, (x, 0))
            x += p.width
        frames.append(canvas)
    return frames


def clip_frames(d: Path, models: list[str], h: int, max_frames: int):
    from PIL import Image
    inp, fps = _read(d / "input.mp4", max_frames)
    ref_p = next((d / n for n in ("ref.jpg", "ref.png") if (d / n).exists()), None)
    ref = Image.open(ref_p).convert("RGB") if ref_p else None
    outs = [(m, _read(d / f"out_{m}.mp4", max_frames)[0]) for m in models if (d / f"out_{m}.mp4").exists()]
    tgt = _read(d / "target.mp4", max_frames)[0] if (d / "target.mp4").exists() else []
    if not outs:
        return [], fps
    n = min([len(inp)] + [len(f) for _, f in outs] + ([len(tgt)] if tgt else []))
    per_frame = []
    for k in range(n):
        p = [(inp[k], f"{d.name}  input")]
        if ref is not None:
            p.append((ref, "garment"))
        p += [(f[k], m) for m, f in outs]
        if tgt:
            p.append((tgt[k], "real (target)"))
        per_frame.append(p)
    return compose(per_frame, h), fps


class Writer:
    """H.264 mp4 writer; frames of another size are resized to the first frame's size."""

    def __init__(self, path: Path, fps: float, size):
        import av
        self.c = av.open(str(path), "w")
        self.st = self.c.add_stream("libx264", rate=max(1, round(fps)))
        self.st.width, self.st.height, self.st.pix_fmt = size[0], size[1], "yuv420p"
        self.st.options = {"crf": "23"}
        self.size = size

    def add(self, im) -> None:
        import av
        if im.size != self.size:
            im = im.resize(self.size)
        for pkt in self.st.encode(av.VideoFrame.from_image(im)):
            self.c.mux(pkt)

    def close(self) -> None:
        for pkt in self.st.encode():
            self.c.mux(pkt)
        self.c.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--models", nargs="+", default=["base", "exp2"])
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--max-frames", type=int, default=96)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    every, n_all = None, 0
    for d in sorted(p for p in Path(a.results_dir).iterdir() if p.is_dir()):
        frames, fps = clip_frames(d, a.models, a.height, a.max_frames)
        if not frames:
            continue
        w = Writer(out / f"{d.name}.mp4", fps, frames[0].size)
        every = every or Writer(out / "all.mp4", fps, frames[0].size)
        for f in frames:
            w.add(f)
            every.add(f)
        w.close()
        n_all += len(frames)
        print(f"{d.name}: {len(frames)} frames -> {out / (d.name + '.mp4')}", flush=True)
    if every:
        every.close()
        print(f"all clips: {n_all} frames -> {out / 'all.mp4'}")
    else:
        print("no clip folders with out_<model>.mp4 found")

if __name__ == "__main__":
    main()

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


def _fps(path: Path) -> float:
    import av
    with av.open(str(path)) as c:
        r = c.streams.video[0].average_rate
        return float(r) if r else 16.0


def _read(path: Path, max_frames: int):
    """Yields frames one at a time (CPU Pods have 512 MB)."""
    import av
    with av.open(str(path)) as c:
        for k, f in enumerate(c.decode(c.streams.video[0])):
            if k >= max_frames:
                break
            yield f.to_image().convert("RGB")


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


def compose(panels, h: int):
    """panels: one frame's list of (PIL image, label) -> the stacked frame."""
    from PIL import Image
    parts = [_panel(im, label, h) for im, label in panels]
    w = sum(p.width for p in parts)
    canvas = Image.new("RGB", (w + w % 2, h + LABEL_H), (255, 255, 255))
    x = 0
    for p in parts:
        canvas.paste(p, (x, 0))
        x += p.width
    return canvas


def clip_frames(d: Path, models: list[str], h: int, max_frames: int):
    """Yields the grid frames of one clip; stops at the shortest video."""
    from PIL import Image
    ref_p = next((d / n for n in ("ref.jpg", "ref.png") if (d / n).exists()), None)
    ref = _fit(Image.open(ref_p).convert("RGB"), h) if ref_p else None
    names = [m for m in models if (d / f"out_{m}.mp4").exists()]
    if not names:
        return
    streams = [_read(d / "input.mp4", max_frames)] + [_read(d / f"out_{m}.mp4", max_frames) for m in names]
    labels = [f"{d.name}  input"] + names
    if (d / "target.mp4").exists():
        streams.append(_read(d / "target.mp4", max_frames))
        labels.append("real (target)")
    for frames in zip(*streams):
        panels = [(frames[0], labels[0])]
        if ref is not None:
            panels.append((ref, "garment"))
        panels += list(zip(frames[1:], labels[1:]))
        yield compose(panels, h)


class Writer:
    """H.264 mp4 writer; frames of another size are resized to the first frame's size."""

    def __init__(self, path: Path, fps: float, size):
        import av
        self.c = av.open(str(path), "w")
        self.st = self.c.add_stream("libx264", rate=max(1, round(fps)))
        self.st.width, self.st.height, self.st.pix_fmt = size[0], size[1], "yuv420p"
        # Small look-ahead and few threads: two encoders must fit a 512 MB CPU Pod.
        self.st.options = {"crf": "23", "preset": "veryfast", "x264-params": "rc-lookahead=4:threads=2:sync-lookahead=0"}
        self.st.codec_context.thread_count = 2
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


def still_grids(results_dir: Path, out: Path, models: list[str], h: int, max_items: int, require_all: bool) -> int:
    """Still items (input.png, ref.*, out_<model>.png[, target.png]): one labelled PNG per item and
    all.png with the items stacked, for a quick owner review."""
    from PIL import Image
    rows = []
    for d in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if len(rows) >= max_items or not (d / "input.png").exists():
            continue
        names = [m for m in models if (d / f"out_{m}.png").exists()]
        if not names or (require_all and len(names) < len(models)):
            continue
        ref_p = next((d / n for n in ("ref.jpg", "ref.png") if (d / n).exists()), None)
        panels = [(Image.open(d / "input.png").convert("RGB"), f"{d.name}  input")]
        if ref_p:
            panels.append((Image.open(ref_p).convert("RGB"), "item"))
        panels += [(Image.open(d / f"out_{m}.png").convert("RGB"), m) for m in names]
        if (d / "target.png").exists():
            panels.append((Image.open(d / "target.png").convert("RGB"), "real (target)"))
        row = compose(panels, h)
        row.save(out / f"{d.name}.png")
        rows.append(row)
    if rows:
        w = max(r.width for r in rows)
        sheet = Image.new("RGB", (w, sum(r.height for r in rows)), (255, 255, 255))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        sheet.save(out / "all.png")
        print(f"still grid: {len(rows)} items -> {out / 'all.png'}")
    return len(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--models", nargs="+", default=["base", "exp2"])
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--max-frames", type=int, default=96)
    ap.add_argument("--require-all", action="store_true", help="skip clips missing any of --models")
    ap.add_argument("--max-items", type=int, default=10, help="items in the review (owner: 5-10)")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    still_grids(Path(a.results_dir), out, a.models, a.height, a.max_items, a.require_all)
    every, n_all, n_clips = None, 0, 0
    for d in sorted(p for p in Path(a.results_dir).iterdir() if p.is_dir()):
        if not (d / "input.mp4").exists() or n_clips >= a.max_items:
            continue
        if a.require_all and not all((d / f"out_{m}.mp4").exists() for m in a.models):
            continue
        w, n = None, 0
        for f in clip_frames(d, a.models, a.height, a.max_frames):
            if w is None:
                fps = _fps(d / "input.mp4")
                w = Writer(out / f"{d.name}.mp4", fps, f.size)
                every = every or Writer(out / "all.mp4", fps, f.size)
            w.add(f)
            every.add(f)
            n += 1
        if w is None:
            continue
        w.close()
        n_all += n
        n_clips += 1
        print(f"{d.name}: {n} frames -> {out / (d.name + '.mp4')}", flush=True)
    if every:
        every.close()
        print(f"all clips: {n_all} frames -> {out / 'all.mp4'}")
    elif not any(out.glob("*.png")):
        print("no clip or still items with out_<model> found")

if __name__ == "__main__":
    main()

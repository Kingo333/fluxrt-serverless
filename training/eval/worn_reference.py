"""The same garments as REALLY worn (2026-11-22): what do our measures say about a real video?

Every test clip's garment photo comes from another ViViD clip (the donor), and that clip's own video shows a model
really wearing the garment. This tool lays those real videos out like a render, so the usual scorers can measure
them: how far a real video of the garment is from its photo in colour, detail, light-dark spread and length, and
how much a real garment flickers and wobbles. That is the level a perfect try-on would reach, so favourite #2 and
every model can be read against it (eval.yardstick --worn): a measure where favourite #2 already stands at the
real videos' level has nothing left to win, and a measure whose "ideal" differs from what real videos show (the
fabric score calls the photo's own light-dark spread ideal) is read against the real level instead.

  python -m eval.worn_reference ROOT --sets fx6test dress10 fx12test [--frames 49] [--size 288x512] [--out DIR]

For each item of ROOT/eval/fxa/<set>.json whose garment photo sits beside a video (the donor's ref.jpg and
target.mp4), DIR/<set>/<key>/ (default DIR: ROOT/eval/fxa/worn) gets
  input.mp4 and out_worn.mp4   the donor's video, cut and sized like a test clip (the first --frames frames,
                               filling --size); "camera" and "result" are the same real video
  ref.*                        the garment photo, as the test clip has it
  item.json                    the test item, plus "worn_from"
Items without a donor video (the owner's own photos) are left out. Nothing is rendered; folders that are complete
are kept as they are. Prints one WORN_SETS line.

The movement and flicker of a real video are another person's movement than the test clip's, so those are read
as levels over a whole set, not clip by clip; colour, detail, spread and length belong to the garment itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAME = "worn"


def donor_video(item: dict) -> Path | None:
    """The video beside the item's garment photo (ViViD: <clip>/ref.jpg and <clip>/target.mp4), or None."""
    ref = Path(str(item.get("ref") or ""))
    vid = ref.parent / "target.mp4"
    return vid if ref.name and ref.exists() and vid.exists() else None


def build(root: Path, sets: list[str], frames: int = 49, size: tuple[int, int] = (288, 512), out: Path | None = None) -> dict:
    from eval.generate_fluxrt import _cover, _read_frames, _write_mp4
    fxa = root / "eval/fxa"
    out = out or fxa / "worn"
    w, h = size
    report = {}
    for s in sets:
        try:
            items = json.loads((fxa / f"{s}.json").read_text())
        except (OSError, ValueError):
            continue
        made = kept = skipped = 0
        for it in items if isinstance(items, list) else []:
            vid = donor_video(it) if isinstance(it, dict) and it.get("key") else None
            if vid is None:
                skipped += 1
                continue
            dest = out / s / str(it["key"])
            if (dest / f"out_{NAME}.mp4").exists() and (dest / "input.mp4").exists() and (dest / "item.json").exists():
                kept += 1
                continue
            fr = [_cover(f, w, h) for f in _read_frames(vid, frames)]
            if len(fr) < 2:
                skipped += 1
                continue
            dest.mkdir(parents=True, exist_ok=True)
            ref = Path(it["ref"])
            shutil.copy(ref, dest / f"ref{ref.suffix or '.jpg'}")
            (dest / "item.json").write_text(json.dumps({**it, "worn_from": vid.parent.name}))
            _write_mp4(dest / "input.mp4", fr)
            _write_mp4(dest / f"out_{NAME}.mp4", fr)          # written last: its presence says the folder is complete
            made += 1
        if made or kept:
            report[s] = {"clips": made + kept, "new": made, "without_a_video": skipped}
    print("WORN_SETS " + (" | ".join(f"{s}: {r['clips']} real videos of the test garments ({r['new']} new"
                                     + (f", {r['without_a_video']} garments have no video" if r["without_a_video"] else "") + ")"
                                     for s, r in report.items()) or "none: no test garment has a video of its own"))
    return report


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="/workspace/azyah-vton")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--size", default="288x512")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    w, h = (int(x) for x in a.size.lower().split("x"))
    return build(Path(a.root), a.sets, a.frames, (w, h), Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()

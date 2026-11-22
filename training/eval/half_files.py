"""Half-written files a stopped Pod leaves behind (2026-11-22).

A Pod that is stopped while it writes a video or a segmentation file (a restart by hand, RunPod stopping it when
the balance is used up) leaves half a file under its final name. The renderer then takes the clip for rendered and
the scorers break on it, twice, and the whole model is left out. This tool opens the files that were written
lately and moves every one that cannot be read aside, so it is simply made again:

  python -m eval.half_files RESULTS_DIR [MORE_DIRS...] [--yard YARD] [--hours 72] [--min-age 0] [--no-videos] [--dry-run]

- *.mp4 must open and hold at least two video packets; *.npz must open as an archive. Only files changed in the
  last --hours hours are opened (older ones were read by an earlier scoring), and none younger than --min-age
  seconds (a file that is being written right now is not a half file). --no-videos: the *.npz only.
- An unreadable file is renamed to <name>.bad: nothing is deleted.
- --yard: the yardstick's folder. When the unreadable file is a model's video or segmentation
  (RESULTS/<set>/<clip>/out_<model>.mp4 or seg_<model>.npz) and YARD/<model>/ exists, that model's marks
  (.rendered, .scored, .failed, the counts of tries) and its scores of that set are set aside too (<name>.old), so
  the clip is rendered again and the set scored again.
- --dry-run only lists.
Prints one HALVES line. Without the video library the videos are left unchecked, and the line says so.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

MARKS = (".rendered", ".scored", ".failed", ".render_tries", ".score_tries")
SCORES = ("garment.json", "face_skin.json", "stability.json")


def readable(p: Path, av=None) -> bool:
    """A video with at least two packets, or an archive that opens."""
    try:
        if p.suffix == ".mp4":
            with av.open(str(p)) as c:
                n = 0
                for pkt in c.demux(video=0):
                    n += pkt.size > 0
                    if n >= 2:
                        return True
            return False
        import numpy as np
        with np.load(str(p)) as z:
            z.files
        return True
    except Exception:  # noqa: BLE001 - whatever the reader says, the file cannot be used
        return False


def _aside(p: Path, ext: str) -> None:
    if p.exists():
        os.replace(p, p.with_name(p.name + ext))


def check(roots: list[Path], yard: Path | None = None, hours: float = 72, min_age: float = 0, dry_run: bool = False,
          videos: bool = True) -> dict:
    av = None
    if videos:
        try:
            import av
        except ImportError:
            av = None
    now = time.time()
    opened, bad, again = 0, [], set()
    for root in roots:
        for d, _, files in os.walk(root, followlinks=False):
            for f in files:
                if not (f.endswith(".npz") or (f.endswith(".mp4") and av is not None)):
                    continue
                p = Path(d) / f
                try:
                    age = now - p.stat().st_mtime
                except OSError:
                    continue
                if age > hours * 3600 or age < min_age:
                    continue
                opened += 1
                if readable(p, av):
                    continue
                bad.append(str(p))
                if dry_run:
                    continue
                _aside(p, ".bad")
                stem = p.stem
                model = stem[4:] if stem.startswith(("out_", "seg_")) else None
                job = yard / model if yard is not None and model else None
                if job is not None and (job / "meta.json").exists():
                    again.add(model)
                    for m in MARKS:
                        _aside(job / m, ".old")
                    if f.endswith(".mp4"):                       # its scores of that set were made without this clip
                        for s in SCORES:
                            _aside(job / p.parent.parent.name / s, ".old")
    print(f"HALVES {opened} files written in the last {hours:g} hours opened"
          + ("" if av is not None else " (videos NOT checked" + (": no video library here)" if videos else ")"))
          + f": {len(bad)} cannot be read"
          + ((" (listed only)" if dry_run else " and were moved aside to be made again") + ": "
             + ", ".join(bad[:6]) + (" ..." if len(bad) > 6 else "") if bad else ""))
    return {"opened": opened, "bad": bad, "again": sorted(again), "videos_checked": av is not None}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--yard")
    ap.add_argument("--hours", type=float, default=72)
    ap.add_argument("--min-age", type=float, default=0, help="seconds: younger files are being written, not half")
    ap.add_argument("--no-videos", action="store_true", help="the segmentation files only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return check([Path(r) for r in a.roots if Path(r).is_dir()], Path(a.yard) if a.yard else None, a.hours, a.min_age, a.dry_run,
                 not a.no_videos)


if __name__ == "__main__":
    main()

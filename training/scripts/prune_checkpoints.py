#!/usr/bin/env python3
"""Keep training checkpoints from filling the volume (review R1).

Upstream save_checkpoint() writes every save to <run>/checkpoints/global_step<N>/
(model + optimizer, about 100-160 GB each for Stage 1) and never deletes any.

  python3 prune_checkpoints.py RUN_DIR --keep-last 2 [--dry-run]
  python3 prune_checkpoints.py RUN_DIR --keep-last 2 --watch 300   # every 5 min, during training

Kept: the newest N complete saves, plus any save containing a file named KEEP
(mark a good checkpoint with `touch .../global_step500/KEEP`, e.g. the best eval).
A save still being written (its step_<N>.pth missing or changed in the last 2 minutes)
is never touched. Deletes whole global_step<N> folders only.
"""
from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

STEP = re.compile(r"^global_step(\d+)$")


def saves(run_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for d in run_dir.rglob("global_step*"):
        m = STEP.match(d.name)
        if m and d.is_dir():
            found.append((int(m.group(1)), d))
    return sorted(found)


def complete(step: int, d: Path, settle_s: float = 120.0) -> bool:
    f = d / f"step_{step}.pth"
    return f.is_file() and time.time() - f.stat().st_mtime > settle_s


def prune(run_dir: str, keep_last: int, dry_run: bool = False, settle_s: float = 120.0) -> list[str]:
    all_saves = saves(Path(run_dir))
    done = [(s, d) for s, d in all_saves if complete(s, d, settle_s)]
    keep = {d for _, d in done[-keep_last:]} | {d for _, d in all_saves if (d / "KEEP").exists()}
    removed = []
    for s, d in done:
        if d in keep:
            continue
        removed.append(str(d))
        if not dry_run:
            shutil.rmtree(d)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--keep-last", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watch", type=int, help="repeat every N seconds")
    a = ap.parse_args()
    if a.keep_last < 1:
        raise SystemExit("--keep-last must be at least 1 (resume needs the latest save)")
    while True:
        for r in prune(a.run_dir, a.keep_last, a.dry_run):
            print(("would remove " if a.dry_run else "removed ") + r, flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()

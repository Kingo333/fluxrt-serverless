"""Blinded A/B review: reviewers see two unlabeled results per clip and pick the better one.

  python3 -m eval.ab_sheet make RESULTS_DIR model_a model_b OUT_DIR [--seed 7]
      -> OUT_DIR/review.csv (clip, left, right, choice, notes) + OUT_DIR/key.json (keep hidden)
         and copies the two outputs as left/right files per clip
  python3 -m eval.ab_sheet score OUT_DIR/review.csv OUT_DIR/key.json
      -> win rate per model (choice: left | right | same)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


def make(results_dir: str, model_a: str, model_b: str, out_dir: str, seed: int = 7) -> int:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    for d in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        a = next(d.glob(f"out_{model_a}.*"), None)
        b = next(d.glob(f"out_{model_b}.*"), None)
        if not (a and b):
            continue
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        cdir = out / d.name
        cdir.mkdir(exist_ok=True)
        shutil.copy(left, cdir / f"left{left.suffix}")
        shutil.copy(right, cdir / f"right{right.suffix}")
        for extra in ("ref.jpg", "ref.png", "input.mp4", "input.png"):
            if (d / extra).exists():
                shutil.copy(d / extra, cdir / extra)
        key[d.name] = {"left": model_a if left == a else model_b, "right": model_b if left == a else model_a}
        rows.append({"clip": d.name, "choice": "", "notes": ""})
    with open(out / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "choice", "notes"])
        w.writeheader()
        w.writerows(rows)
    (out / "key.json").write_text(json.dumps(key, indent=2))
    return len(rows)


def score(review_csv: str, key_json: str) -> dict:
    key = json.loads(Path(key_json).read_text())
    wins: dict[str, int] = {}
    same = answered = 0
    for row in csv.DictReader(open(review_csv, newline="")):
        choice = (row.get("choice") or "").strip().lower()
        if choice not in ("left", "right", "same"):
            continue
        answered += 1
        if choice == "same":
            same += 1
            continue
        winner = key[row["clip"]][choice]
        wins[winner] = wins.get(winner, 0) + 1
    return {"answered": answered, "same": same, "wins": wins,
            "win_rate": {m: round(w / answered, 3) for m, w in wins.items()} if answered else {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("results_dir"); m.add_argument("model_a"); m.add_argument("model_b"); m.add_argument("out_dir")
    m.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("review_csv"); s.add_argument("key_json")
    a = ap.parse_args()
    if a.cmd == "make":
        print(f"{make(a.results_dir, a.model_a, a.model_b, a.out_dir, a.seed)} clips ready for blinded review")
    else:
        print(json.dumps(score(a.review_csv, a.key_json), indent=2))


if __name__ == "__main__":
    main()

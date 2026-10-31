"""Our own consented recordings -> staged pairs.

  python3 -m data_tools.ingest_own RAW_DIR STAGING_DIR --test-people p05 --test-garments g17 g18 g19

RAW_DIR layout (see RECORDING_GUIDE.md):
  consent.csv                    person_id,consent_id,train,test,commercial   (yes/no)
  garments/<garment_id>.jpg      flat photo of each garment (white/plain background)
  garments/garments.csv          garment_id,category[,name][,description]     (category: upper|lower|dress)
  clips/<person_id>/<garment_id>/<anything>.mp4    the person WEARING that garment

Rules (enforced, not advisory):
  - no consent row, or train=no and test=no -> the person's clips are skipped
  - test people/garments go to the test split only; if a test clip shares a garment
    or person with training, the garment/person is held out of training entirely
  - train=yes is required for the train split, test=yes for the test split
  - commercial=no -> the sample is tagged research-only (still usable for R&D runs)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .staging import stage_pair

YES = {"yes", "y", "true", "1"}


def _yes(v: str | None) -> bool:
    return (v or "").strip().lower() in YES


def ingest(raw_dir: str, staging_dir: str, test_people: set[str], test_garments: set[str]) -> dict:
    raw = Path(raw_dir)
    consent = {r["person_id"]: r for r in csv.DictReader(open(raw / "consent.csv", newline=""))}
    garments = {r["garment_id"]: r for r in csv.DictReader(open(raw / "garments" / "garments.csv", newline=""))}
    report = {"staged": {"train": 0, "test": 0}, "skipped": []}
    for clip in sorted((raw / "clips").rglob("*.mp4")):
        person, garment = clip.parent.parent.name, clip.parent.name
        c, g = consent.get(person), garments.get(garment)
        photo = next((raw / "garments" / f"{garment}.{e}" for e in ("jpg", "jpeg", "png")
                      if (raw / "garments" / f"{garment}.{e}").exists()), None)
        why = (None if c else "no consent row") or (None if g else "garment not in garments.csv") or \
              (None if photo else "no garment photo")
        if why:
            report["skipped"].append(f"{clip.relative_to(raw)}: {why}")
            continue
        split = "test" if (person in test_people or garment in test_garments) else "train"
        if split == "train" and not _yes(c.get("train")):
            report["skipped"].append(f"{clip.relative_to(raw)}: consent does not cover training")
            continue
        if split == "test" and not _yes(c.get("test")):
            report["skipped"].append(f"{clip.relative_to(raw)}: consent does not cover testing")
            continue
        if g["category"] not in ("upper", "lower", "dress"):
            report["skipped"].append(f"{clip.relative_to(raw)}: bad category {g['category']!r}")
            continue
        key = f"own_{split}_{person}_{garment}_{clip.stem}".replace(".", "_").replace("-", "_")
        stage_pair(Path(staging_dir), key, {f"ref{photo.suffix.lower().replace('.jpeg', '.jpg')}": photo.read_bytes(),
                                            "target.mp4": clip.read_bytes()}, {
            "source": "own" if _yes(c.get("commercial")) else "own_research", "split": split, "category": g["category"], "task": "iv2v",
            "garment_id": f"own-{garment}", "person_id": f"own-{person}", "consent_id": c["consent_id"],
            "commercial_ok": _yes(c.get("commercial")),
            "garment_name": g.get("name") or None, "garment_description": g.get("description") or None,
        })
        report["staged"][split] += 1
    # Leakage guard: a training sample may not share a person or garment with any test sample.
    staged = [json.loads(p.read_text()) for p in Path(staging_dir).glob("own_*/meta.json")]
    test_ids = {m["person_id"] for m in staged if m["split"] == "test"} | {m["garment_id"] for m in staged if m["split"] == "test"}
    leaks = [m for m in staged if m["split"] == "train" and (m["person_id"] in test_ids or m["garment_id"] in test_ids)]
    report["leakage"] = len(leaks)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("staging_dir")
    ap.add_argument("--test-people", nargs="*", default=[])
    ap.add_argument("--test-garments", nargs="*", default=[])
    a = ap.parse_args()
    r = ingest(a.raw_dir, a.staging_dir, set(a.test_people), set(a.test_garments))
    print(json.dumps(r, indent=2))
    if r["leakage"]:
        raise SystemExit(f"{r['leakage']} training clips share a person or garment with the test set: "
                         "add those people/garments to --test-people/--test-garments or re-record")


if __name__ == "__main__":
    main()

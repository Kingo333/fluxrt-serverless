"""ViViD (R&D ONLY, see sources.py) -> staged video pairs.

Layout (github.com/alibaba-yuanjing-aigclab/ViViD README): videos/*.mp4 (person
footage), cloth/*.jpg (garment photo), cloth_mask/, agnostic*/, densepose/ (unused:
we train on pseudo-pairs, plan G1).

How a video maps to its garment is not documented; the repo's sample files do not
share names. So first inspect the extracted folder on the CPU Pod:

  python3 -m data_tools.convert_vivid inspect EXTRACTED_DIR
      -> counts, name patterns, how many videos have a same-stem cloth image

then convert with the pairing that inspect shows works:

  python3 -m data_tools.convert_vivid convert EXTRACTED_DIR STAGING_DIR --pairing stem [--test-frac 0.2]
  python3 -m data_tools.convert_vivid convert EXTRACTED_DIR STAGING_DIR --pairing csv --pairs pairs.csv

pairs.csv columns: video,cloth[,category] (paths relative to EXTRACTED_DIR).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from .staging import stage_pair


# Garment photo folder names: our extract writes cloth/; the released ViViD.zip
# uses images/ (e.g. upper_body/images/1221886_in_xl.jpg next to videos/1221886_detail.mp4).
CLOTH_DIRS = ("cloth", "images")


def pair_id(stem: str) -> str:
    """Pairing key: the leading product number (1221886_detail -> 1221886), else the stem."""
    m = re.match(r"(\d+)", stem)
    return m.group(1) if m else stem


def _cloth_dir(root: Path) -> Path | None:
    for name in CLOTH_DIRS:
        if (root / name).is_dir():
            return root / name
    return None


def _find_roots(d: Path) -> list[Path]:
    """Every folder that holds videos/ and cloth/ or images/ (e.g. upper_body/, lower_body/, dresses/)."""
    roots = [c for c in [d, *sorted(p for p in d.rglob("*") if p.is_dir())]
             if (c / "videos").is_dir() and _cloth_dir(c) is not None]
    if not roots:
        raise SystemExit(f"no folder with videos/ and cloth/ (or images/) under {d}")
    return roots


def _find_root(d: Path) -> Path:
    return _find_roots(d)[0]


def _pattern(name: str) -> str:
    return re.sub(r"\d+", "#", name)


def inspect(extracted: str) -> dict:
    root = _find_root(Path(extracted))
    videos = sorted((root / "videos").rglob("*.mp4"))
    cloths = sorted(p for p in _cloth_dir(root).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    cloth_stems = {pair_id(p.stem) for p in cloths}
    report = {
        "root": str(root),
        "videos": len(videos),
        "cloth_images": len(cloths),
        "video_name_patterns": Counter(_pattern(p.name) for p in videos).most_common(5),
        "cloth_name_patterns": Counter(_pattern(p.name) for p in cloths).most_common(5),
        "video_subfolders": Counter(p.parent.relative_to(root / "videos").as_posix() for p in videos).most_common(10),
        "videos_with_same_stem_cloth": sum(pair_id(p.stem) in cloth_stems for p in videos),
        "other_files": sorted(str(p.relative_to(root)) for p in root.iterdir() if p.is_file())[:20],
        "examples": {"videos": [str(p.relative_to(root)) for p in videos[:5]],
                     "cloth": [str(p.relative_to(root)) for p in cloths[:5]]},
    }
    return report


def _category(path: Path) -> str:
    s = path.as_posix().lower()
    if "dress" in s:
        return "dress"
    if "lower" in s or "pant" in s or "skirt" in s:
        return "lower"
    return "upper"


def _is_test(key: str, frac: float) -> bool:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def convert(extracted: str, staging: str, pairing: str, pairs_csv: str | None = None,
            test_frac: float = 0.2, limit: int | None = None) -> dict:
    roots = _find_roots(Path(extracted))
    root = Path(extracted)
    pairs: list[tuple[Path, Path, str]] = []
    if pairing == "stem":
        for r in roots:
            cloth_by_stem = {pair_id(p.stem): p for p in _cloth_dir(r).rglob("*")
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
            for v in sorted((r / "videos").rglob("*.mp4")):
                c = cloth_by_stem.get(pair_id(v.stem))
                if c is not None:
                    pairs.append((v, c, _category(v)))
    elif pairing == "csv":
        with open(pairs_csv, newline="") as f:
            for row in csv.DictReader(f):
                v, c = root / row["video"], root / row["cloth"]
                pairs.append((v, c, row.get("category") or _category(v)))
    else:
        raise SystemExit("pairing must be stem or csv")
    out = Path(staging)
    n = 0
    for v, c, cat in pairs:
        cloth = c.read_bytes()
        garment_id = "vivid-" + hashlib.sha1(cloth).hexdigest()[:16]
        # Split by garment so no garment is in both train and test.
        split = "test" if _is_test(garment_id, test_frac) else "train"
        key = f"vivid_{split}_{n:06d}"
        stage_pair(out, key, {"ref.jpg": cloth, "target.mp4": v.read_bytes()}, {
            "source": "vivid", "split": split, "category": cat, "garment_id": garment_id,
            "person_id": None, "task": "iv2v", "orig_video": str(v.relative_to(root)),
        })
        n += 1
        if limit and n >= limit:
            break
    return {"pairs_found": len(pairs), "staged": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("extracted")
    c = sub.add_parser("convert")
    c.add_argument("extracted")
    c.add_argument("staging")
    c.add_argument("--pairing", choices=["stem", "csv"], required=True)
    c.add_argument("--pairs")
    c.add_argument("--test-frac", type=float, default=0.2)
    c.add_argument("--limit", type=int)
    a = ap.parse_args()
    import json
    if a.cmd == "inspect":
        print(json.dumps(inspect(a.extracted), indent=2))
    else:
        print(json.dumps(convert(a.extracted, a.staging, a.pairing, a.pairs, a.test_frac, a.limit), indent=2))


if __name__ == "__main__":
    main()

"""Make the case prompts describe the garment photo actually used (fixes local_garments.py stand-ins).

  python3 fix_case_text.py [/workspace/azyah-vton]

local_garments.py replaced unreachable catalogue photos with ViViD garment photos but left the
catalogue name/description/category in test_garments.json, so the prompt described a different item
than the photo (e.g. "graphic tee" text with a cream sleeveless top photo). For every garment photo
that is a byte-for-byte copy of a ViViD ref.jpg, this rewrites the garment entry and the matching
cases/clip_list.json items to that ViViD garment's own category and caption (no catalogue name),
then deletes the case renders (out_*.mp4, perf_*.json) so they are made again. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

APP_CATEGORY = {"upper": "tops", "lower": "bottoms", "dress": "dresses"}


def _sha(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def main(root: str = "/workspace/azyah-vton") -> dict:
    r = Path(root)
    e = r / "eval/fx2"
    gj = r / "azyah-code/training/eval/test_garments.json"
    refs = {}
    for d in sorted(p for p in (r / "data/staging/vivid").iterdir() if p.is_dir()):
        if (d / "ref.jpg").exists() and (d / "meta.json").exists():
            refs.setdefault(_sha(d / "ref.jpg"), d)
    data = json.loads(gj.read_text())
    fixed = {}
    for g in data["items"]:
        photo = e / "cases/garments" / f"{g['key']}.jpg"
        src = refs.get(_sha(photo)) if photo.exists() else None
        if src is None:
            continue  # a real catalogue photo: its own text already matches
        meta = json.loads((src / "meta.json").read_text())
        g.update({"category": APP_CATEGORY.get(meta.get("category"), g["category"]), "name": None,
                  "description": meta.get("garment_description") or None, "stand_in_for": g.get("name"),
                  "photo_from": src.name})
        fixed[g["key"]] = {k: g[k] for k in ("category", "description", "photo_from")}
    bak = gj.with_suffix(".before_fix.json")
    if not bak.exists():
        shutil.copy(gj, bak)
    gj.write_text(json.dumps(data, indent=1))
    by_key = {g["key"]: g for g in data["items"]}
    cl = e / "cases/clip_list.json"
    n_items = 0
    if cl.exists():
        items = json.loads(cl.read_text())
        for it in items:
            g = by_key.get(it.get("garment_key"))
            if g and it["garment_key"] in fixed:
                it.update({"category": g["category"], "name": None, "description": g["description"]})
                n_items += 1
        cl.write_text(json.dumps(items, indent=2))
    removed = 0
    for p in list((e / "results/cases").glob("*/out_*.mp4")) + list((e / "results/cases").glob("*/perf_*.json")):
        p.unlink()
        removed += 1
    report = {"garments_fixed": fixed, "clip_items_fixed": n_items, "case_renders_removed": removed}
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(*sys.argv[1:2])

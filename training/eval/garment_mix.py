"""Review sets per garment type from clips that are already rendered (owner, 2026-11-21: judge the result on
dresses first, and on shirts and trousers too).

  python -m eval.garment_mix ROOT --sets dress10 fx6test cases5 [--per-type 8] [--prefix g_]

Reads the clip lists ROOT/eval/fxa/<set>.json, sorts their clips into dresses, tops and bottoms (the clip's
category) and writes one new clip list per type, ROOT/eval/fxa/g_<type>.json, with up to --per-type clips taken in
turn from the given sets. Its results folder ROOT/eval/fxa/results/g_<type>/<key> points at the clip's existing
folder (a link; a copy of the videos where links are not possible), so nothing is rendered again:
`eval.step_a video ROOT --set g_dresses ...` then makes the dresses-only video from the videos that are there.
Prints one GARMENT_SETS line. --prefix names the lists (g_dresses, ...): the yardstick run uses y_ so that the
training runs' review sets stay as they are.

--outerwear (owner's review, 2026-11-23: a long coat came out as a cropped jacket): the test clips file coats and
jackets as tops, and a top's wording tells the model that the trousers or skirt stay as they are. This writes
<prefix>coats instead: the tops whose name or description says coat, jacket, blazer, trench, parka or puffer, with
the category "outerwear" ("worn over the current outfit"), pointing at the same clip folders, so the same clips
can be rendered once more with that wording and looked at side by side.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from fluxrt_lora.pick_best_save import GARMENT_TYPES, garment_type

WORD = {"dresses": "dress", "tops": "top / shirt", "bottoms": "trousers / bottoms"}
OUTER_WORDS = ("coat", "jacket", "blazer", "trench", "parka", "puffer")


def in_turn(groups: list[list], n: int) -> list:
    """Up to n entries, one from each group in turn (so the dresses come from both dress sets)."""
    out, i = [], 0
    while len(out) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(out) < n:
                out.append(g[i])
        i += 1
    return out


def _point_at(link: Path, target: Path) -> str:
    """link -> target (an existing clip folder). A link when the volume allows it, else a copy of its videos."""
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    try:
        link.symlink_to(target.resolve(), target_is_directory=True)
        return "link"
    except OSError:
        link.mkdir(parents=True)
        for f in target.iterdir():
            if f.is_file() and (f.name == "input.mp4" or f.stem == "ref" or f.name.startswith("out_") or f.name == "item.json"):
                shutil.copy2(f, link / f.name)
        return "copy"


def build(root: Path, sets: list[str], per_type: int = 8, prefix: str = "g_") -> dict:
    fxa = root / "eval/fxa"
    res = fxa / "results"
    found: dict = {t: [] for t in GARMENT_TYPES}
    for s in sets:
        try:
            items = json.loads((fxa / f"{s}.json").read_text())
        except (OSError, ValueError):
            continue
        per_set: dict = {t: [] for t in GARMENT_TYPES}
        for it in items if isinstance(items, list) else []:
            t = garment_type(it.get("category")) if isinstance(it, dict) else None
            if t and it.get("key") and (res / s / str(it["key"])).is_dir():
                per_set[t].append((s, it))
        for t in GARMENT_TYPES:
            found[t].append(per_set[t])
    report = {}
    for t in GARMENT_TYPES:
        rows = in_turn(found[t], per_type)
        name = f"{prefix}{t}"
        if not rows:
            (fxa / f"{name}.json").unlink(missing_ok=True)
            continue
        (res / name).mkdir(parents=True, exist_ok=True)
        items, used, how = [], set(), set()
        for i, (s, it) in enumerate(rows, 1):
            key = str(it["key"]) if str(it["key"]) not in used else f"{s}_{it['key']}"
            used.add(key)
            how.add(_point_at(res / name / key, res / s / str(it["key"])))
            items.append({**it, "key": key, "case": f"{WORD[t]} {i} of {len(rows)} (test set {s})", "from_set": s})
        (fxa / f"{name}.json").write_text(json.dumps(items, indent=1))
        report[t] = {"set": name, "clips": len(items), "from": sorted({s for s, _ in rows}), "as": sorted(how)}
    print("GARMENT_SETS " + (" | ".join(f"{t}: {r['clips']} clips from {', '.join(r['from'])}" for t, r in report.items())
                             or "none: no clip list says what its clips are"))
    return report


def build_outerwear(root: Path, sets: list[str], per: int = 8, prefix: str = "g_") -> dict:
    """<prefix>coats: up to `per` tops that are really coats or jackets, as outerwear, on their own clip folders."""
    fxa = root / "eval/fxa"
    res = fxa / "results"
    name = f"{prefix}coats"
    rows = []
    for s in sets:
        try:
            items = json.loads((fxa / f"{s}.json").read_text())
        except (OSError, ValueError):
            continue
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict) or garment_type(it.get("category")) != "tops" or not it.get("key"):
                continue
            words = f"{it.get('name') or ''} {it.get('description') or ''}".lower()
            if any(w in words for w in OUTER_WORDS) and (res / s / str(it["key"])).is_dir():
                rows.append((s, it))
    rows = rows[:per]
    if not rows:
        (fxa / f"{name}.json").unlink(missing_ok=True)
        print("GARMENT_SETS coats: none of the tops is a coat or a jacket by its name or description")
        return {}
    (res / name).mkdir(parents=True, exist_ok=True)
    items, used = [], set()
    for i, (s, it) in enumerate(rows, 1):
        key = str(it["key"]) if str(it["key"]) not in used else f"{s}_{it['key']}"
        used.add(key)
        _point_at(res / name / key, res / s / str(it["key"]))
        items.append({**{k: v for k, v in it.items() if k != "prompt"}, "key": key, "category": "outerwear",
                      "case": f"coat or jacket {i} of {len(rows)} (test set {s})", "from_set": s})
    (fxa / f"{name}.json").write_text(json.dumps(items, indent=1))
    print(f"GARMENT_SETS coats: {len(items)} tops that are coats or jackets, from {', '.join(sorted({s for s, _ in rows}))}")
    return {"coats": {"set": name, "clips": len(items), "from": sorted({s for s, _ in rows})}}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="/workspace/azyah-vton")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--prefix", default="g_", help="the lists are <prefix>dresses, <prefix>tops, <prefix>bottoms")
    ap.add_argument("--outerwear", action="store_true", help="only <prefix>coats: the tops that are coats, as outerwear")
    a = ap.parse_args(argv)
    if not a.prefix or not a.prefix.replace("_", "").isalnum():
        ap.error("--prefix: letters, digits and _ only")
    if a.outerwear:
        return build_outerwear(Path(a.root), a.sets, a.per_type, a.prefix)
    return build(Path(a.root), a.sets, a.per_type, a.prefix)


if __name__ == "__main__":
    main()

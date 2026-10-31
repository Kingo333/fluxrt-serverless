"""Build JoyAI training manifests from shard catalogs, filtered by licence.

  python3 -m data_tools.manifests catalog.json [more.json ...] --out DIR --mode commercial|research --split train

Writes DIR/i2i.json and DIR/v2v.json in the upstream format
({"resolved_files": [...], "num_samples": N}), plus DIR/selection.json recording
which sources were used and the attribution lines a release must show.
'commercial' mode refuses research-only sources (ViViD, ToyData).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import sources


def load_catalogs(paths: list[str]) -> list[dict]:
    entries: list[dict] = []
    for p in paths:
        entries.extend(json.loads(Path(p).read_text()))
    return entries


def build(entries: list[dict], mode: str, split: str, include: set[str] | None = None) -> dict:
    if mode not in ("commercial", "research"):
        raise ValueError("mode must be commercial or research")
    groups: dict[str, dict] = {}
    used: set[str] = set()
    for e in entries:
        src = sources.get(e["source"])
        if e["split"] != split:
            continue
        if include is not None and e["source"] not in include:
            continue
        if mode == "commercial" and not src.commercial_ok:
            continue
        g = groups.setdefault(e["group"], {"resolved_files": [], "num_samples": 0})
        g["resolved_files"].append(e["path"])
        g["num_samples"] += e["num_samples"]
        used.add(e["source"])
    attribution = sorted({sources.get(s).attribution for s in used if sources.get(s).attribution})
    research_only = sorted(s for s in used if not sources.get(s).commercial_ok)
    return {"groups": groups, "used_sources": sorted(used), "mode": mode, "split": split, "attribution": attribution,
            # Provenance travels with every run and model (owner decision 2026-10-31).
            "research_only_sources": research_only,
            "trained_with_vivid": "vivid" in used,
            "commercial_use_cleared": not research_only,
            "notice": ("Trained with research-only data (" + ", ".join(research_only) + "). Not cleared for commercial "
                       "release without legal sign-off; a commercial-mode retrain excludes it.") if research_only else ""}


def write(selection: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for group, manifest in selection["groups"].items():
        (out / f"{group}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "selection.json").write_text(json.dumps(
        {k: v for k, v in selection.items() if k != "groups"}, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("catalogs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="commercial", choices=["commercial", "research"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--include", nargs="*", help="only these sources")
    a = ap.parse_args()
    sel = build(load_catalogs(a.catalogs), a.mode, a.split, set(a.include) if a.include else None)
    write(sel, a.out)
    print(json.dumps({k: v for k, v in sel.items() if k != "groups"}, indent=2))
    for g, m in sel["groups"].items():
        print(f"{g}: {len(m['resolved_files'])} shards, {m['num_samples']} samples")


if __name__ == "__main__":
    main()

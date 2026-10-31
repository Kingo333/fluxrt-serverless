"""Glance VTON Synthetic Pairs (CC-BY-4.0) -> staged pairs.

  python3 -m data_tools.convert_glance PARQUET_DIR STAGING_DIR [--configs mens-upperwear womens-upperwear] [--limit N]

PARQUET_DIR is a download of glanceai/VTON-Synthetic-Pairs (hf download ... --repo-type dataset),
laid out as <config>/<split>-*.parquet or any *.parquet below it; config and split are
read from the path. Columns: cloth, image, agnostic-mask, image-densepose (768x1024).

Each row becomes: ref.png = cloth, target.png = image, mask.png = agnostic-mask
(kept for pseudo-source generation and garment-region evaluation, never fed to the model).
The source image is generated later (pseudo_source step), then pack.py builds r2i shards.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from .staging import stage_pair

CONFIGS = ("mens-upperwear", "womens-upperwear", "mens-lowerwear", "womens-lowerwear")


def _cfg_split(path: Path) -> tuple[str, str]:
    """Config and split from the path *inside* the dataset folder only."""
    text = path.as_posix()
    # Whole path segments (or a segment's prefix like "mens-upperwear-00001"), never a
    # substring: "mens-lowerwear" is inside "womens-lowerwear".
    cfg = next((c for c in CONFIGS for part in path.parts if part == c or part.startswith(c + "-")), None)
    split = "test" if re.search(r"(^|[/_-])test([/_.-]|$)", text) else "train"
    if cfg is None:
        raise ValueError(f"cannot tell the Glance config from path {path}")
    return cfg, split


def _bytes(cell) -> bytes:
    if isinstance(cell, dict):
        return cell["bytes"]
    raise ValueError("unexpected image cell")


def convert(parquet_dir: str, staging_dir: str, configs: list[str], limit: int | None = None,
            per_config: int | None = None) -> int:
    """per_config: stage at most N train pairs (and N/5 test pairs) per config, so a
    subset stays balanced between e.g. men's and women's (cost cut C5)."""
    import pyarrow.parquet as pq

    root = Path(staging_dir)
    n = 0
    taken: dict[tuple[str, str], int] = {}
    for f in sorted(Path(parquet_dir).rglob("*.parquet")):
        cfg, split = _cfg_split(f.relative_to(parquet_dir))
        if cfg not in configs:
            continue
        cap = None if per_config is None else (per_config if split == "train" else max(1, per_config // 5))
        category = "upper" if "upper" in cfg else "lower"
        table = pq.read_table(f, columns=["cloth", "image", "agnostic-mask"])
        for row in table.to_pylist():
            if cap is not None and taken.get((cfg, split), 0) >= cap:
                break
            taken[(cfg, split)] = taken.get((cfg, split), 0) + 1
            cloth, image, mask = _bytes(row["cloth"]), _bytes(row["image"]), _bytes(row["agnostic-mask"])
            garment_id = "glance-" + hashlib.sha1(cloth).hexdigest()[:16]
            key = f"glance_{cfg.replace('-', '_')}_{split}_{n:07d}"
            stage_pair(root, key, {"ref.png": cloth, "target.png": image, "mask.png": mask}, {
                "source": "glance", "config": cfg, "split": split, "category": category,
                "garment_id": garment_id, "person_id": None, "task": "r2i",
            })
            n += 1
            if limit and n >= limit:
                return n
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_dir")
    ap.add_argument("staging_dir")
    ap.add_argument("--configs", nargs="*", default=["mens-upperwear", "womens-upperwear"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-config", type=int, help="balanced subset: N train pairs (+N/5 test) per config")
    a = ap.parse_args()
    print(f"staged {convert(a.parquet_dir, a.staging_dir, a.configs, a.limit, a.per_config)} Glance pairs")


if __name__ == "__main__":
    main()

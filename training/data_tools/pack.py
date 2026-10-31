"""Staged pairs (with generated sources) -> JoyAI shards + catalog.

  python3 -m data_tools.pack STAGING_DIR OUT_DIR --name glance_upper [--per-shard 500]

Pairs without a source yet are skipped and counted. Images become r2i samples
(0.png garment, 1.png source, 2.png target); videos become iv2v samples
(0.jpg garment, 0.mp4 source, 1.mp4 target). Writes OUT_DIR/<name>.catalog.json,
which manifests.py turns into training manifests.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

from .shards import Sample, write_shards
from .staging import find, instruction, iter_pairs


def _as_jpeg(data: bytes) -> bytes:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        out = io.BytesIO()
        im.convert("RGB").save(out, format="JPEG", quality=95)
        return out.getvalue()


def _sample(d: Path, meta: dict) -> Sample | None:
    ref, src, tgt = find(d, "ref"), find(d, "source"), find(d, "target")
    if not (ref and src and tgt):
        return None
    common = dict(key=d.name, caption=meta.get("caption") or instruction(meta["category"]),
                  source=meta["source"], split=meta["split"], garment_id=meta.get("garment_id"),
                  person_id=meta.get("person_id"), consent_id=meta.get("consent_id"),
                  extra={k: meta[k] for k in ("config", "category", "pseudo_source") if k in meta})
    if tgt.suffix == ".mp4":
        return Sample(task="iv2v", files={"0.jpg": _as_jpeg(ref.read_bytes()), "0.mp4": src.read_bytes(),
                                          "1.mp4": tgt.read_bytes()}, **common)
    files = {"0.png": ref.read_bytes(), "1.png": src.read_bytes(), "2.png": tgt.read_bytes()}
    if any(p.suffix != ".png" for p in (ref, src, tgt)):
        files = {"0.jpg": _as_jpeg(ref.read_bytes()), "1.jpg": _as_jpeg(src.read_bytes()), "2.jpg": _as_jpeg(tgt.read_bytes())}
    return Sample(task="r2i", files=files, **common)


def pack(staging_dir: str, out_dir: str, name: str, per_shard: int = 500) -> dict:
    buckets: dict[tuple, list[Sample]] = defaultdict(list)
    missing = 0
    for d, meta in iter_pairs(Path(staging_dir)):
        s = _sample(d, meta)
        if s is None:
            missing += 1
            continue
        buckets[(s.task, s.source, s.split)].append(s)
    catalog = []
    for (task, source, split), items in sorted(buckets.items()):
        catalog += write_shards(items, out_dir, f"{name}-{task}-{source}-{split}", per_shard)
    path = Path(out_dir) / f"{name}.catalog.json"
    path.write_text(json.dumps(catalog, indent=2))
    return {"catalog": str(path), "shards": len(catalog), "samples": sum(e["num_samples"] for e in catalog),
            "skipped_no_source": missing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-shard", type=int, default=500)
    a = ap.parse_args()
    print(json.dumps(pack(a.staging_dir, a.out_dir, a.name, a.per_shard), indent=2))


if __name__ == "__main__":
    main()

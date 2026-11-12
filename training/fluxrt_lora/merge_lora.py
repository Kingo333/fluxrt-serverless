"""Merge LoRA files into ONE file, exactly (owner, 2026-11-09): the live worker loads one LoRA.

  python -m fluxrt_lora.merge_lora --inputs A.safetensors B.safetensors --weights 0.5 0.5 --scale 0.65 \
      --out OUT_DIR [--selection-from DIR_WITH_selection.json] [--note "..."]

Each LoRA changes a layer by B_i x A_i (rank r_i). The blend w1*B1A1 + w2*B2A2, times the strength s, equals
[s*w1*B1 | s*w2*B2] x [A1 ; A2]: one LoRA of rank r1+r2, with no approximation. The file's settings record
(diffusers "lora_adapter_metadata": rank, alpha, target layers) is rewritten for the new rank with alpha =
rank, so it loads at exactly strength 1.0 of the merged change. selection.json (provenance) is copied next
to the file, as the worker requires. Writes OUT_DIR/pytorch_lora_weights.safetensors, selection.json, README.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

METADATA_KEY = "lora_adapter_metadata"


def merge_state_dicts(dicts: list[dict], weights: list[float], scale: float = 1.0) -> dict:
    """Rank-concatenate LoRA state dicts (same keys): lora_A stacked along rank, lora_B scaled by weight*scale
    and stacked along rank. Non-LoRA tensors (if any) must be identical and are kept from the first file."""
    import torch
    if len(dicts) != len(weights) or not dicts:
        raise ValueError("one weight per LoRA file")
    keys = set(dicts[0])
    for d in dicts[1:]:
        if set(d) != keys:
            missing = sorted(keys ^ set(d))[:5]
            raise ValueError(f"the LoRA files do not have the same layers; e.g. {missing}")
    out = {}
    for k in sorted(keys):
        ts = [d[k] for d in dicts]
        dtype = ts[0].dtype
        if ".lora_A." in k:
            out[k] = torch.cat([t.to(torch.float32) for t in ts], dim=0).to(dtype)
        elif ".lora_B." in k:
            out[k] = torch.cat([t.to(torch.float32) * (w * scale) for t, w in zip(ts, weights)], dim=1).to(dtype)
        else:
            if any(not torch.equal(t, ts[0]) for t in ts[1:]):
                raise ValueError(f"non-LoRA tensor differs between files: {k}")
            out[k] = ts[0]
    return out


def merged_metadata(metas: list[dict | None], new_rank: int) -> dict | None:
    """The settings record for the merged file: rank and alpha = new rank (scaling 1.0), per-layer rank
    patterns cleared (every layer has the new rank), everything else from the first file."""
    base = next((m for m in metas if m), None)
    if base is None:
        return None
    meta = dict(base)
    for key in list(meta):
        leaf = key.rsplit(".", 1)[-1]
        if leaf in ("r", "lora_alpha"):
            meta[key] = new_rank
        elif leaf in ("rank_pattern", "alpha_pattern"):
            meta[key] = {}
    return meta


def read_metadata(path: Path) -> dict | None:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        raw = (f.metadata() or {}).get(METADATA_KEY)
    return json.loads(raw) if raw else None


def rank_of(sd: dict) -> int:
    a = next((v for k, v in sd.items() if ".lora_A." in k), None)
    return int(a.shape[0]) if a is not None else 0


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="strength baked into the file (e.g. 0.65)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection-from", help="folder whose selection.json (provenance) is copied next to the file")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    from safetensors.torch import load_file, save_file
    paths = [Path(p) for p in a.inputs]
    dicts = [load_file(str(p)) for p in paths]
    ranks = [rank_of(d) for d in dicts]
    merged = merge_state_dicts(dicts, a.weights, a.scale)
    new_rank = sum(ranks)
    meta = merged_metadata([read_metadata(p) for p in paths], new_rank)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "pytorch_lora_weights.safetensors"
    sf_meta = {"format": "pt"}
    if meta:
        sf_meta[METADATA_KEY] = json.dumps(meta, indent=2, sort_keys=True)
    save_file(merged, str(dst), metadata=sf_meta)
    sel = {}
    if a.selection_from and (Path(a.selection_from) / "selection.json").exists():
        sel = json.loads((Path(a.selection_from) / "selection.json").read_text())
    sel["merged_from"] = [{"file": str(p), "weight": w, "rank": r} for p, w, r in zip(paths, a.weights, ranks)]
    sel["strength_baked_in"] = a.scale
    sel["merged_rank"] = new_rank
    sel["merged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sel.setdefault("commercial_use_cleared", False)
    (out / "selection.json").write_text(json.dumps(sel, indent=2) + "\n")
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    readme = (f"Merged FluxRT LoRA ({out.name}).\n"
              + "".join(f"  {w:.3f} x {p.name} from {p.parent.name} (rank {r})\n" for p, w, r in zip(paths, a.weights, ranks))
              + f"  strength {a.scale} baked in; merged rank {new_rank}; load at strength 1.0.\n"
              + (f"  {a.note}\n" if a.note else "")
              + f"  sha256 {digest}\n  size {dst.stat().st_size} bytes\n")
    (out / "README.txt").write_text(readme)
    report = {"out": str(dst), "rank": new_rank, "tensors": len(merged), "bytes": dst.stat().st_size, "sha256": digest,
              "metadata_rewritten": meta is not None}
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()

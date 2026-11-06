#!/usr/bin/env python3
"""Sanity-check a JoyAI DiT checkpoint before serving it: tensor layout and size.

  python3 check_dit_checkpoint.py /runpod-volume/joyai/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth \
      [--compare .../joyai_video_edit_dit_0811.pth]

Prints the top-level tensor count, the total parameter count and the file size, and with
--compare checks that both files have the same tensor names and shapes (0811 has 894 tensors).
A face++ file that does not match 0811's layout would not load in the server. Exit code 1 on a
mismatch, 2 on a missing file. torch is imported lazily (mmap load, no GPU).
"""

from __future__ import annotations

import argparse
import os
import sys


def tensors_of(path: str, loader=None) -> dict:
    """{name: shape} of every tensor in a .pth (nested 'state_dict'/'model' dicts unwrapped)."""
    if loader is None:
        import torch

        def loader(p):
            return torch.load(p, map_location="cpu", mmap=True, weights_only=True)
    obj = loader(path)
    for key in ("state_dict", "model", "module"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    out = {}
    for k, v in obj.items():
        if hasattr(v, "shape"):
            out[str(k)] = tuple(int(d) for d in v.shape)
    return out


def summarize(path: str, loader=None) -> dict:
    t = tensors_of(path, loader)
    params = sum(_prod(s) for s in t.values())
    return {"path": path, "tensors": len(t), "params": params, "bytes": os.path.getsize(path), "shapes": t}


def compare(a: dict, b: dict) -> list[str]:
    """Layout differences between two summaries, as plain lines (empty = same layout)."""
    sa, sb = a["shapes"], b["shapes"]
    out = []
    for k in sorted(set(sa) - set(sb))[:20]:
        out.append(f"only in {a['path']}: {k}")
    for k in sorted(set(sb) - set(sa))[:20]:
        out.append(f"only in {b['path']}: {k}")
    for k in sorted(set(sa) & set(sb)):
        if sa[k] != sb[k]:
            out.append(f"shape differs for {k}: {sa[k]} vs {sb[k]}")
    return out


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--compare", help="a known-good checkpoint (e.g. 0811) to compare the layout with")
    a = ap.parse_args(argv)
    for p in (a.checkpoint, a.compare):
        if p and not os.path.isfile(p):
            print(f"missing: {p}", file=sys.stderr)
            return 2
    s = summarize(a.checkpoint)
    print(f"{s['path']}: {s['tensors']} tensors, {s['params'] / 1e9:.2f}B params, {s['bytes']} bytes")
    if not a.compare:
        return 0
    c = summarize(a.compare)
    print(f"{c['path']}: {c['tensors']} tensors, {c['params'] / 1e9:.2f}B params, {c['bytes']} bytes")
    diffs = compare(s, c)
    if diffs:
        print("LAYOUT MISMATCH:\n  " + "\n  ".join(diffs))
        return 1
    print("same layout: every tensor name and shape matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Write a checkpoint's tensor names, shapes and dtypes to JSON, without loading it into RAM.

Used to check whether a trained checkpoint can be served by the live JoyAI loader:
dump the training start point and the deployed serving checkpoint, then compare.

  python3 dump_checkpoint_keys.py /path/to/model.pth out.json
  python3 dump_checkpoint_keys.py --compare a.json b.json
"""
import json
import sys


def dump(path: str, out: str) -> None:
    import torch

    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(sd, dict) and not any(hasattr(v, "shape") for v in sd.values()):
        # Some checkpoints wrap the weights, e.g. {"state_dict": {...}} or {"module": {...}}.
        for key in ("state_dict", "model", "module", "generator", "ema"):
            if isinstance(sd.get(key), dict):
                print(f"using nested '{key}'")
                sd = sd[key]
                break
    entries = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sd.items() if hasattr(v, "shape")}
    params = sum(int(__import__("math").prod(e["shape"])) for e in entries.values())
    json.dump({"path": path, "tensors": len(entries), "params": params, "keys": entries}, open(out, "w"))
    print(f"{path}: {len(entries)} tensors, {params / 1e9:.2f}B params -> {out}")


def compare(a_path: str, b_path: str) -> int:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    ka, kb = set(a["keys"]), set(b["keys"])
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    shape_diff = sorted(k for k in ka & kb if a["keys"][k]["shape"] != b["keys"][k]["shape"])
    dtype_diff = sorted(k for k in ka & kb if a["keys"][k]["dtype"] != b["keys"][k]["dtype"])
    print(f"A {a['path']}: {a['tensors']} tensors, {a['params'] / 1e9:.2f}B")
    print(f"B {b['path']}: {b['tensors']} tensors, {b['params'] / 1e9:.2f}B")
    print(f"common {len(ka & kb)} | only A {len(only_a)} | only B {len(only_b)} | shape differs {len(shape_diff)} | dtype differs {len(dtype_diff)}")
    for label, items in (("only A", only_a), ("only B", only_b), ("shape", shape_diff), ("dtype", dtype_diff)):
        for k in items[:15]:
            print(f"  {label}: {k}")
        if len(items) > 15:
            print(f"  {label}: ... {len(items) - 15} more")
    same = not (only_a or only_b or shape_diff)
    print("VERDICT: same layout (names and shapes match)" if same else "VERDICT: layouts differ (see above)")
    return 0 if same else 2


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    if len(sys.argv) == 3:
        dump(sys.argv[1], sys.argv[2])
        sys.exit(0)
    sys.exit(__doc__)

"""How strong is a LoRA compared with favourite #2, and how much of it is new? (owner, 2026-11-17)

  python -m fluxrt_lora.lora_geometry --ref FAVOURITE2.safetensors "label=FILE.safetensors" [...] [--json OUT.json]

Favourite #2 is 0.5 x fxreal3 + 0.5 x fxreal4 turned DOWN to strength 0.65, a strength picked on the live
(distilled, 2-step) model. Every later run started from that file and trained on the base model at full strength.
If training turns the strength back up, every trained model is over-driven on the live model whatever it learned.
This tool measures it without a GPU. Each LoRA changes a layer by dW = B x A; over all layers:

  strength   <dW, dW_ref> / <dW_ref, dW_ref>: how much of favourite #2's change the file carries.
             1.00 = favourite #2 as it runs live; 1.54 (= 1 / 0.65) = favourite #2 before it was turned down.
  new part   size of what is left after taking that out, in units of favourite #2's size (0 = nothing new).
  cosine     direction match with favourite #2 (1 = the same change, only stronger or weaker).
  size       |dW| / |dW_ref|.

<dW1, dW2> = sum((B1^T B2) * (A1 A2^T)) per layer, so files of different rank compare exactly and no full
weight matrix is built. Prints one GEOMETRY line per file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METADATA_KEY = "lora_adapter_metadata"


def pair_layers(sd: dict) -> dict:
    """{layer: (A [rank x in], B [out x rank])} in float64 from a diffusers LoRA state dict."""
    out = {}
    for k, a in sd.items():
        if ".lora_A." not in k:
            continue
        kb = k.replace(".lora_A.", ".lora_B.")
        if kb not in sd:
            continue
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(sd[kb], dtype=np.float64)
        out[k.split(".lora_A.")[0]] = (a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))
    return out


def inner(x: dict, y: dict) -> float:
    """<dW_x, dW_y> over the layers both have (a layer missing from one file changes nothing there)."""
    total = 0.0
    for name, (ax, bx) in x.items():
        if name not in y:
            continue
        ay, by = y[name]
        if ax.shape[1] != ay.shape[1] or bx.shape[0] != by.shape[0]:
            raise ValueError(f"layer {name} has different sizes in the two files")
        total += float(np.sum((bx.T @ by) * (ax @ ay.T)))
    return total


def geometry(ref: dict, x: dict, ref_scale: float = 1.0, x_scale: float = 1.0) -> dict:
    """strength, new part, cosine and size of x against ref (scales: alpha / rank of each file, 1.0 for ours)."""
    rr = inner(ref, ref) * ref_scale * ref_scale
    xx = inner(x, x) * x_scale * x_scale
    xr = inner(x, ref) * x_scale * ref_scale
    if rr <= 0:
        raise ValueError("the reference LoRA changes nothing")
    k = xr / rr
    rest = max(0.0, xx - k * k * rr)
    return {"strength": k, "new_part": float(np.sqrt(rest / rr)),
            "cosine": xr / float(np.sqrt(xx * rr)) if xx > 0 else 0.0, "size": float(np.sqrt(xx / rr)),
            "layers": len(x), "shared_layers": sum(1 for n in x if n in ref)}


def file_scale(path: Path) -> float:
    """alpha / rank from the file's settings record; 1.0 when there is none (our files store alpha = rank)."""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="np") as f:
            raw = (f.metadata() or {}).get(METADATA_KEY)
        meta = json.loads(raw) if raw else {}
    except Exception:  # noqa: BLE001 - a file without a readable record counts as alpha = rank
        return 1.0
    r = next((v for k, v in meta.items() if k.rsplit(".", 1)[-1] == "r" and isinstance(v, (int, float))), None)
    alpha = next((v for k, v in meta.items() if k.rsplit(".", 1)[-1] == "lora_alpha" and isinstance(v, (int, float))), None)
    return float(alpha) / float(r) if r and alpha else 1.0


def load_lora(path: Path) -> dict:
    try:
        from safetensors.torch import load_file
        sd = {k: v.float().numpy() for k, v in load_file(str(path)).items()}
    except ImportError:
        from safetensors.numpy import load_file
        sd = dict(load_file(str(path)))
    return pair_layers(sd)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="favourite #2's file")
    ap.add_argument("files", nargs="+", help="label=FILE")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    ref_path = Path(a.ref)
    ref, ref_scale = load_lora(ref_path), file_scale(ref_path)
    out = {}
    for item in a.files:
        label, _, path = item.partition("=")
        p = Path(path or label)
        if not p.exists():
            print(f"GEOMETRY {label}: file not found ({p})")
            continue
        try:
            g = geometry(ref, load_lora(p), ref_scale, file_scale(p))
        except Exception as e:  # noqa: BLE001 - one unreadable file must not stop the others
            print(f"GEOMETRY {label}: could not be read ({type(e).__name__}: {e})")
            continue
        out[label] = g
        print(f"GEOMETRY {label}: strength x{g['strength']:.2f} of favourite #2, new part {g['new_part']:.2f}, "
              f"cosine {g['cosine']:.2f}, size x{g['size']:.2f} ({g['shared_layers']}/{g['layers']} layers)", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()

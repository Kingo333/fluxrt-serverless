#!/usr/bin/env python3
"""Fold a Stage 2 / LHAD LoRA generator into its Stage 1 backbone, producing one
full DiT checkpoint the serving worker can load (Phase I).

  python3 merge_lora.py STAGE1.pth STAGE2_step_N.pth OUT.pth [--check-keys serving_keys.json] [--dtype bf16]

- STAGE2 checkpoints (upstream save_dmd_checkpoint) hold "generator_lora" (PEFT state
  dict) and "generator_lora_config" {r, lora_alpha}; they do NOT include the backbone,
  so the matching Stage 1 checkpoint is required.
- W_merged = W + (lora_alpha / r) * B @ A for every adapted Linear.
- --check-keys compares the result with a dump of the serving checkpoint
  (dump_checkpoint_keys.py on joyai_video_edit_dit_0811.pth) and fails on any
  missing/extra key or shape mismatch, so an unloadable file never reaches staging.
"""

from __future__ import annotations

import argparse
import json
import re

PREFIXES = ("base_model.model.", "module.", "_orig_mod.", "model.")


def _strip(k: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if k.startswith(p):
                k, changed = k[len(p):], True
    return k


def unwrap_backbone(ckpt: dict) -> dict:
    """Stage 1 saves may wrap the weights; accept {"model": {...}} or a plain state dict."""
    for key in ("model", "state_dict", "module", "generator"):
        if isinstance(ckpt.get(key), dict) and any(hasattr(v, "shape") for v in ckpt[key].values()):
            ckpt = ckpt[key]
            break
    return {_strip(k): v for k, v in ckpt.items() if hasattr(v, "shape")}


def merge(base: dict, lora: dict, r: int, alpha: float) -> tuple[dict, int]:
    """base: {name: W}; lora: PEFT keys like '<path>.lora_A.weight' / '<path>.lora_B.weight'
    (adapter name possibly inside: '<path>.lora_A.<adapter>.weight'). Returns merged copy and count."""
    scale = alpha / r
    pat = re.compile(r"^(?P<mod>.+)\.lora_A(?:\.[^.]+)?\.weight$")
    out = dict(base)
    merged = 0
    for key, a in lora.items():
        m = pat.match(_strip(key))
        if not m:
            continue
        mod = m.group("mod")
        b_key = next((k for k in lora if _strip(k) in (f"{mod}.lora_B.weight",) or
                      re.fullmatch(rf"{re.escape(mod)}\.lora_B\.[^.]+\.weight", _strip(k))), None)
        if b_key is None:
            raise SystemExit(f"LoRA A without B for {mod}")
        w_key = f"{mod}.weight"
        if w_key not in out:
            raise SystemExit(f"LoRA targets {w_key}, which is not in the Stage 1 backbone")
        w = out[w_key]
        delta = (lora[b_key] @ a) * scale
        if tuple(delta.shape) != tuple(w.shape):
            raise SystemExit(f"shape mismatch for {w_key}: {tuple(w.shape)} vs delta {tuple(delta.shape)}")
        out[w_key] = (w.astype("float32") if hasattr(w, "astype") else w.float()) + delta
        merged += 1
    if merged == 0:
        raise SystemExit("no LoRA layers found to merge")
    return out, merged


def check_keys(state: dict, serving_keys_json: str) -> list[str]:
    ref = json.load(open(serving_keys_json))["keys"]
    problems = []
    for k in sorted(set(ref) - set(state))[:20]:
        problems.append(f"missing for serving: {k}")
    for k in sorted(set(state) - set(ref))[:20]:
        problems.append(f"not in serving checkpoint: {k}")
    for k in sorted(set(ref) & set(state)):
        if list(state[k].shape) != ref[k]["shape"]:
            problems.append(f"shape {k}: {list(state[k].shape)} vs serving {ref[k]['shape']}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1")
    ap.add_argument("stage2")
    ap.add_argument("out")
    ap.add_argument("--check-keys")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()
    import torch
    base = unwrap_backbone(torch.load(a.stage1, map_location="cpu", mmap=True, weights_only=True))
    s2 = torch.load(a.stage2, map_location="cpu", weights_only=False)
    if not s2.get("dmd_use_lora"):
        raise SystemExit("Stage 2 checkpoint is not a LoRA checkpoint (dmd_use_lora missing)")
    cfg = s2["generator_lora_config"]
    merged, n = merge(base, s2["generator_lora"], int(cfg["r"]), float(cfg["lora_alpha"]))
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    merged = {k: v.to(dt) if v.is_floating_point() else v for k, v in merged.items()}
    if a.check_keys:
        problems = check_keys(merged, a.check_keys)
        if problems:
            print("\n".join(problems))
            raise SystemExit(f"{len(problems)} layout problem(s): not saving")
    torch.save(merged, a.out)
    print(f"merged {n} LoRA layers (r={cfg['r']}, alpha={cfg['lora_alpha']}) -> {a.out}")


if __name__ == "__main__":
    main()

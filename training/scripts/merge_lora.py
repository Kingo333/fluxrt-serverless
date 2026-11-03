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
# Wrapper names that can appear anywhere in a key (activation checkpointing, FSDP, compile, PEFT).
INNER = ("_checkpoint_wrapped_module.", "_fsdp_wrapped_module.", "_orig_mod.", "base_layer.")


def _strip(k: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if k.startswith(p):
                k, changed = k[len(p):], True
    for seg in INNER:
        k = k.replace(seg, "")
    return k


def _dense(v):
    """DTensor / sharded values from FSDP saves -> plain tensors."""
    return v.full_tensor() if hasattr(v, "full_tensor") else v


def unwrap_backbone(ckpt: dict) -> dict:
    """Stage 1 saves may wrap the weights; accept {"model": {...}} or a plain state dict."""
    for key in ("model", "state_dict", "module", "generator"):
        if isinstance(ckpt.get(key), dict) and any(hasattr(v, "shape") for v in ckpt[key].values()):
            ckpt = ckpt[key]
            break
    return {_strip(k): v for k, v in ckpt.items() if hasattr(v, "shape")}


def find_lora(ckpt: dict) -> dict:
    """The LoRA tensors: ckpt["generator_lora"] (our Stage 1/2 saves), else any dict (top level or
    one level down) whose keys contain '.lora_A'."""
    if isinstance(ckpt.get("generator_lora"), dict) and ckpt["generator_lora"]:
        return ckpt["generator_lora"]
    for cand in [ckpt, *[v for v in ckpt.values() if isinstance(v, dict)]]:
        if any(".lora_A" in str(k) for k in cand):
            return {k: v for k, v in cand.items() if "lora_" in str(k) and hasattr(v, "shape")}
    return {}


def describe(ckpt: dict, n: int = 20) -> str:
    """Key dump for when nothing matches: top-level layout and the first n keys of each dict."""
    lines = [f"top-level keys: {[(k, type(v).__name__) for k, v in list(ckpt.items())[:30]]}"]
    for k, v in ckpt.items():
        if isinstance(v, dict) and v:
            sample = list(v.items())[:n]
            lines.append(f"[{k}] {len(v)} entries; first {len(sample)}:")
            lines += [f"    {kk}  {tuple(getattr(vv, 'shape', ())) or type(vv).__name__}" for kk, vv in sample]
    return "\n".join(lines)


def merge(base: dict, lora: dict, r: int, alpha: float) -> tuple[dict, int]:
    """base: {name: W}; lora: PEFT keys '<path>.lora_A[.<adapter>...].weight' and the matching
    '<path>.lora_B[...]'. W_merged = W + (alpha / r) * B @ A. Returns merged copy and count."""
    scale = alpha / r
    norm = {_strip(k): _dense(v) for k, v in lora.items()}
    out = dict(base)
    merged = 0
    for key, a in norm.items():
        if ".lora_A." not in key or not key.endswith("weight"):
            continue
        mod, rest = key.split(".lora_A.", 1)
        b_key = f"{mod}.lora_B.{rest}"
        if b_key not in norm:
            raise SystemExit(f"LoRA A without B for {mod}")
        w_key = f"{mod}.weight"
        if w_key not in out:
            raise SystemExit(f"LoRA targets {w_key}, which is not in the Stage 1 backbone")
        w = out[w_key]
        delta = (norm[b_key] @ a) * scale
        if tuple(delta.shape) != tuple(w.shape):
            raise SystemExit(f"shape mismatch for {w_key}: {tuple(w.shape)} vs delta {tuple(delta.shape)}")
        out[w_key] = (w.astype("float32") if hasattr(w, "astype") else w.float()) + delta
        merged += 1
    if merged == 0:
        raise SystemExit("no LoRA layers found to merge; first keys: " + ", ".join(list(norm)[:20]))
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
    lora = find_lora(s2)
    if not lora:
        print(describe(s2))
        raise SystemExit("no LoRA tensors in the checkpoint (layout printed above)")
    cfg = s2.get("generator_lora_config") or {}
    ranks = {int(v.shape[0]) for k, v in lora.items() if ".lora_A" in str(k)}
    r = int(cfg.get("r") or (ranks.pop() if len(ranks) == 1 else 0))
    if not r:
        raise SystemExit(f"cannot tell the LoRA rank (config missing, ranks {sorted(ranks)})")
    alpha = float(cfg.get("lora_alpha") or r)
    print(f"LoRA tensors: {len(lora)}, r={r}, alpha={alpha}; first keys: {list(lora)[:3]}")
    try:
        merged, n = merge(base, lora, r, alpha)
    except SystemExit:
        print(describe(s2))
        raise
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    merged = {k: v.to(dt) if v.is_floating_point() else v for k, v in merged.items()}
    if a.check_keys:
        problems = check_keys(merged, a.check_keys)
        if problems:
            print("\n".join(problems))
            raise SystemExit(f"{len(problems)} layout problem(s): not saving")
    torch.save(merged, a.out)
    print(f"merged {n} LoRA layers (r={r}, alpha={alpha}) -> {a.out}")


if __name__ == "__main__":
    main()

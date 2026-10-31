#!/usr/bin/env python3
"""Rough GPU memory and cost estimate for JoyAI training runs. ESTIMATES, not
measurements: the Phase B smoke test logs real peak memory; feed it back here.

  python3 estimate.py stage1 --steps 2000 --step-seconds 40
  python3 estimate.py stage2 --steps 1000 --step-seconds 60

Model: 16.26B-parameter DiT (bf16 weights).
  stage1       upstream full fine-tune: all 16B with AdamW, sharded across GPUs (HSDP)
  stage1-lora  Azyah patch: frozen bf16 backbone + LoRA adapters (~0.5B params at r64)
  stage2       upstream LoRA mode (dmd_use_lora, the default): ONE shared frozen backbone
               with two adapters (generator, fake score); the real score is that base with
               adapters off (train_t2v_dmd.load_lora_base_dit). Activations are higher:
               generator rollout + fake-score + real-score passes.
"""
import argparse
import math

PARAMS = 16.26e9
GB = 1e9
TEXT_ENCODER_GB = 16.0      # MiMo-VL-7B in bf16, one copy per GPU unless offloaded
VAE_GB = 1.5
OVERHEAD_GB = 4.0           # CUDA context, allocator fragmentation, all-gather buffers
ACTIVATIONS_GB = (12.0, 30.0)  # 49 frames @ 480x832, micro-batch 1, activation checkpointing (guess range)

# name: (VRAM GB usable, $/hr per GPU from the latest readings; re-check live)
GPUS = {
    "rtx5090": (32, 0.99),
    "a100-80": (80, 1.39),
    "rtxpro6000": (96, 1.99),
    "h100-80": (80, 2.89),
    "b200": (180, 5.89),
}


def stage1_state_gb(low: bool) -> float:
    # weights bf16 (2) + grads bf16..fp32 (2..4) + AdamW m,v bf16..fp32 (4..8) bytes per parameter
    per_param = 2 + 2 + 4 if low else 2 + 4 + 8
    return PARAMS * per_param / GB


LORA_PARAMS = 0.5e9  # rank 64 on attention + MLP linears of 40 double blocks (rough)


def stage2_frozen_gb() -> float:
    return PARAMS * 2 / GB  # one shared bf16 backbone


def per_gpu_gb(stage: str, gpus: int, low: bool, offload_text_encoder: bool = False) -> float:
    act = ACTIVATIONS_GB[0] if low else ACTIVATIONS_GB[1]
    te = 0.0 if offload_text_encoder else TEXT_ENCODER_GB
    if stage == "stage1":
        sharded = stage1_state_gb(low) / gpus
    elif stage == "stage1-lora":
        lora = LORA_PARAMS * (2 + 2 + 8) / GB  # adapter weights, grads, AdamW states
        sharded = (PARAMS * 2 / GB + lora) / gpus
    else:
        lora = 2 * LORA_PARAMS * (2 + 2 + 8) / GB  # generator + fake-score adapters
        sharded = (stage2_frozen_gb() + lora) / gpus
        act *= 1.5  # three passes per step
    return sharded + act + te + VAE_GB + OVERHEAD_GB


def options(stage: str, steps: int, step_seconds: float, load_hours: float = 0.3):
    rows = []
    for name, (vram, price) in GPUS.items():
        for n in (1, 2, 4, 8):
            lo, hi = per_gpu_gb(stage, n, True), per_gpu_gb(stage, n, False)
            fit = "yes" if hi <= vram * 0.95 else "maybe" if lo <= vram * 0.95 else "no"
            # More GPUs = more samples per step, not faster steps; keep step time fixed per run.
            hours = steps * step_seconds / 3600 + load_hours
            rows.append({"gpu": name, "count": n, "per_gpu_gb": (round(lo), round(hi)), "vram": vram,
                         "fits": fit, "hours": round(hours, 2), "cost": round(hours * price * n, 2)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage1-lora", "stage2"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step-seconds", type=float, default=45.0, help="guess until the smoke test measures it")
    a = ap.parse_args()
    print(f"{a.stage}: {a.steps} steps x {a.step_seconds}s (ESTIMATE; prices to re-check live)")
    print(f"{'gpu':<12}{'n':>3}{'GB/GPU':>12}{'VRAM':>6}{'fits':>7}{'hours':>7}{'$':>9}")
    for r in options(a.stage, a.steps, a.step_seconds):
        if r["fits"] == "no":
            continue
        lo, hi = r["per_gpu_gb"]
        print(f"{r['gpu']:<12}{r['count']:>3}{f'{lo}-{hi}':>12}{r['vram']:>6}{r['fits']:>7}{r['hours']:>7}{r['cost']:>9}")


if __name__ == "__main__":
    main()

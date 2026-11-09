# FluxRT favourite #2: the fxreal3/fxreal4 blend, fused into INT8

Owner decision, 2026-11-09. Favourite #1 is plain FluxRT (Release #43 settings, no LoRA) and stays the
default. Favourite #2 adds one merged LoRA file and two settings. Both run the same image.

## The file

`/runpod-volume/fluxrt_lora/blend05s065/` on the endpoint volume (azyah-joyai):

- `pytorch_lora_weights.safetensors`: 0.5 x fxreal3 + 0.5 x fxreal4, strength 0.65 baked in, exact
  rank-64 merge (`training/fluxrt_lora/merge_lora.py`, proven by `verify_merge.py`: mean pixel
  difference 0.2-0.3 against the live blend). sha256 `8a2f5635482cde5a690a2044f5fa2bb06c80decf1bcbfc00a61734a9037d587f`.
- `selection.json`: provenance. Trained on research-licence data, so the worker loads it only with
  `FLUXRT_ALLOW_RESEARCH_MODEL=1` (owner's use and demos until the rights are cleared).
- `README.txt`: what it is and the checksum.

## Settings

| Setting | Favourite #1 | Favourite #2 |
|---|---|---|
| `FLUXRT_LORA_PATH` | unset | `/runpod-volume/fluxrt_lora/blend05s065/pytorch_lora_weights.safetensors` |
| `FLUXRT_ALLOW_RESEARCH_MODEL` | unset | `1` |
| `FLUXRT_LORA_FUSE` | (unused) | default `true`: fused into the weights before INT8, zero per-frame cost; `0` = attach the LoRA at run time instead (5-15% slower per frame) |
| `FLUXRT_LORA_SCALE` | (unused) | default `1.0`; the file already carries 0.65 |
| `FLUXRT_SHARPEN` | unset (off) | optional, `0` to `1`; start at `0.3`; above `0.5` edges halo |

Everything else stays as Release #43. Switching back = delete the LoRA settings. Each change restarts
the workers (one cold start).

## How the fuse works

`fluxrt/patches/fluxrt-fuse-lora.patch` (FluxRT's model process): with `use_lora` and `fuse_lora`, the
full-precision transformer (`FLUX.2-klein-4B/transformer`, always on the volume) is loaded, the LoRA is
added into its weights at `lora_scale`, and the result is quantized to INT8 with the same layers and
weight type as the pre-quantized model (read from its `quanto_qmap.json`). The text encoder keeps using
the pre-quantized folder. Cold start grows by the time to load 8 GB and quantize once (about a minute).

## What the log says

- `warmup: LoRA <path> (folder blend05s065, fused before INT8: True, scale 1.0)` or `warmup: no LoRA (plain FluxRT)`
- `warmup: output sharpen 0.3` or `off`
- the worker status has `lora`, `lora_fused`, `lora_scale`, `sharpen`
- a failure to fuse or load shows as `warmup failed: ...` and the worker never becomes ready: roll back the settings.

## Calibration of FLUXRT_SHARPEN

Measured on the owner's 5 clips (fx4 output vs the original, whole frame): amount 0.15 restores parity,
0.25 about +5%, 0.35 about +9%; the garment area alone loses 16-20% without it, so 0.3 is the suggested
start. Frame-to-frame change grows by 2-6% at 0.3 (no visible flicker). Cost about 0.5 ms per frame.

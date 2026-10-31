# FluxRT garment LoRA (lite / backup path)

This is a small LoRA on **FLUX.2-klein-4B**, the model FluxRT serves.
It teaches FluxRT our try-on task: put the garment on the person in the camera frame and keep everything else.
It is the cheap counterpart to the JoyAI fine-tune, and it uses the same data.

## What FluxRT does at inference (and what we train)

Each frame, FluxRT runs `Flux2KleinPipeline(image=[camera_frame, garment], prompt=…)` for 2 steps
(`model_inference_subprocess.py`, `process_frame_with_pipeline`).
The images get time offsets 10 and 20 (`_prepare_image_ids`).

Training uses the same setup:

| Training | Inference |
|---|---|
| condition 1 = **person source** (pseudo-source from P8), 288×512 | camera frame at `FLUXRT_WIDTH×HEIGHT` (288×512) |
| condition 2 = **garment**, white-padded to 384×512 | garment reference, padded the same way by `server.py` |
| target = **real person wearing the garment**, 288×512 | output (then 2× Flow Upscaler) |
| caption = `fluxrt_prompt()`, the Python twin of the app's `buildTryOnPrompt()` | the app's prompt |

The data is the same r2i triples as JoyAI (Glance, own clips), plus 2 matching stills per ViViD clip pair
(`--video-frames`, research-flagged). `selection.json` travels with the LoRA.

## Trainer and licences

| Piece | Licence | Notes |
|---|---|---|
| diffusers `examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py` @ `86ef420` | Apache-2.0 | Official FLUX.2-klein edit LoRA trainer. Its output is the diffusers LoRA format, which FluxRT's `Flux2LoraLoaderMixin.load_lora_weights` reads as-is. |
| `training/patches/flux2-klein-garment.patch` (ours) | — | Adds the second condition image (`--cond_image_column_2`, `--cond_2_size`). Also reuses repeated caption embeddings, which saves about 24 GB of GPU memory for 3,000 samples. |
| FLUX.2-klein-4B (base) | Apache-2.0 | 3.88B params; what FluxRT serves. The LoRA is trained on it directly. |
| FLUX.2-klein-base-4B | Apache-2.0 | Undistilled variant. Fallback if training on the distilled model looks unstable. `KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B` |
| Considered: ostris/ai-toolkit | MIT | Supports klein-4B with several control images. Not chosen: it saves LoRAs with ComfyUI key prefixes and adds a large dependency. |

`git apply --check` passes on the pinned commit and the patched file compiles.
**It has not run on a GPU yet**; the trial below is the first real test.

## Run (paid, owner's "yes" first)

```bash
curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/run_train.sh | \
  AZ_REF=<SHA> AZ_MANIFEST_DIR=/workspace/azyah-vton/manifests/research-train AZ_EXP=fluxlora-exp1 bash
```

Defaults:
- 3,000 samples, batch 4, rank 32, LR 1e-4, 1,500 steps (about 2 passes over the data)
- checkpoints every 100 steps (newest 2 kept)
- resumes automatically after a spot interruption
- no random flip, because flipping would mirror logos against the unflipped garment

Use `AZ_MAX_STEPS=100` for the trial.

Serve it (internal test only while ViViD is in the data):
1. Copy `outputs/fluxrt_lora/<exp>/pytorch_lora_weights.safetensors` and `selection.json` onto the endpoint volume.
2. Set `FLUXRT_LORA_PATH=/runpod-volume/…/pytorch_lora_weights.safetensors` and `FLUXRT_ALLOW_RESEARCH_MODEL=1`.

`server.py` refuses a LoRA without `selection.json`, or one not cleared for commercial use, unless that second variable is set.

## Cost (1 GPU Pod; all numbers are estimates until the trial measures step time)

Per step at 288×512, the transformer sees about 1,920 image tokens (576 target + 576 person + 768 garment) plus 512 text tokens.
- Forward pass: about 2 × 3.9B × 2.4K ≈ 1.9e13 FLOPs per sample
- Training with gradient checkpointing: about 4× that, ≈ 7.8e13 per sample
- At about 35% utilisation, that is about 1 s per sample on a 5090 (bf16)

| | 1× RTX 5090 ($0.99/h) | 1× RTX PRO 6000 ($1.99/h) |
|---|---|---|
| Memory | about 8 GB model + activations, fits 32 GB with gradient checkpointing (text encoder only during caching) | fits easily |
| Setup (env, 16 GB weights to the volume, caching) | about 30 min | about 30 min |
| Trial: 100 steps | about 10 min → **about $1** with setup | about $1.50 |
| Full: 1,500 steps × batch 4 (6,000 samples) | about 1.7–3.5 h → **about $2–4** | about 1.5–3 h → about $3–6 |
| Cap | **$8** | $12 |

The **5090 is the right card**: the PRO 6000 is about the same speed for this size of model, at twice the price.

It needs the P8 pseudo-sources (shared with JoyAI), so it can run right after P8 and before P9.
It is cheap enough to be the first trained model we evaluate.
For evaluation, run `eval/run_eval.py` on the same held-out items, with FluxRT plus the LoRA versus FluxRT without it.

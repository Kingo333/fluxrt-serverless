# JoyAI 0811 LoRA (improve the live JoyAI, keep its 2-step streaming)

A small LoRA on a **clone** of the model the app serves today, `joyai_video_edit_dit_0811.pth`.
The clone already sits on the training volume `azyah-vton-train`. The scripts refuse any base on `/runpod-volume` (the live `azyah-joyai` volume) or anywhere outside the training volume.
Nothing here touches the live endpoint, volume or image.

## What it trains

The LoRA learns one thing: make the live 2-step output closer to the real person wearing the real garment.

Each step:
1. **Inputs, as the live session feeds them** (upstream `train_t2v.prepare_inputs`):
   - person clip, 480×840 portrait, 24 fps
   - garment photo resized like live (basesize 768, aspect kept, upstream default buckets) and encoded as the reference latent
   - the app's instruction (`data_tools.staging.instruction`, the exact Python twin of the app's `buildTryOnInstruction`, full form with name and description), read together with the first camera frame (`use_vit`)
2. **The live rollout** (upstream `train_t2v_dmd.generate_dmd_latents`, the rollout JoyAI used to distil 0811 itself):
   - every latent chunk in order: chunk size 1, causal, window 3 with the first chunk as sink, `max_temporal_ids` 8 (the live id path)
   - the live scheduler's 2 steps: t = 1000 and 837.64, from shift 5.159 and `set_timesteps(2)`; no CFG
   - one step per chunk is back-propagated: `AZ_EXIT=random` (default, upstream) picks step 0 or step 1; `last` always picks step 1, the frame the app shows
   - history chunks are the model's own earlier outputs, detached (self-forcing)
3. **Loss**: MSE between that x0 prediction and the real target latents. Only the LoRA ("generator" adapter, rank 32) learns; the 16.6B base is frozen.

The trainer checks the live settings before step 1 (`self_check`) and stops on any difference:
timesteps, chunk/causal, `max_temporal_ids`, garment resize, `cfg_rate`/`use_vit`, frozen base.

## Why it keeps the 2-step behaviour

0811 is distilled: it was trained to finish in exactly these 2 steps.
- Ordinary fine-tuning at random noise levels (Stage 1 style) would pull it back toward a 30-step model.
- Here the model is only ever run, and corrected, on the 2-step live trajectory.

Guards:
- small LoRA (rank 32)
- low LR (3e-5, 50 warm-up steps)
- short run (1,000 steps)
- **every save kept** (250, 500, 750, 1000) and compared with untouched 0811 before anything ships

## Data (steady inputs only)

The owner found that the P8 ViViD pseudo-source **videos** flicker (clothes change within every clip).
So the default data is the Glance still triples (person pseudo-source still → real target still, plus the garment photo), each turned into a static 25-frame clip (`make_static_clips.py`):
- person and target get the same cover crop to 480×840
- 25 frames = 4 latent chunks = the sink plus the live 3-chunk window, so every history layout the live session uses is trained
- `--mode commercial` (default) keeps only commercially cleared sources
- `selection.json` (provenance, `commercial_use_cleared`) travels with the run
- **ViViD clip pairs are only included with `AZ_VIVID_VIDEOS=1`**, an explicit owner opt-in. That flag also switches the data to research mode (not commercially cleared).

## Gaps vs live (known, accepted)

- **History:** live keeps finished chunks in a KV cache. The training rollout recomputes them each forward as clean (time 0) tokens.
  - This is upstream's own approximation, the one used to train 0811.
- **Motion:** static clips teach garment fidelity and keeping the person, not motion.
  - Temporal behaviour stays with the frozen base.
  - A video round needs pseudo-sources rebuilt the live way, with a steady per-session reference.
- **Precision:** live runs FP8 weights and SageAttention on the 5090; training and eval are bf16. A merged LoRA sees the same FP8 conversion as 0811.
- **Long sessions:** the live KV reset every 600 frames and garment-swap restarts are not exercised.
- **Eval size:** the eval runs 480×832, upstream's 480 benchmark bucket; live is 480×840. Base and LoRA are compared the same way.

## Risks

- **Blur (regression to the mean):** a plain MSE target can make a few-step model softer.
  - Mitigations: low LR, small rank, short run, all saves kept, each one compared with 0811 by eye and with the metrics.
  - If every save is softer or less steady than 0811, nothing ships.
- **Speed:** the step time is not measured yet.
  - Estimate: 5–9 s per step (4 chunks × about 1.5 forwards + 4 backwards at about 4k tokens, plus VAE and text encoder).

## GPU, time, cost (estimates until the first log lines)

| | 1× RTX PRO 6000 96 GB ($2.49/h) |
|---|---|
| Setup and data (first run; data is cached for the same AZ_EXP) | about 20 min |
| 1,000 steps at 5–9 s | 1.4–2.5 h |
| Total | about 1.7–2.8 h, **about $4.25–7** |
| Cap | `timeout 165m` on training, about $6.85 |
| Eval of 4 saves + 0811 on 4 cross clips | about 40 min, about $1.70, cap `timeout 60m` |

- If the cap stops training early, the saves already written stay usable.
- Re-running the same `AZ_EXP` resumes from the newest save.
- If the first 20 steps average more than 8 s, set `AZ_MAX_STEPS=750` on the re-run.

## Run (paid; owner's yes first). Plain text for the extension

Training (Pod: 1× RTX PRO 6000, EU-RO-1, volume azyah-vton-train at /workspace):

for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo waiting for network; sleep 10; done
export AZ_REF=<SHA>
timeout 165m bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/joyai_0811_lora/run_train.sh | AZ_REF=$AZ_REF AZ_MANIFEST_DIR=/workspace/azyah-vton/manifests/research-train AZ_EXP=joyai0811-exp1 bash"; echo TRAIN_EXIT=$?
timeout 60m bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/joyai_0811_lora/eval_saves.sh | AZ_REF=$AZ_REF AZ_EXP=joyai0811-exp1 bash"; echo EVAL_EXIT=$?
runpodctl stop pod $RUNPOD_POD_ID

What to paste back:
- the "self-check ok" line
- the first 20 "Step N/1000: loss ... step_time" lines
- each "Stage 1 LoRA save: N adapter tensors" line (N about 640 at rank 32)
- GOOD_SAVES
- the eval report table

Then the owner watches the grid video (`/workspace/azyah-vton/eval/joyai0811_joyai0811-exp1_grid/all.mp4`): input | garment | live0811 | lora250 | lora500 | lora750 | lora1000.

## Serving a save later (separate approval)

1. `scripts/merge_lora.py <0811 clone> <step_N.pth> <merged.pth>` makes a full DiT. It's low-memory: the merge peaked at 64.5 GB RAM on exp2.
2. A **test** endpoint or volume would point `JOYOMNI_DIT_CKPT` at it.
3. Never on the live `azyah-joyai` volume without the owner's approval.

## Files

- `live_schedule.py`: the live numbers (timesteps, windows, exit policy). Pure Python, tested.
- `make_config.py`: `configs/azyah_joyai0811_lora.py` = the fashion config plus the live settings.
- `make_static_clips.py`: still triples → static iv2v clips, manifests and `selection.json`.
- `train.py`: the trainer (copied to upstream `training/src/azyah_train_0811_lora.py`).
- `run_train.sh`: env, patches, base guard, data, config, train, save check.
- `eval_saves.sh`: 0811 vs every save. Saves are fused at load with `generate_vivid --base`, so no merged files are written.

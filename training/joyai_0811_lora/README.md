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
  - Mitigations: the garment-weighted + edge loss (step 3a), low LR, small rank, short run, all saves kept, each one compared with 0811 by eye and with the metrics.
- **Remove examples:** the extra item is added by the 0811 clone itself; if it looks fake, the LoRA could learn to remove only fake-looking items. The 3b counts and a spot check of `live_source.png` files show it.
- **Edge weight:** 0.5 is a first guess; too high can sharpen into noise. The mse/edge numbers are logged per step.
  - If every save is softer or less steady than 0811, nothing ships.
- **Speed:** the step time is not measured yet.
  - Estimate: 5–9 s per step (4 chunks × about 1.5 forwards + 4 backwards at about 4k tokens, plus VAE and text encoder).

## Step 3a: what targets the owner's step 1 review

The owner watched live0811 (`--live --cross`, step 1) and found the proportions right but fine detail
simplified. Each change below maps to one item of that review.

| Owner's item (clip) | What targets it |
|---|---|
| Exact print (000002 generic flowers) | Garment-weighted loss (`AZ_MASK_WEIGHT` 3: errors inside the Glance garment mask count 3x) + edge term (`AZ_EDGE_WEIGHT` 0.5: L1 on finite differences inside the garment) + detail oversampling (most detailed third of garment photos x2) + detail captions |
| Trims: ruffles, bow/tie neck, collars, necklines (000010, 000002) | Edge term, oversampling, MiMo-VL captions naming neckline/collar/trims |
| Fabric texture: knit/rib (000010) | Edge term (ribs are edges), captions naming texture |
| Hem details, exact length (000010) | ViViD dress stills (full-length dresses; Glance has upper-body crops only), captions naming length/hem |
| Full replacement: leftover belt (000000) | "Remove" examples: 25% of live-way inputs get an extra item (belt, bag, scarf, hat) the real target lacks |
| Nothing added or duplicated: doubled bag (000002) | Real targets keep exactly what they have; remove examples teach "drop what isn't in the garment photo" |
| Garment resolution | Eval baseline `live0811_ref1024` (garment photo at basesize 1024). Training and serving stay 768 unless the owner approves a serving change |

Other 3a pieces:
- **Live-way inputs** (`data_tools/live_source.py`): the 0811 clone edits each real target with a fixed garment photo of another same-category item, as the app runs it, so inputs are steady (the P8 text-only videos drifted). Written as `live_source.*`; old `source.*` never touched; `pack.py --prefer-live-source` uses the new ones.
- **Bottoms:** Glance lowerwear (`glance_lower`, commercial-clean).
- **Dresses (owner's Option A, on by default):** about 400 ViViD train dress clips, one steady real frame each (`data_tools/vivid_stills.py`), input made the live way. ViViD is research-only: any LoRA trained with it has `commercial_use_cleared: false`. `AZ_WITH_VIVID_DRESSES=0` retrains the commercial-clean, Glance-only LoRA (use a new `AZ_EXP`; data is cached per experiment).
- **Captions:** `caption_garments.py --backend mimo` (MiMo-VL-7B-RL-2508, JoyAI's own text encoder, local) writes <=25-word details; half the training samples use the app's full instruction with them (`AZ_DETAIL_CAPTION_FRAC` 0.5), half the plain one, as wardrobe items without a description arrive.
- **Eval:** 16 cross clips. The owner's regression set (vivid_test_000000, 000002, 000010, 000029) keeps step 1's exact garments (`--pairs-from` step 1's run), plus 12 across categories. Full-height (832 px) grids for the regression set.

What it cannot fix:
- Exact flower-level print at 480x840: a whole dress covers a few hundred pixels of the live frame. Expect trims, collars, texture, removal and counts to improve more than exact print copying.
- Dress length and hems for commercial use: the only dress data is ViViD (research-only).
- Motion: still clips; steadiness over time stays with the frozen base.

## GPU, time, cost (estimates until the first log lines)

| Step | Pod | Expected | Cap |
|---|---|---|---|
| 3b CPU part: Glance lowerwear + ViViD dress stills | CPU Pod | about 30 min, about $0.10 | `timeout 60m` |
| 3b GPU part: ViViD top/bottom stills, live-way inputs (800 Glance upper + 800 Glance lower + 400 ViViD dresses + 600 ViViD tops + 343 ViViD bottoms at ~3 s, 5 model loads), captions (~1 s each), pack | 1x RTX PRO 6000 from the endpoint image ($2.49/h) | about 4.3 h, **about $10.70** | outer `timeout 280m`: **about $11.60** (resumable) |
| 3c training: 1,000 steps at 5–9 s + setup | 1x RTX PRO 6000 | 1.7–2.8 h, **about $4.25–7** | `timeout 165m`, about $6.85 |
| 3c eval: 16 clips x (2 baselines + 4 saves) at 49 frames, plus the 4 regression clips at 97 frames x (live0811 + 4 saves), frame strips | same Pod | about 1.7 h, about $4.20 | `timeout 150m`, about $6.20 |
| **Total 3b + 3c** | | **about $15.50–18.30** | **about $21.30** |

- If a cap stops training early, the saves already written stay usable; re-running the same `AZ_EXP` resumes.
- If the first 20 steps average more than 8 s, re-run with `AZ_MAX_STEPS=750`.
- 3b is resumable: re-running skips pairs that already have a live input or a caption.

## Run (paid; owner's yes first). Plain text for the extension

3b CPU part (CPU Pod, volume azyah-vton-train at /workspace):

for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo waiting for network; sleep 10; done
export AZ_REF=<SHA>
timeout 60m bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/joyai_0811_lora/build_data.sh | AZ_REF=$AZ_REF AZ_PART=cpu AZ_VIVID_DRESSES=1 bash"; echo CPU_EXIT=$?
runpodctl stop pod $RUNPOD_POD_ID

3b GPU part (1x RTX PRO 6000, EU-RO-1, Pod template = the endpoint image as for P8, volume azyah-vton-train at /workspace):

for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo waiting for network; sleep 10; done
export AZ_REF=<SHA>
nvidia-smi
timeout 280m bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/joyai_0811_lora/build_data.sh | AZ_REF=$AZ_REF AZ_PART=gpu AZ_VIVID_DRESSES=1 AZ_VIVID_STILLS=1 bash"; echo GPU_EXIT=$?
runpodctl stop pod $RUNPOD_POD_ID

3c train + eval (1x RTX PRO 6000, EU-RO-1, RunPod PyTorch CUDA 12.8 template, volume azyah-vton-train at /workspace):

for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo waiting for network; sleep 10; done
export AZ_REF=<SHA>
nvidia-smi
timeout 165m bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/joyai_0811_lora/run_train.sh | AZ_REF=$AZ_REF AZ_EXP=joyai0811-exp2 AZ_WITH_VIVID_DRESSES=1 bash"; echo TRAIN_EXIT=$?
timeout 150m bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/joyai_0811_lora/eval_saves.sh | AZ_REF=$AZ_REF AZ_EXP=joyai0811-exp2 bash"; echo EVAL_EXIT=$?
runpodctl stop pod $RUNPOD_POD_ID

Commercial-clean retrain (Glance only): the same 3c lines with AZ_EXP=joyai0811-exp2c AZ_WITH_VIVID_DRESSES=0.

What to paste back:
- 3b: the "live inputs tried / ok / remove examples / captioned" counts and both selection.json files
- 3c: the "self-check ok" line, the loss line ("garment weight 3.0 ... masks in ..."), the first 20 "Step N/1000: loss ... (mse ... edge ... masked ...)" lines, each "Stage 1 LoRA save: N adapter tensors" line (about 640), GOOD_SAVES, the eval report table
- the owner's look at the regression grids (`/workspace/azyah-vton/eval/joyai0811_joyai0811-exp2_regression_grid/`, 832 px): input | garment | live0811 | live0811_ref1024 | lora250 | lora500 | lora750 | lora1000

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
- `losses.py`: garment-weighted MSE + edge term (pure array maths, tested).
- `build_data.sh`: step 3b data (CPU and GPU parts).
- `eval_saves.sh`: 0811 + ref1024 baselines and every save on 16 cross clips; regression grids.
- `../data_tools/live_source.py`, `vivid_stills.py`, `caption_garments.py --backend mimo`: 3a data tools.
- `eval_saves.sh`: 0811 vs every save. Saves are fused at load with `generate_vivid --base`, so no merged files are written.

### ViViD tops and bottoms (owner decision 2026-11-05)

P8L (FluxRT, Glance-only synthetic stills) looked worse than base on real video, so step 3 adds real
ViViD frames for tops (600) and bottoms (343) next to the 400 dresses, inputs made the live way.
make_static_clips takes every ViViD real-frame sample first (--prefer-source vivid) and fills the
rest of --max with Glance. Research-only like the dresses; AZ_WITH_VIVID=0 (new AZ_EXP) is the
commercial-clean Glance-only fallback.

# Stage 1 fashion training plan

**Goal:** adapt JoyAI's bidirectional editing model to garment try-on on our data. Then distil it (Stage 2) back to a real-time, 2-step model. Stage 1 alone doesn't run in real time; it's the teacher and starting point for Stage 2.

Everything below is **estimated** until the Phase B smoke test measures real step time and memory. Rerun `scripts/estimate.py` with the measured numbers.

## Data mix (from `data_tools.manifests`)

| Run | Manifest mode | Image pairs (`i2i.json`, `r2i` samples) | Clips (`v2v.json`, `iv2v` samples) |
|---|---|---|---|
| **exp1, commercial** | `--mode commercial` | Glance upper-wear (about 23.7K train pairs, minus failed pseudo-sources) | Own clips (when recorded) |
| exp1-research | `--mode research` | Same | Own clips + **ViViD** (R&D only, not shippable without legal sign-off) |

**Task weights:** `AZ_I2I_PROB` and `AZ_V2V_PROB` (default 1:1). If there are no own clips yet, the clip task switches off automatically: its manifest is missing, so its probability becomes 0.

**Captions (review R2):** both forms the app sends live, built by a Python twin of the app's `buildTryOnInstruction` (tested on identical strings in both repos):
- `caption`: the short form ("Substitute the current top with the garment in the reference image. Keep the person's face…")
- `caption_detailed`: the long form with the garment name and description

The config samples detailed 70% / short 30% (`AZ_DETAILED_CAPTION_PROB`); samples without a description always use the short form. Descriptions come from `garments.csv` for our own clips and from `data_tools/caption_garments.py` (paid Gemini step P7b) for Glance and ViViD.

**Clean-input pairs (review R5):** `pack.py` adds about 10% "identity" pairs, where the source is the real target, so the model also sees clean camera-like input. The real-camera held-out test (own clips) checks how pseudo-source training transfers to live input.

## Settings (`scripts/make_fashion_config.py`, all set by env vars)

| Setting | Value | Why |
|---|---|---|
| Resolution | 480×832, 49 frames | Upstream smoke and real-time size; matches the live 480×840 |
| Learning rate | 2e-5 (upstream 5e-5), warm-up 100 | Fine-tuning: lower rate, less forgetting |
| Steps | 500 → 2000 (stepwise, see the ladder) | Evaluate before spending more |
| Checkpoints | every 250 steps, auto-resume | A stopped Pod loses at most 250 steps |
| Sharding | `HSDP=GPUS` on one node | Full parameter sharding needed for 16B with AdamW |
| Text-to-image and text-to-video tasks | off | No such data; keeps compute on try-on |

**Forgetting risk:** training only on fashion can wear away general skills the base model learned from upstream's large mix (temporal consistency, editing other things). For a try-on-only product that's mostly acceptable. We guard against it with:
- the low learning rate and short runs
- **evaluating on our held-out clips** (`eval/run_eval.py`) at every checkpoint, comparing flicker and preservation with the baseline

If they drop, we add a small "replay" set of general editing data. A licence check is needed first: LongV2VBench and OpenVE-Bench are evaluation sets, not training data.

## Experiment ladder (each step needs its own approval)

Costs come from `scripts/estimate.py` at 45 s per step (a guess): 2× B200 at about $11.8/hr, or 4× H100 at about $11.6/hr.

| Step | What | Hours | Cost | Go to the next step only if |
|---|---|---|---|---|
| S0 | Phase B smoke test (ToyData, 20 + 10 steps) | about 1 | $9–18 (maximum $25) | It trains, saves and resumes; real step time and memory measured |
| S1 | exp1, 500 steps, Glance only | about 6.5 | **about $75** | Loss falls; held-out garment colour and detail improve over the base model; flicker and preservation not worse |
| S2 | exp1, continue to 2000 steps (+ own clips if ready) | about 19 more | **about $225** | Clear win on held-out clips and in the blinded A/B |
| S3 | Stage 2 distillation from the best S1/S2 checkpoint (LoRA) | estimated separately once S2 passes (`estimate.py stage2`) | — | Real-time 2-step model at least matches the baseline speed |

**Storage needed before S1:** a **500 GB** volume (see `DATA_PLAN.md` G4). Each Stage 1 save with optimizer state is about 100–160 GB; `run_fashion.sh` keeps the newest 2 plus any save marked `KEEP` (`AZ_KEEP_LAST`), and deletes the rest automatically.

## Commands (after merge; run only when approved)

```bash
# on a CPU Pod: data -> shards -> manifests
python3 -m data_tools.manifests $AZ/shards/*.catalog.json --out $AZ/manifests/commercial-train --mode commercial --split train
# on the GPU Pod
curl -fsSL …/run_fashion.sh | GPUS=2 HSDP=2 AZ_MANIFEST_DIR=$AZ/manifests/commercial-train AZ_EXP=fashion-exp1 AZ_MAX_STEPS=500 bash
```

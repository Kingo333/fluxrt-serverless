# Stage 1 fashion training plan

**Goal:** adapt JoyAI's bidirectional editing model to garment try-on on our data. Then distil it (Stage 2) back to a real-time, 2-step model. Stage 1 alone doesn't run in real time; it's the teacher and starting point for Stage 2.

Everything below is **estimated** until the Phase B smoke test measures real step time and memory. Rerun `scripts/estimate.py` with the measured numbers.

## Data mix (from `data_tools.manifests`)

**Owner decision (2026-10-31): ViViD is in the default mix.** Experiments run in `--mode research`, so Stage 1 gets real video and motion data.

| Run | Manifest mode | Image pairs (`i2i.json`, `r2i` samples) | Clips (`v2v.json`, `iv2v` samples) |
|---|---|---|---|
| **exp1 (default)** | `--mode research` | Glance upper-wear (about 23.7K train pairs, minus failed pseudo-sources) | **ViViD** (about 7.8K train clips) + own clips when recorded |
| exp1-commercial (fallback, only if legal says no) | `--mode commercial` | Same | Own clips only |

**Provenance is enforced, not optional:**
- every run's `selection.json` records `trained_with_vivid: true`, `research_only_sources: ["vivid"]`, `commercial_use_cleared: false` and a notice
- `scripts/model_card.py` writes a `MODEL_CARD.md` that opens with "Not cleared for commercial use… Trained with ViViD: yes", plus a registry entry with `commercial_use_cleared: false`
- the worker **refuses** such a model unless `JOYAI_ALLOW_RESEARCH_MODEL=1`, which is set on staging only (Phase K gate)
- `--mode commercial` keeps working, so a ViViD-free retrain stays a one-flag change

**Task weights:** `AZ_I2I_PROB` and `AZ_V2V_PROB` (default 1:1: half the steps on Glance stills for garment fidelity, half on ViViD and own clips for motion). If there are no own clips yet, the clip task switches off automatically: its manifest is missing, so its probability becomes 0.

**Captions (review R2):** both forms the app sends live, built by a Python twin of the app's `buildTryOnInstruction` (tested on identical strings in both repos):
- `caption`: the short form ("Substitute the current top with the garment in the reference image. Keep the person's face…")
- `caption_detailed`: the long form with the garment name and description

The config samples detailed 70% / short 30% (`AZ_DETAILED_CAPTION_PROB`); samples without a description always use the short form. Descriptions come from `garments.csv` for our own clips and from `data_tools/caption_garments.py` (paid Gemini step P7b) for Glance and ViViD.

**Clean-input pairs (review R5):** `pack.py` adds about 10% "identity" pairs, where the source is the real target, so the model also sees clean camera-like input. The real-camera held-out test (own clips) checks how pseudo-source training transfers to live input.

## Cost-cut decisions (2026-10-31)

| Cut | Decision | Why |
|---|---|---|
| C1 Stage 1 LoRA | **Adopted** (`patches/stage1-lora.patch`, `AZ_STAGE1_LORA=1`, r64). Full fine-tune (`AZ_STAGE1_LORA=0`) is the fallback only if LoRA doesn't improve held-out fidelity | 1 GPU instead of 2× B200; about 1 GB checkpoints; no 16B AdamW states. Upstream Stage 1 has no LoRA path. The patch reuses upstream's Stage 2 `build_lora_config` and saves the same `generator_lora` format that `load_dit` and `merge_lora.py` read. **Risk:** LoRA may not be enough for the bidirectional → causal adaptation; the smoke test and exp1 eval tell us. **Untested on a GPU** until the P9 smoke test, which runs it first |
| C2 Smoke test merged into exp1 | **Adopted** (`scripts/run_first_session.sh`) | One Pod setup and one model load; exp1 starts only if the smoke passed |
| C3 Spot Pods | **Adopted if RunPod offers it with network volumes** | Checkpoints every 50 steps (`AZ_CKPT_INTERVAL`), auto-resume with optimizer and dataloader state (`resume_optimizer=True` now set); rerun the same command after an interruption |
| C4 Joint stills + clips, flip/affine, no colour jitter | **Joint mixing: already the plan (1:1).** **Augmentation: deferred** | exp1 sees each sample less than once (≤500 steps, batch 1, about 7K samples), so augmentation adds nothing yet. When runs go multi-epoch, flip garment photo, source and target together (text mirrored consistently), plus a small shared affine on source and target only; never colour jitter (shifts garment colour). Needs a loader patch |
| C5 Subsets + short captions | **Adopted:** 5,000 Glance (`--per-config 2500`) + 2,000 ViViD (`vivid_zip extract --sample 2000`, no full extraction); `AZ_DETAILED_CAPTION_PROB=0`; the trained model is served with `JOYAI_INSTRUCTION_STYLE=short` so the live prompt matches its captions | Pseudo-sources about $7 instead of $27–52; Gemini about $0 instead of $4–17 |
| C6 Only 500 steps pre-approved | **Adopted** | Go or no-go on the P9e eval before anything else |
| C7 Skip Stage 1 (CausVid/Self-Forcing-style DMD) | **Not now; researched option** | See below |

**C7 details: what upstream supports.**
- In **LoRA mode** (the default, `dmd_use_lora`), `load_lora_base_dit` loads **one** base for generator, fake score and real score. So teacher and student can't start from different checkpoints (e.g. a fashion-LoRA'd `bid_version` teacher and an `0811` student) without code changes.
- In **non-LoRA mode**, `dmd_generator_ckpt`, `dmd_fake_score_ckpt` and `dmd_real_score_ckpt` *can* differ. But that loads 3 full copies and trains 2 with full AdamW, the most expensive setup (multi-GPU).
- **Risks:**
  1. The Stage 2 configs build the **causal** DiT. A bidirectional `bid_version` teacher loaded into it hasn't been adapted to causal generation, which is exactly what Stage 1 does, so its scores could be poor.
  2. `0811` is already distilled to 2 steps, so it's a good student init but not a teacher.
  3. A two-base LoRA variant needs a code change: about 2 backbones (65 GB) on one B200.
- **Verdict:** revisit only if exp1 shows Stage 1 LoRA can't learn the fashion task cheaply.

## Settings (`scripts/make_fashion_config.py`, all set by env vars)

| Setting | Value | Why |
|---|---|---|
| Resolution | 480×832, 49 frames | Upstream smoke and real-time size; matches the live 480×840 |
| Learning rate | 2e-5 (upstream 5e-5), warm-up 100 | Fine-tuning: lower rate, less forgetting |
| Steps | 500 → 2000 (stepwise, see the ladder) | Evaluate before spending more |
| Checkpoints | every 250 steps, auto-resume | A stopped Pod loses at most 250 steps |
| Training mode | **LoRA r64 (`AZ_STAGE1_LORA=1`)**, 1 GPU | Cost cut C1; full fine-tune is the fallback |
| Sharding | `HSDP=GPUS` on one node | Only matters for the full fine-tune fallback |
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
| S1 | exp1, 500 steps, Glance + ViViD (research mode) | about 6.5 | **about $75** | Loss falls; held-out garment colour and detail improve over the base model; flicker and preservation not worse |
| S2 | exp1, continue to 2000 steps (+ own clips if ready) | about 19 more | **about $225** | Clear win on held-out clips and in the blinded A/B |
| S3 | Stage 2 distillation from the best S1/S2 checkpoint (LoRA) | estimated separately once S2 passes (`estimate.py stage2`) | — | Real-time 2-step model at least matches the baseline speed |

**Storage needed before S1:** a **500 GB** volume (see `DATA_PLAN.md` G4). Each Stage 1 save with optimizer state is about 100–160 GB; `run_fashion.sh` keeps the newest 2 plus any save marked `KEEP` (`AZ_KEEP_LAST`), and deletes the rest automatically.

## Commands (after merge; run only when approved)

```bash
# on a CPU Pod: data -> shards -> manifests
python3 -m data_tools.manifests $AZ/shards/*.catalog.json --out $AZ/manifests/research-train --mode research --split train
# fallback only if legal says no to ViViD:
python3 -m data_tools.manifests $AZ/shards/*.catalog.json --out $AZ/manifests/commercial-train --mode commercial --split train
# on the GPU Pod
curl -fsSL …/run_fashion.sh | GPUS=2 HSDP=2 AZ_MANIFEST_DIR=$AZ/manifests/research-train AZ_EXP=fashion-exp1 AZ_MAX_STEPS=500 bash
```

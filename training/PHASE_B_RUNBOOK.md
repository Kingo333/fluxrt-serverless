# Phase B runbook: training environment and smoke test

**Goal:** prove the upstream Stage 1 training runs on our hardware. That means forward and backward passes run, a checkpoint saves, resume works, and we know the peak GPU memory. Before that, two cheap checks: download all assets, and compare the checkpoint layouts.

Each step says **who** runs it and **what it costs**. **`<SHA>`** in every command is the merged commit Claude Code gives you after the merge (review R4: never run from a moving `main`). The scripts log it to `reports/code_refs.jsonl`. Steps marked 💲 need the owner's written "yes" first. Scripts are in `training/scripts/` and are fetched with `curl` from this public repo after merge, so **run nothing until the PR is merged.**

Prices are the roadmap's and the extension's latest readings. **The extension must re-check live prices in B2.**

| Step | Who | What | Cost |
|---|---|---|---|
| B2 | Extension | Live Pod prices and free capacity for: 4× H100 80 GB (SXM or PCIe), 2× B200, 1× B200, a CPU Pod (8 GB RAM is enough). Which datacenters have both the GPU option and CPU Pods | Free |
| B4 | Extension (owner OKs the names) | Create private Hugging Face repos `kingo333/azyah-vton-checkpoints` (model) and `kingo333/azyah-vton-data` (dataset). The owner creates a **write** token and adds it as RunPod secret `HF_TOKEN_WRITE` | Free |
| B5 💲 | Extension | Network volume `azyah-vton-train`, **200 GB**, in the B2 datacenter | About $14/month |
| B5a 💲 | Extension | CPU Pod (Ubuntu 22.04 based, for example a RunPod PyTorch template; 20 GB container disk) with `azyah-vton-train` attached, then run the asset download and env setup (below) | About $0.08–0.16/hr, about 1–1.5 h ≈ **$0.25** |
| B5b 💲 | Extension | Checkpoint key comparison (below). Uses one CPU Pod per volume | About **$0.10** |
| B6 💲 | Extension | GPU Pod smoke test (below) | Estimate below |

## B5a: assets and environment (CPU Pod, volume at `/workspace`)

```bash
curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/scripts/download_assets.sh | bash
curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/scripts/setup_training_pod.sh | bash
```

**Expect:**
- `ok` lines, `DiT size matches Hugging Face`, then `Done. Training assets are on the volume.`
- Then `Setup done.`, showing `torch 2.9.1+cu128`. "No GPU" is fine on a CPU Pod.

Paste the last 30 lines of each command. Both are safe to re-run.

## B5b: does the training checkpoint match the serving layout?

1. **Pod 1** (`azyah-vton-train` volume):
   ```bash
   python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
   curl -fsSLO https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/scripts/dump_checkpoint_keys.py
   python3 dump_checkpoint_keys.py /workspace/azyah-vton/checkpoints/dit/pretrain_editing_bid_version.pth /workspace/azyah-vton/reports/keys_bid.json
   ```
2. **Pod 2**: a CPU Pod in EU-RO-1 with `azyah-joyai` attached **read-only use, no writes** except `/tmp`. Same install and download lines as Pod 1, then:
   ```bash
   python3 dump_checkpoint_keys.py /workspace/joyai/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth /tmp/keys_0811.json
   ```
   Copy `/tmp/keys_0811.json` to Pod 1's volume (`/workspace/azyah-vton/reports/`). For example, upload it to the private Hugging Face data repo from Pod 2 with `HF_TOKEN_WRITE`, then download it on Pod 1. Claude Code gives the exact commands at that point.
3. On Pod 1:
   ```bash
   python3 dump_checkpoint_keys.py --compare /workspace/azyah-vton/reports/keys_bid.json /workspace/azyah-vton/reports/keys_0811.json
   ```

Paste the full output. A `same layout` verdict means a trained Stage 1 model loads in the serving code as is. Differences tell us what conversion the staging deploy (Phase I) needs.

## B6: smoke test (GPU Pod, volume at `/workspace`)

Stage 1 trains all 16B parameters, estimated at 150–230 GB of GPU memory in total (`PHASE_A_AUDIT.md` §5), so use one of:

| Option | GPUs | Command |
|---|---|---|
| A (preferred: fits per `scripts/estimate.py`, 99–165 GB per GPU) | 2× B200 180 GB | `curl -fsSL …/run_smoke.sh \| GPUS=2 HSDP=2 bash` |
| B (may run out of memory: 66–108 GB per GPU against 80) | 4× H100 80 GB | `curl -fsSL …/run_smoke.sh \| GPUS=4 HSDP=4 bash` |
| Memory probe only (risky) | 1× B200 | `GPUS=1 HSDP=1`. Likely out of memory; only if A and B are unavailable, and only to learn the limit |

(`…` = `https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/scripts`)

**Steps:**
1. Re-run `setup_training_pod.sh` first (about 1 minute; the environment is already on the volume).
2. Then run `run_smoke.sh`. It runs 20 steps, then resumes to 30. It logs GPU memory every 5 s and writes `reports/smoke_<time>.log` and `_gpu.csv`.

**Time estimate:** setup recheck about 2 min, model load about 5 min per run, about 20 + 10 steps at an unknown speed (guess 30–60 s per step for 49 frames at 480×832) ≈ **45–90 min**.

**Cost estimate:**

| Option | Hourly rate | Expected | Maximum (hard stop at 2 h) |
|---|---|---|---|
| A: 2× B200 | about 2 × $5.89 ≈ $11.8/hr | **about $9–18** | **$25** |
| B: 4× H100 | about 4 × $2.89 ≈ $11.6/hr (PCIe; SXM costs more) | **about $9–18** | **$25** |

**Stop conditions:** 2 hours, $25, an out-of-memory error, or the same error twice. Stop and delete the Pod right away when done; keep the volume.

**Done when** the summary shows:
- 20 steps logged, then a checkpoint `.pth` under `outputs/stage1_smoke/…/global_step20/`
- run 2 logs "Auto-resume" / "Resuming from checkpoint" and reaches step 30
- the peak memory per GPU is listed
- no errors

Upload `reports/` and the step-30 checkpoint to `kingo333/azyah-vton-checkpoints`. The checkpoint is large: check its size and ask the owner before uploading it, since Hugging Face storage limits may apply.

## Budget summary for Phase B

| Item | Cost |
|---|---|
| Volume, 200 GB | about $14/month |
| CPU Pods (B5a, B5b) | about $0.35 |
| GPU smoke test | about $9–18 (maximum $25) |

The RunPod balance must cover the maximum plus a margin: **top up about $40** for all of Phase B, or only about $10 for the free-to-cheap steps up to B5b.

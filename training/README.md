# Azyah-VTON training workspace

Code and docs for training a fashion-specialised JoyAI model. **Code and configs only.** Datasets, checkpoints and logs go to private storage, never this repo:
- RunPod volume `azyah-vton-train`
- Hugging Face `kingo333/azyah-vton-checkpoints` and `kingo333/azyah-vton-data`

This folder lives in a public repo on purpose, so a RunPod Pod can download its scripts with `curl` from `raw.githubusercontent.com`.

| File | What it is |
|---|---|
| `PHASE_A_AUDIT.md` | How the live system works, checkpoint compatibility, weights, licences, memory estimates |
| `BASELINE_PLAN.md` | How we measure today's JoyAI, and the cost of that test |
| `PHASE_B_RUNBOOK.md` | Training environment, checkpoint check and smoke test, with commands and costs |
| `DATA_PLAN.md` | Datasets, licences, pseudo-pair design (G1), own recordings, storage and cost (G4), consent (G5), attribution (G6) |
| `DATA_SOURCES.md`, `CONSENT_TEMPLATE.md` | Required notices; participant consent template |
| `data_tools/` | Shard writer, licence-aware manifests, validator, Glance and ViViD converters |
| `scripts/` | `download_assets.sh`, `setup_training_pod.sh`, `run_smoke.sh`, `make_smoke_config.py`, `dump_checkpoint_keys.py` |

Production (endpoint `azyah-runpod-fluxrt-bootstrap`, volume `azyah-joyai`) is never used for training.

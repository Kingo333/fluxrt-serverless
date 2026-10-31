# Azyah-VTON training workspace

Code and docs for training a fashion-specialised JoyAI model. **Code and configs only.** Datasets, checkpoints and logs go to private storage, never this repo:
- RunPod volume `azyah-vton-train`
- Hugging Face `kingo333/azyah-vton-checkpoints` and `kingo333/azyah-vton-data`

This folder lives in a public repo on purpose, so a RunPod Pod can download its scripts with `curl` from `raw.githubusercontent.com`.

| File | What it is |
|---|---|
| `PHASE_A_AUDIT.md` | How the live system works, checkpoint compatibility, weights, licences, memory estimates |
| `BASELINE_PLAN.md` | How we measure today's JoyAI, and the cost of that test |

Production (endpoint `azyah-runpod-fluxrt-bootstrap`, volume `azyah-joyai`) is never used for training.

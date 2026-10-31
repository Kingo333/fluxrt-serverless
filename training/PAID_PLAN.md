# Paid plan: everything that costs money, in order

**All free work is done** (PRs listed in step P0). Every step below needs the **owner's written "yes"** for its own budget. Prices are the latest readings: RunPod Serverless 5090 $1.79/hr, 96 GB PRO $3.49/hr; Pods 5090 $0.99, RTX PRO 6000 $1.99, H100 $2.89, B200 $5.89 per GPU-hour; network volume about $0.07/GB/month. **The extension re-checks live prices before each step.**

**Rules for every step:**
- Stop and delete Pods as soon as their step is done. Volumes are kept.
- The extension reports the actual cost and the balance after each step.
- Hitting a step's cap or time limit, or the same error twice, means stop.

| # | Step | Who | Needs first | Cost (expected / cap) | Time limit |
|---|---|---|---|---|---|
| P0 | **Merges** (free): fluxrt-serverless #19 → #20 → #21 → #22, azyah-shopping #9 → #10. The worker image builds automatically (about 30 min; re-run once if GitHub's runner runs out of memory) | Owner says "merge"; Claude Code merges | — | $0 | — |
| P1 | **Top-up** RunPod (balance was −$0.27). **Immediately** set Active workers = 0 on `azyah-runpod-fluxrt-bootstrap` | Owner pays; extension sets Active 0 | — | Suggested **$30** now (covers P2–P6); **+$100** before P7–P9 | — |
| P2 | Update the endpoint to the new image (free to save). Optional G8: rotate the Hugging Face token and store it as a RunPod Secret | Extension | P0, P1 | $0 | — |
| P3 | **A3 baseline benchmark** on production (5090) | Extension, then Claude Code reads the stats | P2 | ~$1.20 / **$4** | 60 min |
| P4 | **Training volume** `azyah-vton-train`, 200 GB, in a datacenter with B200 Pods + CPU Pods (B2 price/capacity check first) | Extension | P1 | ~$14/month | — |
| P5 | **CPU Pod:** asset download + training env + checkpoint key dump (bid); second CPU Pod on `azyah-joyai` for the `0811` key dump (`PHASE_B_RUNBOOK.md` B5a/B5b) | Extension runs Claude Code's commands | P4 | ~$0.35 / **$2** | 3 h |
| P6 | **Smoke test** on 2× B200 Pod (20 steps + resume to 30; measures real step time and memory) | Extension | P5 | ~$9–18 / **$25** | 2 h |
| P7 | **Data prep, CPU Pod:** Glance upper-wear (about 6 GB) → staged pairs. ViViD (R&D only): grow the volume to 400 GB first (+~$14/month), download 32 GB, `convert_vivid inspect` | Extension | P6 passed | ~$0.50 / **$3** (+ volume) | 4 h |
| P8 | **Pseudo-source trial** on 50 pairs (1× 5090 or RTX PRO 6000 Pod); then the **full run** sized from the trial's measured seconds per pair | Extension | P7 | Trial ~$1 / **$3**; full run quoted after the trial (expected ~$5–15 for 23.7K Glance pairs) | Trial 1 h |
| P9 | **Stage 1 exp1**, 500 steps (2× B200), then `eval` against the baseline on held-out data | Extension; Claude Code evaluates | P8; volume ≥ 400 GB | ~$75 / **$90** | 8 h |
| P10 | **Stage 1** continue to 2000 steps (+ own clips if recorded) | Extension | P9 shows improvement | ~$225 / **$260** | 22 h, in resumable chunks |
| P11 | **Stage 2 distillation** (real-time model) | Extension | P10 passed | quoted after P10 (`estimate.py stage2`) | — |
| P12 | **Staging endpoint** with `JOYAI_MODEL_VERSION=azyah-vton-experimental` (`STAGING.md`), owner tests on phone | Extension; owner | P11 | Serverless while testing, ~$1.79–3.49/hr; **Active 0** when idle | — |

**Not RunPod costs, but needed:**
- a lawyer's review of `CONSENT_TEMPLATE.md` before any own recording
- participant payments, if any
- Hugging Face storage if private uploads exceed the free quota

**Running total if every step passes:**
- P1–P6: about **$25–35**
- through P9: about **$110–130**
- through P10: about **$340–390**
- volume: $14–49/month depending on phase

The numbers before P6 are the most uncertain; the smoke test replaces the guesses with measurements.

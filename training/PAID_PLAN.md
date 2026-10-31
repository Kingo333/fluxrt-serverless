# Paid plan: everything that costs money, in order

**All free work is done** (PRs listed in step P0). Every step below needs the **owner's written "yes"** for its own budget. Prices are the latest readings: RunPod Serverless 5090 $1.79/hr, 96 GB PRO $3.49/hr; Pods 5090 $0.99, RTX PRO 6000 $1.99, H100 $2.89, B200 $5.89 per GPU-hour; network volume about $0.07/GB/month. **The extension re-checks live prices before each step.**

**Scripts are pinned:** every Pod command uses the merged commit SHA Claude Code provides (`…/fluxrt-serverless/<SHA>/training/scripts/…`), and it's logged to `reports/code_refs.jsonl` (review R4).

**Rules for every step:**
- Stop and delete Pods as soon as their step is done. Volumes are kept.
- The extension reports the actual cost and the balance after each step.
- Hitting a step's cap or time limit, or the same error twice, means stop.

| # | Step | Who | Needs first | Cost (expected / cap) | Time limit |
|---|---|---|---|---|---|
| P0 | **Merges** (free): fluxrt-serverless #19 → #20 → #21 → #22, azyah-shopping #9 → #10. The worker image builds automatically (about 30 min; re-run once if GitHub's runner runs out of memory) | Owner says "merge"; Claude Code merges | — | $0 | — |
| P1 | **Top-up** RunPod (balance was −$0.27). **Immediately** set Active workers = 0 on `azyah-runpod-fluxrt-bootstrap` | Owner pays; extension sets Active 0 | — | Suggested **$30** now (covers P2–P6); **+$100** before P7–P9 | — |
| P2 | Update the endpoint to the new image (free to save) **with the baseline switches on**: `JOYAI_CLIENT_HINTS=false`, `JOYAI_OUTPUT_QUALITY=70` (the pre-#18 behaviour; GPU profile on the 5090 is unchanged). Optional G8: rotate the Hugging Face token and store it as a RunPod Secret | Extension | P0, P1 | $0 | — |
| P3 | **Before/after benchmark in one window** (review R3). **Run A, today's live behaviour:** the baseline switches above, plus the app opened with `?livecam_legacy=1` (turns off #10's Gemini details and photo clean-up). **Run B, new:** remove the two env vars, save, and open the app normally. Same garments, same script including 3 garment swaps (`GARMENT_SWITCH.md`). New timing (#19/#9) works in both runs | Extension, then Claude Code reads the stats | P2 | ~$1.80 / **$5** | 75 min |
| P4 | **Training volume** `azyah-vton-train`, **200 GB to start** (grown to 400 GB at P7 and **500 GB before P9**; it can't shrink), in a datacenter with B200 Pods + CPU Pods (B2 price and capacity check first) | Extension | P1 | ~$14/month → ~$35/month at 500 GB | — |
| P5 | **CPU Pod:** asset download + training env + checkpoint key dump (bid); second CPU Pod on `azyah-joyai` for the `0811` key dump (`PHASE_B_RUNBOOK.md` B5a/B5b) | Extension runs Claude Code's commands | P4 | ~$0.35 / **$2** | 3 h |
| P6 | **Smoke test** on 2× B200 Pod (20 steps + resume to 30; measures real step time and memory) | Extension | P5 | ~$9–18 / **$25** | 2 h |
| P7 | **Data prep, CPU Pod (both required):** grow the volume to **400 GB** first (+~$14/month). Glance upper-wear (about 6 GB) → staged pairs. **ViViD:** download the 32 GB zip, unzip, `convert_vivid inspect` (Claude Code confirms the video-to-garment pairing), `convert_vivid convert`, then delete the zip | Extension | P6 passed | ~$0.70 / **$4** (+ volume) | 5 h |
| P7b | **Garment descriptions** for Glance **and ViViD** garment photos with Gemini Flash (`data_tools/caption_garments.py`): the detailed training caption (review R2). Trial on 50 photos, then the full run. Needs a Gemini API key the owner enters on the Pod | Extension; owner enters the key | P7 | Trial < $0.10; full about **34K photos** (24K Glance + 9.7K ViViD), about **$4–17** / **$20** (quote confirmed from the trial) | 4 h |
| P8 | **Pseudo-source trial** on 50 Glance stills + 20 ViViD clips (1× 5090 Pod); then the **full run** sized from the measured seconds per pair. It can be split across 2–4 Pods to shorten wall time (same total) | Extension | P7 | Trial ~$1–2 / **$4**. Full run quoted after the trial: Glance ~$5–15 + **ViViD ~$17–22** (about 7.8K train clips × about 5 s at about 832×624, roughly 8–10 s GPU each) ≈ **$22–37** / cap set from the trial | Trial 1.5 h |
| P9 | **Stage 1 exp1**, 500 steps (2× B200), then `eval` against the baseline on held-out data. Checkpoint pruning keeps the newest 2 + marked best | Extension; Claude Code evaluates | P8, P7b; **volume 500 GB** | ~$75 / **$90** | 8 h |
| P10 | **Stage 1** continue to 2000 steps (+ own clips if recorded) | Extension | P9 shows improvement | ~$225 / **$260** | 22 h, in resumable chunks |
| P11 | **Stage 2 distillation** (real-time model) | Extension | P10 passed | quoted after P10 (`estimate.py stage2`) | — |
| P11b | **Model card + registry entry** (`scripts/model_card.py`): states "Trained with ViViD: yes, not cleared for commercial use" | Claude Code | P11 | $0 | — |
| P12 | **Staging endpoint** with `JOYAI_MODEL_VERSION=azyah-vton-experimental` and `JOYAI_ALLOW_RESEARCH_MODEL=1` (staging only; `STAGING.md`), owner tests on phone | Extension; owner | P11b | Serverless while testing, ~$1.79–3.49/hr; **Active 0** when idle | — |
| K | **Pre-launch gate (Phase K):** production may serve the model only with **(a) written legal sign-off on ViViD**, recorded by setting `commercial_use_cleared: true` in its registry entry, **or (b) the ViViD-free retrain** (`--mode commercial`, Stage 1 + Stage 2 again, about $300+). Production never gets `JOYAI_ALLOW_RESEARCH_MODEL`; the worker refuses an uncleared model there | Owner + lawyer; Claude Code | P12 passed | (b) only: ~$300–400 | — |

**Not RunPod costs, but needed:**
- a lawyer's review of `CONSENT_TEMPLATE.md` before any own recording
- participant payments, if any
- Hugging Face storage if private uploads exceed the free quota

**Running total if every step passes:**
- P1–P6: about **$26–36**
- through P9: about **$145–175** (was $110–130; ViViD adds about $20–30 in pseudo-sources and captions)
- through P10: about **$370–420**
- volume: $14/month (200 GB) → $28 (400 GB, from P7) → **$35 (500 GB, from P9)**
- contingent: a ViViD-free retrain if legal says no, about **$300–400**

The numbers before P6 are the most uncertain; the smoke test replaces the guesses with measurements.

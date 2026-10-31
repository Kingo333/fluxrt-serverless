# Paid plan: everything that costs money, in order

> **Cost-cut revision (2026-10-31, owner budget $15–20 now).** Stage 1 is **LoRA on 1 GPU** (`patches/stage1-lora.patch`, about 1 GB checkpoints). The smoke test and exp1 run in **one Pod session** (`scripts/run_first_session.sh`). exp1 uses **data subsets** (5,000 Glance + 2,000 ViViD) and **short captions only**. **Only exp1 (≤500 steps) is pre-planned**; then a go/no-go. The volume stays at **200 GB**. The full fine-tune, Gemini captions and 400/500 GB volumes come back only if exp1 shows a need.

> **Live Pod prices (extension, 2026-10-31, on-demand; spot not shown):** RTX 5090 $1.19/h, RTX PRO 6000 $2.49/h, B200 $7.99/h, H100 SXM $3.99/h, A100 $1.79/h. Serverless (P3) is unchanged: 32 GB PRO $1.79/h.
> **Datacenter for `azyah-vton-train`: EU-RO-1.** It has both the 5090 (P8, P8L) and the PRO 6000 (P9), and it is the same DC as `azyah-joyai`, which B5b needs. A volume ties every Pod to its DC. **No B200 is listed there**, so the B200 fallback would need a new volume elsewhere (or the 2× PRO 6000 route); decide only if P9's smoke runs out of memory.

## Steps to the first evaluated result

| # | Step | Who | Expected / cap | Notes |
|---|---|---|---|---|
| P0 | ✅ **Merged 2026-10-31**: fluxrt-serverless #19 → #23 (main `3673d0c`), azyah-shopping #9 → #11 (main `70a2bcb`) | Owner said "merge" | $0 | Done |
| P1 | ✅ Topped up $15; Active workers 0 (balance $14.72) | Owner; extension | — | Done |
| P2 | New image on the endpoint, with the baseline switches (`JOYAI_CLIENT_HINTS=false`, `JOYAI_OUTPUT_QUALITY=70`). **Write down the current image tag (ending `62a8da4f`) as the rollback first.** The image also carries the pinned FluxRT `32206e8` (TAEF2 + Flow Upscaler on, cu128 torch) | Extension | $0 | — |
| P3 | **Before/after baseline**, 3 garment swaps each. **JoyAI** — A: today's behaviour + `?livecam_legacy=1`; B: new. **FluxRT** (same window, set `LIVECAM_ENGINE=fluxrt`, same 32 GB PRO card) — A: `ENABLE_FLOW_UPSCALER=false ENABLE_TINY_VAE=false FLUXRT_REF_WIDTH=288 FLUXRT_REF_HEIGHT=512 FLUXRT_CLIENT_HINTS=false`; B: remove them. Record fps, latency, GPU memory and garment colour/logo | Extension | ~**$3** / **$6**, about 100 min | Kept: necessary (+$1 for FluxRT) |
| P4 | Volume `azyah-vton-train` **200 GB** (enough: LoRA checkpoints are about 1 GB; ViViD is read from the zip) | Extension | ~$14/month (~$0.47/day) | Kept |
| P5 | **CPU Pod:** weights + env + checkpoint key dumps (`PHASE_B_RUNBOOK.md` B5a/B5b) | Extension | ~$0.35 / **$2** | Kept: necessary |
| P7 | **CPU Pod, data subsets:** Glance upper-wear `convert_glance --per-config 2500` (5,000 train pairs); ViViD zip download, `vivid_zip inspect` (I confirm the pairing), `vivid_zip extract --sample 2000`, `convert_vivid convert`, then delete the zip | Extension | ~$0.50 / **$3**, 4 h | Changed (C5) |
| P8 | **Pseudo-sources** for the subsets on 1× 5090 Pod (spot if available): trial 50 stills + 20 clips, then about 5,000 stills × ~1.5 s + 2,000 clips × ~9 s ≈ 7 h | Extension | trial ~$1.20; full ~**$8.50** on-demand at $1.19/h (re-priced from the trial) / **$12** | Changed (C5); live price |
| P8L | **FluxRT garment LoRA** (lite/backup path, `fluxrt_lora/README.md`): 1× **RTX 5090** Pod, same pseudo-source data (3,000 triples incl. 2 stills per ViViD clip). Trial 100 steps, then 1,500 steps | Extension | trial ~$1.20; full ~**$3–5** / **$8** | New, optional; can run before P9 |
| P9 | **First session, 1× RTX PRO 6000 Pod (spot if available)** with `run_first_session.sh`: setup, ToyData smoke (20 + resume to 30, LoRA; proves the patch, measures memory and step time), then exp1 **≤500 steps**, LoRA r64, checkpoints every 50 steps, auto-resume after interruption | Extension | ~**$34** on-demand (about 13.5 h at ~90 s/step × $2.49/h, a guess until the smoke measures it) / spot unknown / **$40** | Changed (C1, C2, C3, C6); live price. If the 96 GB card runs out of memory: 1× B200 (~$100 at $7.99/h, not in EU-RO-1), owner's yes first |
| P9e | **Evaluate exp1 vs baseline** offline on held-out Glance/ViViD test items (+ own clips when recorded): `eval/run_eval.py` + blinded A/B | Extension; Claude Code | ~$2.50 / **$5** | The go/no-go |

**Total to the first evaluated result (live on-demand prices, volume for about 2 weeks ≈ $7 included):**
- **JoyAI path** (P3–P9e): about **$56** (was $43 at the older prices); spot could cut P8/P9 if RunPod offers it in EU-RO-1
- **FluxRT LoRA path** (P3–P8 + P8L + eval): about **$24–26**
- both: about $60

**With $14.72 now:** the approved steps P3 (~$3), P4 (accrues ~$0.47/day) and P5 (~$0.35) fit with room to spare. **P7 + P8 (~$9) also fit; P8L needs about $5 more, P9 about $35 more.**

## Only after a "go" from P9e (each re-priced from P9's measured step time)

| # | Step | Expected |
|---|---|---|
| P10 | Continue exp1 (more steps, the full ViViD/Glance sets, own clips) | priced from P9 |
| P7b | Gemini detailed captions (34K photos) **if** exp1 shows a gain; live prompt then switches back to `full` | ~$4–17 |
| P11 | Stage 2 distillation (LoRA, **one shared backbone**: 1× B200, or a PRO 6000 if it fits). Stage 1 LoRA is merged into the bid base first (`merge_lora.py`) | priced later |
| P11b | Model card + registry entry (`model_card.py`; "Trained with ViViD: yes") | $0 |
| P12 | Staging endpoint (`JOYAI_ALLOW_RESEARCH_MODEL=1`, `JOYAI_INSTRUCTION_STYLE=short` for a short-caption model) | per hour while testing |
| K | **Pre-launch gate:** legal sign-off on ViViD, or the ViViD-free (`--mode commercial`) retrain | (b) only: retrain cost |
| — | Fallback: full Stage 1 fine-tune (2× B200) **only if** LoRA doesn't improve held-out fidelity | ~$75+ per 500 steps |

## Reference: rules and the earlier full plan

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

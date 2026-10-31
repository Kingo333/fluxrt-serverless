# Azyah-VTON data plan

How we turn public datasets plus a small set of our own recordings into JoyAI training data. Covers gaps G1–G8 raised by the extension and the owner. Checked through the Hugging Face connector and the dataset and paper pages on 2026-10-31.

## Sources

| Source | Licence | Commercial checkpoint? | Use |
|---|---|---|---|
| `glanceai/VTON-Synthetic-Pairs` | CC-BY-4.0 | **Yes, with attribution** (see G6) | Garment fidelity on stills (`r2i`). About 53K pairs, 4 configs, **19.6 GB of parquet** (upper-wear only is about 6.1 GB). Front-facing, grey background, no motion |
| `alibaba-yuanjing-aigclab/ViViD` | Hugging Face tag Apache-2.0, but the paper says "available for research purposes" and the footage is scraped from Net-A-Porter | **No, until legal sign-off.** **In the default training mix by owner decision (2026-10-31)**, with provenance tracked on every checkpoint | Motion and video (`iv2v`). 32 GB zip, about 9.7K video-garment pairs (7,759 train / 1,941 test), full-body studio and catwalk |
| Own recordings (Azyah) | Owned, under signed consent | Yes, if consent covers commercial use | Phone/webcam half-body clips at home: fine-tuning **and** the held-out test |
| JoyAI ToyData | — | No | Smoke tests only |
| Rejected | Tstars-VTON (CC-BY-NC-ND), mikusama99 (no licence), VITON-HD, DressCode, TikTokDress (non-commercial) | — | — |

**Enforcement:**
- `data_tools/sources.py` records each source's terms.
- **Default experiments use `--mode research`** (Glance + ViViD + own). `selection.json` then states `trained_with_vivid: true` and `commercial_use_cleared: false`.
- `scripts/model_card.py` puts that at the top of every model card and in the registry entry.
- The worker refuses such a model unless `JOYAI_ALLOW_RESEARCH_MODEL=1` (staging only).
- `--mode commercial` still **drops ViViD and ToyData automatically**, for a ViViD-free retrain if legal says no.

## G1: train/inference mismatch → **pseudo-pairs** (decided)

We train **"generated source with a different garment → real target"**, not "masked (agnostic) frame → original".

**Why:**
- **Inference input stays a raw camera frame.** A masked approach would need a real-time human parser in the worker for every frame. That adds latency and one more model to run, and parser mistakes at the garment edges would become output artefacts.
- **It's JoyAI's native task.** Upstream's reference-guided formats (`r2i`, `iv2v` in ToyData) are exactly *unmasked source + garment photo → edited target*. No change to the model's inputs or architecture.
- **The target is always real.** Synthetic artefacts appear only in the *input*, and the model learns to output realistic images.
- **It removes the alignment problem.** We don't need to film the same motion twice. One real clip of a person wearing garment X plus a photo of X is a training pair, once we generate a source that shows another garment.

**How sources are generated (a GPU batch job, paid, later):**
- Run the **current deployed JoyAI (`0811`)** offline on each real target with a *text-only* edit, e.g. "Substitute the current top with a plain {random colour} {random type} top", with no reference image.
- Keep only results where the garment region changed and the face and background didn't, using the offline masks (Glance's `agnostic-mask`, or SAM 2 masks for our own clips).
- We randomise colours and types so the model never learns a single "source look".

**Risk:** the model could learn JoyAI's own artefacts as "normal input". That's acceptable, because at inference it sees real camera frames, which are cleaner. We measure this on the real held-out set.

**Masks are offline only:** used to generate and filter sources and for garment-region evaluation metrics. Never a model input.

## G2: domain gap → our own clips are required

All public data is studio, full-body and evenly lit. Azyah is a phone or webcam, half-body, home lighting, close range.

**Plan:** about **5 people × 10–20 garments**, about 20 s per clip:
- Front view, slow left/right turns, arm raises, step closer and back.
- Phone held and laptop webcam.
- 2 lighting conditions.

Each clip is a person wearing **the real garment**, plus a flat photo of that garment.
- **Test split:** 1–2 whole people and 3–5 whole garments, never seen in training. This is the main quality benchmark.
- **The rest:** fine-tuning, after Glance (fidelity) and optionally ViViD (motion, R&D).

## G3: masks for our own clips

- **SAM 2** (Meta; code and released checkpoints Apache-2.0) for garment and person masks tracked through each clip, prompted with **Grounding DINO** (Apache-2.0) boxes ("shirt", "jacket"…).
- **DensePose isn't needed**, because pseudo-pairs don't use pose maps.
- **No Ultralytics/YOLO** (AGPL).
- Licences to re-check at install time: the SAM 2 checkpoint card, and the Grounding DINO weights.

## G4: storage (re-estimated)

| Item | Size |
|---|---|
| Training weights: bid DiT 32.5, VAE 1.5, MiMo-VL 16 | 50 GB |
| Deployed `0811` DiT for pseudo-source generation | 32.5 GB |
| ToyData | 0.1 GB |
| Glance parquet (all configs) + staged pairs + shards | 20 + 20 + 25 ≈ 65 GB (upper-wear only: about 20 GB) |
| ViViD zip + extracted + staged + generated sources + shards | 32 + 35 + 35 + 35 + 35 ≈ 170 GB peak. Delete the zip and extracted copy after staging: about 105 GB steady |
| Own clips: raw + masks + sources + shards | about 15 GB |
| Stage 1 checkpoints: full weights plus optimizer state, if saved | **about 100–160 GB each**; keep 2 ≈ 200–320 GB |
| Stage 2/LHAD LoRA checkpoints | small (< 5 GB each) |
| Reports, logs, evaluation videos | about 10 GB |

**Volume size by phase.** RunPod network volumes can grow but not shrink. At about $0.07/GB/month (our 100 GB volume is $7/month):

| Phase | Size | Monthly cost |
|---|---|---|
| Phase B smoke test only | 200 GB | about $14 |
| Data prep (Glance + ViViD + own clips) | 400 GB | about $28 |
| Stage 1 real runs, **keep-last-2 + 1 marked best** (`scripts/prune_checkpoints.py`, runs automatically in `run_fashion.sh`) | **500 GB** (3 × up to 160 GB + data) | about $35 |

**Cost-cut revision:** with Stage 1 LoRA (about 1 GB checkpoints), data subsets and ViViD read straight from the zip (`vivid_zip extract --sample`), **200 GB is enough for exp1**: about 50 GB weights + 32.5 GB `0811` + 32 GB zip (deleted after sampling) + about 25 GB staged, sources and shards + a few GB of checkpoints. The larger sizes below apply only to the full fine-tune fallback.

**Recommendation (full fine-tune only):** start at **200 GB** and grow when each phase starts (200 → 400 for data prep → 500 before Stage 1). Without pruning, upstream keeps every save: about 8 saves at 100–160 GB each over 2000 steps ≈ 0.8–1.3 TB (review R1).
- Hugging Face private storage: back up final model weights and small reports only. Stage 1 optimizer checkpoints stay on the volume.
- Free-tier private storage on Hugging Face is limited, so check `kingo333`'s quota before large uploads.

## G5: consent and data protection

`training/CONSENT_TEMPLATE.md` is a plain-language consent form and data notice. It covers the UAE PDPL (Federal Decree-Law No. 45 of 2021) and the GDPR for EU participants: purpose, data, retention, transfers to US/EU cloud providers (RunPod, Hugging Face, GitHub), withdrawal and deletion, adults only. **A template, not legal advice: have a lawyer review it before the first recording.** Every own sample carries a `consent_id`; the shard writer refuses own samples without one.

## G6: attribution

If a released model uses Glance data, show:

> Glance VTON Synthetic Dataset (2026), https://huggingface.co/datasets/glanceai/VTON-Synthetic-Pairs, CC BY 4.0

Put it on a credits or licences page in the app and in the model card. `manifests.py` writes the exact lines a run needs into `selection.json`. JoyAI (Apache-2.0) and MiMo-VL (MIT) also need their licence notices kept in the model card and image. Full list in `training/DATA_SOURCES.md`.

## G7: Hugging Face connector sign-in is short-lived

Claude Code re-checks `whoami` before each use and asks for a reconnect when it expires. Nothing on Pods depends on the connector: Pods use public downloads, or `HF_TOKEN_WRITE` for private uploads.

## G8: Hugging Face token saved in plain text on the endpoint

This is the owner's call. When the account is funded and the endpoint is edited anyway:
1. rotate the token on Hugging Face
2. store the new one as a **RunPod Secret**, referenced from the env var instead of the raw value
3. use a **read-only** token there

The training write token (`HF_TOKEN_WRITE`) stays separate.

## Pipeline

```
Glance parquet ──convert_glance──┐
ViViD zip ──unzip + convert_vivid──┼─> staged pairs (ref + real target [+ mask])
own clips ──ingest (later)───────┘        │
                                          ├─ pseudo_source (GPU, JoyAI 0811, text-only edit) -> source
                                          └─ pack -> shards + catalog -> validate -> manifests (commercial|research)
```

Every step except `pseudo_source` runs on a CPU Pod.

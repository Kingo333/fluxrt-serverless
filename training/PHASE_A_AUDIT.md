# Phase A audit: live try-on system and training compatibility

Written from the code in `Kingo333/fluxrt-serverless` (main `00b7fb3`), `Kingo333/azyah-shopping` (main `228dcdf`) and upstream `jd-opensource/JoyAI-Video-Edit` at the pinned commit `d875895`. Hugging Face facts were checked through the connector.

"Estimate" means calculated, not measured. Nothing here has run on a GPU yet.

## 1. How a live session works today

1. **Camera.** The app (`src/components/ai-studio/live-cam/useLiveCamSession.ts`) center-crops the camera to the capture size and encodes each frame as JPEG.
   - JoyAI: 480×840, 24 fps, quality 0.85 (sent in the adapter's `ready` message).
   - FluxRT: 288×512, 12 fps, 0.7.
2. **Session start.** Supabase Edge Function `live-cam-session-start` (live version v38, no tier logic) → Cloudflare Worker `fluxrt-orchestrator` (live May 22 version) signs a session token and returns the RunPod WebSocket URL.
3. **Transport.** WebSocket, with JSON messages and frames as base64 JPEG (about 33% larger than binary).
   - The app drops frames when more than 2 MB is waiting to send.
4. **RunPod.** Load Balancing endpoint `azyah-runpod-fluxrt-bootstrap`, port 8765, one shopper per GPU worker. `MAX_SESSION_SECONDS=300`.
5. **Adapter** (`joyai/adapter/`). Checks the token, then translates messages: `set_reference_image` + `set_prompt` become a JoyAI `start`; `frame` becomes binary JPEG plus a `frame_meta` `{seq, t_capture_ms}`.
6. **JoyAI** (unmodified upstream, 127.0.0.1:8080).
   - Model: DiT `joyai_video_edit_dit_0811.pth` (16.26B params, FP8 on the 5090), VAE, MiMo-VL-7B text encoder, YuNet face gate.
   - 2 denoising steps, recent-chunk memory `max_temporal_ids=8`, KV reset every 600 frames.
   - The garment photo is resized to about 768 px.
7. **Output.** Each output frame comes with `output_frame {source_seq, t_capture_ms, server_elapsed, profile}`, then the JPEG at quality 90.
   - Before this PR the adapter threw the header away. It now passes the matching client `seq`, `t_client_ms` and `server_ms` to the app, plus `profile` when `JOYAI_PROFILE_TIMINGS=true`.
8. **Playback.** The app decodes each JPEG and draws it on a canvas.
9. **Garment switch.** A new `set_reference_image` + `set_prompt` becomes one JoyAI restart on the same socket. JoyAI's edit state resets.
10. **Recovery.**
    - Upstream drops: reconnect, up to 3 times.
    - `/ping`: 204 while loading (or 200 with `PING_READY_WHILE_LOADING`), 503 on error.
    - A cold first start compiles kernels into the volume cache.
11. **Cost.** One GPU per active shopper. Serverless about $1.58/hr (5090) or about $3.49/hr (RTX PRO 6000), billed per second while a worker runs. Idle timeout is 300 s.

**Gaps for benchmarking:**
- The app's latency number measured "time since the last frame sent", not the matching frame. Fixed by the seq/timing passthrough in this PR, plus the paired app change.
- There is no per-stage breakdown. Opt-in through `JOYAI_PROFILE_TIMINGS`.

## 2. Weights (verified on Hugging Face)

| Weight | Repo / file | Size | Licence | Used for |
|---|---|---|---|---|
| Distilled real-time DiT (deployed) | `jdopensource/JoyAI-Video-Edit` `dit/joyai_video_edit_dit_0811.pth` | 32.5 GB | Apache-2.0 | Serving |
| Earlier distilled DiT | same repo, `dit/joyai_video_edit_dit_0804.pth` | 32.5 GB | Apache-2.0 | Not used |
| **Bidirectional editing DiT** | same repo, `dit/pretrain_editing_bid_version.pth` | 32.5 GB | Apache-2.0 | **Training start (Stage 1)** |
| VAE | same repo, `vae/` | 1.5 GB | Apache-2.0 | Both |
| Text encoder | `XiaomiMiMo/MiMo-VL-7B-RL-2508` | about 16 GB | MIT | Both |
| Face detector | OpenCV Zoo YuNet | 0.2 MB | MIT | Serving gate |
| Test data | `EasonXiao-888/JoyAI-Video-Edit_ToyData` (`t2i`, `i2i`, `t2v`, `v2v`, `iv2v`, `r2i` `.tar` files + 4 `.json` manifests) | about 0.1 GB | check the card | Smoke test only |

Weights plus ToyData come to about 50 GB, so **the training volume should be at least 200 GB** to leave room for checkpoints. Each full DiT checkpoint is about 32.5 GB in bf16; a Stage 1 save with optimizer state is several times that. Nothing here is gated. YOLOv8n (AGPL) stays off.

## 3. Checkpoint compatibility

- **Training starts from `pretrain_editing_bid_version.pth`**, not the deployed `0811`. The upstream path is:
  1. Stage 1: autoregressive adaptation, full parameters.
  2. Stage 2: Resilient-DMD distillation into a 2-step generator (LoRA).
  3. LHAD: long-horizon tuning (LoRA).
- **`0811` is not a good training starting point.** It's already distilled to a few steps. Stage 2 needs a multi-step Stage 1 model as its teacher (`real_score`) and initialisation (`train_t2v_dmd.py:493–498` load generator, fake score and real score from `JOYAI_STAGE1_CKPT`).
- **Stage 2 and LHAD save LoRA adapters only** (`training/README.md`): "do not include the full backbone". To serve one we must merge the adapters into the matching Stage 1 backbone and save it in the key format the serving `load_dit` expects (`deploy/xvideo/models/models.py`). **Unverified.** This needs a key comparison of `pretrain_editing_bid_version.pth` against `0811`, which can run on a cheap CPU Pod with `torch.load(..., mmap=True)`. That's planned as step B3b.
- Upstream doesn't publish a Stage 1 (AR) checkpoint, so **we have to run Stage 1 ourselves** before any Stage 2.

## 4. Data loader (training/src/dataset/webdataset.py)

- **The multi-video format** fits our garment try-on data directly. Each sample has `0.mp4`, `1.mp4` and an optional `0.jpg` reference image (lines 252–261). **The last video is the target** (`target_reader = video_readers[-1]`, line 448). Plan: `0.mp4` = source (person in their own clothes), `1.mp4` = target (same person in the garment), `0.jpg` = garment photo.
- The reference image is resized with its own bucket (`ref_image_basesize`).
- Captions come from the keys and probabilities in `train_multiple_videos_caption_keys`.
- **Open question for C1:** exactly how the JSON metadata switches reference conditioning on. To check while writing the converter.

## 5. Memory estimates (to confirm in the smoke test)

16.26B parameters, bf16, AdamW. Upstream defaults: activation checkpointing on, `micro_batch_size=1`, `sp_size=1`, `hsdp_shard_dim=1`. FSDP CPU offload is available (`cfg.cpu_offload`).

| Stage | What's trained | Rough total memory | Fits on |
|---|---|---|---|
| Stage 1 | **All 16B parameters** | weights 32.5 + grads 32.5 + Adam states about 65–130 + activations + text encoder 16 ≈ **150–230 GB**, shared across GPUs | **4× H100 80 GB or 2× B200 180 GB** with sharding (`hsdp_shard_dim`). **Not** one RTX PRO 6000 96 GB. One B200 only with CPU offload (slow) |
| Stage 2 | LoRA on generator and fake score; real score frozen | up to 3 copies of the backbone ≈ 98 GB + text encoder + activations ≈ **130–170 GB** | 1× B200, or 2× H100 sharded |
| LHAD | LoRA, longer rollouts | Stage 2 plus longer activations | Same as Stage 2 or more |

So the roadmap's "start on one 96 GB RTX PRO 6000" will most likely run out of memory in Stage 1.

## 6. Licences

- JoyAI code and weights: Apache-2.0.
- MiMo-VL: MIT. YuNet: MIT.
- ToyData: check the card before relying on it beyond smoke tests.
- Research try-on datasets (TikTokDress, VITON-HD, DressCode) are not for commercial training.
- Optional tools to check before use: SAM 2 (Apache-2.0 code, check the weights), Grounding DINO (Apache-2.0), CVAT and Label Studio (self-host or check where the cloud version stores data).

## 7. What we can do now without GPU spend

1. Frame-ID timing in the adapter (this PR) and the app (paired PR), plus the benchmark script.
2. Training scripts for Phase B: Pod setup, a CPU-Pod asset download, the smoke-test config, a resume check, and the key-comparison script.
3. The dataset schema, the converter to WebDataset `.tar` files, the validator, and synthetic CI data (Phase C).
4. Free quality wins in the live app: Gemini garment details in the JoyAI instruction, and garment photo cleanup.

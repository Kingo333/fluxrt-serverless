# JoyAI engine (Premium HD)

> Lives in `fluxrt-serverless/joyai/` and runs when `LIVECAM_ENGINE=joyai` in the
> dual-engine image (`Dockerfile.livecam`). See `../LIVECAM_ENGINES.md` for switching.

Azyah Live Cam **Premium HD** worker for RunPod Serverless (Load Balancing).
Runs [JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit)
(Apache-2.0, 16B) on an RTX 5090 at 480×840 portrait, 24 fps.

```
browser ──wss──> Cloudflare Worker /ws ──> RunPod LB endpoint :8765
                                             │  our FastAPI adapter (this repo)
                                             │  - HMAC token check (fail closed), 5-min cap
                                             │  - FluxRT message contract <-> JoyAI protocol
                                             └─ws──> JoyAI (unmodified) on 127.0.0.1:8080
                                                     --no-preload, loaded via POST /load
```

The adapter speaks exactly the same messages as `fluxrt-serverless`, so the app
uses one client for both tiers.

## Endpoints (port 8765)

| Route | Auth | Behavior |
|---|---|---|
| `GET /ping` | none | **204** while loading, **200** when ready, **503** on error (RunPod health) |
| `GET /health` | none | State as JSON |
| `POST /warmup` · `POST /api/load` | token | Start loading; `?wait=true` blocks until ready/error |
| `GET /debug/startup` | token | Non-secret diagnostics: state, env names, disk, which weights exist |
| `WS /ws?token=…` | token | Live session |

Auth tokens are the Worker's session tokens
(`userId.garmentId.tier.maxSession.exp.<hmac>`). Tokens for the tiers in
`ACCEPTED_TIERS` (default `standard,premium`, legacy 3-part tokens count as Standard) are accepted. Admin routes read the token from
`X-Session-Token` or `?token=` (RunPod's gateway uses `Authorization`). The
process refuses to start without `SESSION_SIGNING_SECRET`.

## Message contract

Client → server: `set_reference_image {image_b64}`, `set_prompt {prompt}`,
`frame {frame_b64}`, `set_param {name, value}`, `ping`.

Server → client: `warming`, `ready`, `ack {name}`, `frame {frame_b64}`,
`error {message}`, `pong`, plus optional `status {status}`:

| status | Meaning |
|---|---|
| `waiting_face` | JoyAI's entry gate is waiting for a centered, still face (`reason` passed through) |
| `no_person` | Subject left the frame; JoyAI holds black output |
| `session_reset` | Upstream restarted the edit (or the adapter reconnected) |
| `queued` | Another session holds JoyAI (it serves one session at a time) |
| `session_limit` | 5-minute cap reached; socket closes next |

How messages map to JoyAI (verified against upstream commit `d875895`,
`deploy/xvideo/serving/serve_joyomni_streaming.py`):

| App | JoyAI |
|---|---|
| `set_prompt` (after `set_reference_image`); uses the app's short `instruction` when sent | `start {prompt, ref_image, width:480, height:840, fps:24, num_inference_steps:2, kv_reset_frames:600, input_codec:"mjpeg", output_codec:"mjpeg", use_pe:false, gate_enabled:true}` |
| `frame` | `frame_meta` + binary JPEG (dropped until JoyAI replies `started`) |
| `set_param steps/seed/kv_reset_frames` | new `start` with the value |
| mid-session `set_reference_image` / `set_prompt` (garment swap) | one coalesced `start` on the same socket |
| `set_param output_quality` | `set_output_quality` |
| ← `output_frame` + binary JPEG | `frame {frame_b64}` |
| ← `session_timeout` / socket closed | reconnect + `start` (up to 3 times), `status: session_reset` |

Behaviors worth knowing:
- JoyAI serves **one session per worker**; extra clients queue. Scale with more workers.
- JoyAI releases a session after **10 s with no output frames**; the adapter reconnects transparently.
- The app currently sends 288×512 frames for Premium (placeholder); JoyAI rescales to 480×840. Raise the capture size in `liveCamTier.ts` once quality is checked.

## Environment variables

| Name | Default | Notes |
|---|---|---|
| `SESSION_SIGNING_SECRET` | — | **Required.** Same value as the Cloudflare Worker and fluxrt-serverless |
| `MAX_SESSION_SECONDS` | `300` | Streaming cap after `ready` (token can only lower it) |
| `PORT` / `PORT_HEALTH` | `8765` | RunPod LB port + health port |
| `AUTO_LOAD` | `true` | Start loading JoyAI at boot (so `/ping` goes 204 → 200) |
| `LOAD_TIMEOUT_S` | `2400` | Max wait for JoyAI `POST /load` |
| `PING_READY_WHILE_LOADING` | `false` | `true`: `/ping` answers 200 while loading, so RunPod (which stops workers still answering 204 after ~8 min) lets a slow first start finish. Sessions wait with `warming`. |
| `UPSTREAM_BOOT_TIMEOUT_S` | `300` | Max wait for JoyAI `/health` after spawn |
| `JOYAI_SESSION_WIDTH` / `_HEIGHT` | `480` / `840` | Session size sent in `start` (portrait) |
| `JOYAI_STEPS` | `2` | Denoise steps |
| `JOYAI_KV_RESET_FRAMES` | `600` | KV-cache reset interval |
| `JOYAI_MAX_TEMPORAL_IDS` | `8` | Recent-chunk memory (smaller = faster, larger = steadier) |
| `JOYAI_OUTPUT_QUALITY` | `70` | JPEG quality of returned frames |
| `JOYAI_GATE_ENABLED` | `true` | Face/person presence gates |
| `JOYAI_PROMPT_TEMPLATE` | `{prompt}` | Optional wrapper, e.g. `Replace the subject's outfit with the garment from the reference image. {prompt}` |
| `ENABLE_PERSON_GATE` | `false` | Use YOLOv8n body gate (AGPL-3.0, see Licenses) |
| `JOYOMNI_CKPT_ROOT` | `/runpod-volume/joyai/checkpoints` | Weights on the network volume |
| `JOYOMNI_CACHE_ROOT` | `/runpod-volume/joyai/cache_rtx5090` | torchinductor / triton / nv caches (persist across workers) |
| `JOYOMNI_SAGE_ATTN` / `JOYOMNI_FP8_FAST_ACCUM` / `JOYOMNI_LOW_VRAM` | `1` / `1` / `1` | Upstream's RTX 5090 settings |
| `JOYOMNI_FPS` | `24` | Target fps |
| `LOG_LEVEL` | `INFO` | |

Prompt enhancement is off (D3): `--no-use-pe`, and `OPENAI_*`/`PE_MODEL` are
unset before JoyAI starts. Session recording is off: JoyAI is launched without
`--record-dir`, so camera frames are never written to disk.

## Image

`../Dockerfile.livecam` builds on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` with Python
3.10, JoyAI pinned to `d875895`, SageAttention `d1a57a5` + upstream's
CUDA-graph patch (`TORCH_CUDA_ARCH_LIST=12.0`), and `joyomni_ops` built for
`120a` against cutlass `dcf215af`. **No weights in the image.**

GitHub Actions (`build-livecam-dual.yml`) pushes
`ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1` (+ `-<sha>`) from `main` and `feat/livecam-joyai-engine`.
The CUDA builds are heavy for a standard runner; if it runs out of disk or time,
use a larger runner or build once on a pod and push.

Weights go on a network volume via `joyai/scripts/populate_volume.sh` (see
`RUNPOD_SETUP.md`).

## Test order

1. `GET /ping` → 204 during load, then 200.
2. `POST /warmup?wait=true` with `X-Session-Token` → `{"status":"ready"}`.
   `GET /debug/startup` with the token shows which weights were found.
3. `WS /ws?token=…` with a Premium token: expect `ready`, then send
   `set_reference_image`, `set_prompt`, frames.

Mint a test token without printing the secret:

```bash
python - <<'PY'
import os, time
from session_token import sign
print(sign(f"test.none.premium.300.{int(time.time()) + 600}", os.environ["SESSION_SIGNING_SECRET"]))
PY
```

Local tests (no GPU, uses a fake JoyAI): from the repo root, `pip install -r requirements-test.txt && pytest -q tests`.

## Licenses

| Component | License | Notes |
|---|---|---|
| JoyAI-Video-Edit code + weights | Apache-2.0 | Upstream repo LICENSE |
| joyomni_ops (in JoyAI repo) | Apache-2.0 | Extracted from sgl-kernel |
| MiMo-VL-7B-RL-2508 (XiaomiMiMo) | MIT | Hugging Face model card |
| YuNet `face_detection_yunet_2023mar.onnx` (OpenCV Zoo) | MIT | "All files in this directory are licensed under MIT" |
| YOLOv8n (Ultralytics) | **AGPL-3.0** (or paid Enterprise license) | Using it in a commercial service can trigger AGPL obligations. **Off by default** (`ENABLE_PERSON_GATE=false`); JoyAI then skips the body gate and keeps the face gate. Get legal sign-off or an Ultralytics license before enabling. |
| SageAttention | Apache-2.0 | Built from source |
| cutlass | BSD-3-Clause | Build-time only |

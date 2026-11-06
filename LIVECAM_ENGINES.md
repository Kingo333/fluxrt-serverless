# Live Cam engines: switch FluxRT ⇄ JoyAI on one RunPod endpoint

One image, `ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1-<sha>`, holds both engines.
The `LIVECAM_ENGINE` environment variable picks which one a worker runs.

| | FluxRT (Standard, default) | JoyAI-Video-Edit (Premium HD) |
|---|---|---|
| `LIVECAM_ENGINE` | `fluxrt` (or unset) | `joyai` |
| GPU (endpoint "GPU configuration") | **32 GB PRO** or **96 GB PRO** (also older 48 GB cards) | **32 GB PRO** (RTX 5090) only |
| Network volume | none needed | **azyah-joyai** (weights at `/runpod-volume/joyai`) |
| Resolution | input `FLUXRT_WIDTH/HEIGHT` (288×512 today), output 2× (576×1024) with the Flow Upscaler | 480×840 portrait, 24 fps |
| Code | `server.py` (unchanged features: playback scheduler, latest-frame-wins) | `joyai/` adapter + unmodified JoyAI on 127.0.0.1:8080 |

## FluxRT engine version (pinned)

FluxRT is pinned to `32206e8` (2026-06-13, "flow upscaler and taef2") in both Dockerfiles.
Change `FLUXRT_REF` on purpose, never back to `main`.

Two upgrades are **on by default**. Their weights download at warm-up, next to FLUX.2-klein-4B, at pinned revisions:

| Extension | Weights | Size | Licence | What it does |
|---|---|---|---|---|
| TAEF2 (`ENABLE_TINY_VAE`) | `madebyollin/taef2` @ `bd244eb` → `taef2/` | 11 MB | MIT | faster tiny decoder |
| Flow Upscaler (`ENABLE_FLOW_UPSCALER`) | `TensorForger/FlowUpscaler` @ `ae33429` → `FlowUpscaler/` | 237 MB | Unlicense | 2× latent upscale of every frame (59M params, about 8 ms) |

FluxRT itself is Unlicense (public domain).

Other FluxRT settings in this image:
- **Garment reference: 384×512** (`FLUXRT_REF_WIDTH/HEIGHT`, multiples of 16). FluxRT's default is 256×256; before this change we used the camera size.
  The worker now pads the garment photo with white to 3:4 instead of letting FluxRT stretch it.
  It also passes the photo as RGB; before, it passed BGR, so red and blue were swapped in the garment.
- **Frames:** input stays at the model size, output is 2× that, JPEG quality `FLUXRT_OUTPUT_JPEG_QUALITY` (82).
  The `ready` message now tells the app `engine: "fluxrt"`, the input and output size, `input_fps` 10 and `input_jpeg_quality` 0.8
  (`FLUXRT_CLIENT_HINTS=false` turns this off). The upscaler magnifies JPEG blocks, so the app sends better frames.
  FluxRT renders about 5–12 frames a second on a 5090 (more are dropped).
- **PyTorch:** the dual image now installs torch 2.9.1 cu128 for FluxRT. The old cu124 build cannot run on the Blackwell cards (5090, PRO 6000).

**Rollback.** Two options:
- Fast, no rebuild: set `ENABLE_FLOW_UPSCALER=false` and `ENABLE_TINY_VAE=false` on the endpoint.
  Add `FLUXRT_REF_WIDTH=288` and `FLUXRT_REF_HEIGHT=512` for the old reference size.
- Full: go back to the image the endpoint runs today (tag ending `62a8da4f`).
  Note it in the extension's log before P2. That image installs FluxRT `main` as it was on its build date, with the cu124 torch build.

Both engines speak the same WebSocket messages and use the same `SESSION_SIGNING_SECRET`,
so the app and the Cloudflare Worker don't change when you flip.

## FluxRT live-quality switches (2026-11-06)

All OFF by default: with none of them set the worker behaves exactly as before. Each is an
endpoint environment variable; turning one off is editing the endpoint and letting the
workers restart (minutes, no code change). `/ping` shows what they resolved to under
`live_switches`, and warm-up logs the same line.

| Variable | Default | On | What it does | Off / rollback |
|---|---|---|---|---|
| `FLUXRT_OUTPUT_PACING` | `false` | `1` | Reads FluxRT's output on its own clock (`FLUXRT_OUTPUT_POLL_HZ`, default 30) so its x4 interpolated frames reach the app, instead of only when a camera frame arrives. Adds `playback_buffer`/`playback_fps` to the `ready` hints so the app enables its jitter buffer. Both sender modes consume the frames unchanged. | unset |
| `FLUXRT_COMPOSITE` | `false` | `1` | Keeps the camera frame's face, hair and background and takes only the clothing from FluxRT (MediaPipe multiclass selfie segmenter, baked into the image). Rules by the item's category from the app: garments keep face/hair/background minus the new clothing (no face in view: background only); shoes keep face/hair only; bags face only; accessories, unknown category, no person or a failed segmentation = pass-through. `FLUXRT_COMPOSITE_FEATHER` (3 px edge), `FLUXRT_COMPOSITE_LAG_MS` (unset = one processing time), `FLUXRT_COMPOSITE_OUT_SEG_EVERY` (1). | unset |
| `FLUXRT_REF_CLEAN` | `false` | `1` | Cuts the garment out of an on-model product photo (keeps it on white) before FluxRT sees it. Flat-lay photos, photos without a person and garments under 5% of the photo are left untouched. | unset |
| `FLUXRT_CACHE_MASK` | `auto` | `seg` | Drives FluxRT's spatial cache from the segmentation: the person/garment and anything that moved are recomputed every frame, static background cells reuse the cache; everything is recomputed on the first frame, every `FLUXRT_CACHE_MASK_REFRESH` (48) frames, after a prompt/garment change, and always for shoes/bags/accessories/unknown categories (identical to today for them). `FLUXRT_CACHE_MASK_DILATE` (2 cells). | `auto` |
| `FLUXRT_ANCHOR` | `false` | `1` | `FLUXRT_ANCHOR_DELAY_S` (1.5) after a garment is set, keeps one generated frame and passes it as a third reference every frame, so the garment rendering stays consistent (needs the FluxRT patch in this image; about 20-30% more per frame). Cleared on every garment swap. | unset |
| `FLUXRT_MU` | unset | e.g. `1.9` | Fixed sampler shift instead of diffusers' resolution-based value (lower = sharper detail, higher = softer and steadier). | unset |
| `FLUXRT_INPUT_SCALE` | `1` | `2` | Asks the app for camera frames at model size x 2 (576x1024 for 288x512); FluxRT still runs at model size, the composite keeps the sharp real pixels. About 2x the upload. | `1` |

Recommended combination for the owner's app test (live endpoint values in brackets):
`FLUXRT_OUTPUT_PACING=1` with `ENABLE_PLAYBACK_SCHEDULER=true` (already true), `PLAYBACK_TARGET_FPS=24`
(was 12) and `REPEAT_LAST_FRAME_WHEN_IDLE=true` (already true): the poller refreshes the latest
output up to 30 times a second and the scheduler sends it at 24 fps. If the scheduler's target
is above the real output rate, it repeats the last frame, as today. Then `FLUXRT_COMPOSITE=1`,
`FLUXRT_REF_CLEAN=1`, `FLUXRT_INPUT_SCALE=2`; second round `FLUXRT_CACHE_MASK=seg`,
`FLUXRT_ANCHOR=1`. Quality test matrix (one at a time): `FLUXRT_WIDTH=384 FLUXRT_HEIGHT=672`
(both multiples of 16; the ready hints follow), `ENABLE_FLOW_UPSCALER=false`, `FLUXRT_MU=1.9` vs `2.5`.

If the segmenter fails to load, `FLUXRT_COMPOSITE`, `FLUXRT_REF_CLEAN` and `FLUXRT_CACHE_MASK=seg`
turn themselves off (logged, visible in `/ping`); the session runs as today.

## Switch to JoyAI

On the endpoint → **Edit endpoint**:
1. Container image: `ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1-<sha>` (once; it also runs FluxRT).
2. GPU configuration: untick **48 GB**, tick **32 GB PRO**.
3. Network volume: attach **azyah-joyai** (the endpoint then runs only in that volume's datacenter).
4. Environment variables: set `LIVECAM_ENGINE` = `joyai` and `PING_READY_WHILE_LOADING` = `true`.
   Keep the others; FluxRT-only ones are ignored.
5. Container disk: at least **60 GB** (the dual image is large).
6. Save. Workers restart; anyone mid-session is cut off.

First JoyAI start compiles kernels into the volume's cache and is slow. Warm it once:
Cloudflare Worker `POST /admin/load` (orchestrator key), or call the endpoint's
`/warmup?wait=true` with an `X-Session-Token`. `/ping` answers 204 while loading, 200 when ready.

## Switch back to FluxRT (fallback)

Edit endpoint → `LIVECAM_ENGINE` = `fluxrt` (or delete it) → tick **48 GB** and untick **32 GB PRO**
→ the volume can stay attached (FluxRT ignores it) → Save.

## Notes

- **Max workers must be ≥ 1** or nothing runs (the endpoint showed 0).
- **`PING_READY_WHILE_LOADING=true` is needed for JoyAI.** RunPod stops a load-balancing worker
  whose `/ping` still answers 204 after about 8 minutes, and a first JoyAI start (48 GB of weights
  plus kernel compile) takes longer, so the worker restarts forever. With this set, `/ping`
  answers 200 while loading and a session that arrives early gets `warming` until JoyAI is ready.
  Errors still answer 503.
- JoyAI serves **one session per worker**; raise Max workers for more shoppers.
- Unknown `LIVECAM_ENGINE` values stop the container instead of guessing.
- `ACCEPTED_TIERS` (default `standard,premium`) lets either engine accept tokens for either
  tier, so flipping needs no Worker change. Set `ACCEPTED_TIERS=premium` on a dedicated
  Premium endpoint if you later run both engines side by side (same image, two endpoints).
- `MAX_SESSION_SECONDS` (default 300) caps streaming time in both engines.
- The old FluxRT-only tags (`fluxrt-playback-scheduler-v1`, `fluxrt-bootstrap-v1`, …) are
  untouched; switching the image back to one of them is the full rollback.
- YOLOv8n person gate is off (`ENABLE_PERSON_GATE=false`): it's AGPL-3.0. The face gate stays on.

See `joyai/README.md` for the adapter, message mapping, env vars and licenses, and
`joyai/RUNPOD_SETUP.md` for creating and filling the volume.

## Option B: two endpoints from the same image (Standard and Premium side by side)

Same image, no code change: a second Load Balancing endpoint with `LIVECAM_ENGINE=joyai`.
Then the Cloudflare Worker's `PREMIUM_RUNPOD_LB_ENDPOINT_ID` points at it, Standard keeps
running on the existing endpoint, and the app's Premium HD button reaches JoyAI.

| Premium endpoint setting | Value |
|---|---|
| Type | Load Balancing (same as Standard), HTTP port **8765** |
| Image | `ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1-<sha>` + GHCR registry credentials (package is private) |
| GPU | **32 GB PRO** (RTX 5090) only |
| Max workers | your concurrent Premium sessions (one shopper per GPU) |
| Active workers | 1 during testing hours, 0 otherwise (a warm worker bills all the time) |
| Idle timeout | 300 s (longer than Standard, so shoppers don't hit cold starts) |
| FlashBoot | On |
| Network volume | `azyah-joyai` (locks the endpoint to that datacenter) |
| Container disk | start at 60 GB; lower it once you see the pulled image size |
| Env | `LIVECAM_ENGINE=joyai`, `SESSION_SIGNING_SECRET` (same as the Worker, as a RunPod Secret), `ACCEPTED_TIERS=premium`, `MAX_SESSION_SECONDS=300` |

## JoyAI tuning (Lucy-VTON-like feel)

JoyAI reaches real-time with 2 denoising steps, FP8 weights and short chunks, keeping the
first chunk as an anchor so the person doesn't drift. JD reports ~30 fps at 720×1280 on a
B200; there are no published RTX 5090 numbers, so treat these as starting points.

| Knob | Setting | Default here | Notes |
|---|---|---|---|
| Denoising steps | `JOYAI_STEPS` | `2` | The distilled real-time path |
| FP8 | `JOYOMNI_FP8_IMG` / `JOYOMNI_FP8_TXT` | `1` / `1` | Plus `JOYOMNI_FP8_FAST_ACCUM=1` on the 5090 |
| Compile / CUDA graphs | `JOYOMNI_CUDA_GRAPH` | `1` | torch.compile caches persist on the volume |
| Attention / precision | `JOYOMNI_SAGE_ATTN`, `JOYOMNI_FP8_FAST_ACCUM`, `JOYOMNI_LOW_VRAM` | per GPU | 5090: int8 SageAttention, fp16 accumulate, low-VRAM (needed). RTX PRO 6000: cuDNN, fp32 accumulate, full layout (more faithful). Picked by `joyai/start_upstream.sh` |
| Output JPEG | `JOYAI_OUTPUT_QUALITY` | `90` | Was 70, which blurred prints and logos |
| Camera frames | sent to the app in `ready` | 480×840, 24 fps, JPEG 0.85 | The app adopts these when the engine is JoyAI |
| Mode | reference image in `start` | on | Garment photo drives the edit (IV2V) |
| Garment swap | app → `set_reference_image` + `set_prompt` | live | Coalesced into one JoyAI restart on the same socket, no reconnect |
| Prompt | app's `instruction` | "Substitute the current top with the garment in the reference image. …" | Optional wrapper: `JOYAI_PROMPT_TEMPLATE` (default `{prompt}`) |
| Chunk size / first-chunk anchor | built into JoyAI | native | Not exposed as settings upstream |
| Recent-chunk memory | `JOYAI_MAX_TEMPORAL_IDS` | `8` | Smaller = faster, larger = steadier |
| KV reset | `JOYAI_KV_RESET_FRAMES` | `600` | |
| Resolution | `JOYAI_SESSION_WIDTH/HEIGHT` | `480` / `840` | For 720p also set `JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720`, session `720`×`1248`; only if the 5090 holds 24 fps |
| Boot warm-up | built into JoyAI's load | on | `POST /load` runs a 4-chunk warm-up in both orientations before `/ping` turns 200 |
| CUDA / PyTorch | image | CUDA 12.8, torch 2.9.1 cu128 | Blackwell (sm_120a) |

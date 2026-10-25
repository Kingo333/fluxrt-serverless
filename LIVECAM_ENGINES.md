# Live Cam engines: switch FluxRT ⇄ JoyAI on one RunPod endpoint

One image, `ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1-<sha>`, holds both engines.
The `LIVECAM_ENGINE` environment variable picks which one a worker runs.

| | FluxRT (Standard, default) | JoyAI-Video-Edit (Premium HD) |
|---|---|---|
| `LIVECAM_ENGINE` | `fluxrt` (or unset) | `joyai` |
| GPU (endpoint "GPU configuration") | today's card: **48 GB** | **32 GB PRO** (RTX 5090) only |
| Network volume | none needed | **azyah-joyai** (weights at `/runpod-volume/joyai`) |
| Resolution | `FLUXRT_WIDTH/HEIGHT` (288×512 today) | 480×840 portrait, 24 fps |
| Code | `server.py` (unchanged features: playback scheduler, latest-frame-wins) | `joyai/` adapter + unmodified JoyAI on 127.0.0.1:8080 |

Both engines speak the same WebSocket messages and use the same `SESSION_SIGNING_SECRET`,
so the app and the Cloudflare Worker don't change when you flip.

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

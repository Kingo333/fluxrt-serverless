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
4. Environment variables: set `LIVECAM_ENGINE` = `joyai`. Keep the others; FluxRT-only ones are ignored.
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

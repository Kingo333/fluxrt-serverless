# RunPod setup for Premium HD (do by hand, or approve Claude to do it)

Nothing here has been created yet. Values marked **check** could not be
verified from this session (no RunPod access).

## 1. Network volume (weights + compile caches)

| Setting | Value |
|---|---|
| Name | `azyah-joyai` |
| Size | **100 GB** (weights ~51 GB + compile caches + headroom) |
| Region / datacenter | One that offers **RTX 5090 for Serverless** — **check** in the volume-create screen; the endpoint can only use GPUs in the volume's datacenter |

## 2. Fill the volume (one-off pod)

1. Deploy a cheap **CPU pod** (or any GPU pod) in the same datacenter with the
   volume attached (mounted at `/workspace`). Container disk 20 GB.
2. In the pod terminal:
   ```bash
   curl -fsSLO https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/feat/livecam-joyai-engine/joyai/scripts/populate_volume.sh
   # If the repo is private, curl gets a 404: open the file on GitHub, view Raw,
   # and paste it into /workspace/populate_volume.sh instead.
   # optional: export HF_TOKEN=...   (not required for these public repos)
   VOLUME_ROOT=/workspace/joyai bash populate_volume.sh
   ```
   Add `--with-yolo` only after the AGPL question is settled (README → Licenses).
3. Stop and delete the pod. The volume keeps the files.

## 3. Serverless endpoint (Load Balancing)

> Using one endpoint for both engines? Follow `../LIVECAM_ENGINES.md` instead of creating a new one.

| Setting | Value |
|---|---|
| Endpoint type | **Load Balancing** (HTTP/WebSocket), same as the FluxRT endpoint |
| Container image | `ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1-<sha>` (pin the sha) + env `LIVECAM_ENGINE=joyai` |
| Registry auth | GHCR credentials if the package is private |
| GPU | **RTX 5090 (32 GB)** only |
| Network volume | `azyah-joyai` (mounted at `/runpod-volume`) |
| Container disk | **60 GB** (dual image: CUDA devel + two torch stacks + kernels) |
| Ports | HTTP **8765** only |
| Min workers | `0` while testing (`1` for instant starts, billed while idle) |
| Max workers | `1` to start (one session per worker), raise later |
| Idle timeout | `300` s or more while testing |
| Execution timeout | at least `600` s (WebSocket sessions are 5 min + warm-up) |

### Environment variables

| Name | Value for testing |
|---|---|
| `SESSION_SIGNING_SECRET` | **same secret** as the Cloudflare Worker (RunPod secret reference, e.g. `{{ RUNPOD_SECRET_session_signing_secret }}`) |
| `PORT` | `8765` |
| `PORT_HEALTH` | `8765` |
| `MAX_SESSION_SECONDS` | `300` |
| `AUTO_LOAD` | `true` |
| `ENABLE_PERSON_GATE` | `false` |
| `LOG_LEVEL` | `INFO` |

Everything else uses image defaults (README → Environment variables).

## 4. First start (fills the compile cache)

The very first worker compiles kernels and writes them to
`/runpod-volume/joyai/cache_rtx5090`. Expect it to be much slower than later
starts (**check** the actual time in the logs). Warm it once before testing the app:

```bash
TOKEN=$(python - <<'PY'
import os, time; from session_token import sign
print(sign(f"admin.warmup.premium.60.{int(time.time())+3000}", os.environ["SESSION_SIGNING_SECRET"]))
PY
)
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://<endpoint-id>.api.runpod.ai/ping -o /dev/null -w "%{http_code}\n"
curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "X-Session-Token: $TOKEN" \
  "https://<endpoint-id>.api.runpod.ai/warmup?wait=true"
```

Or, once the Worker is deployed: `POST /admin/load {"tier":"premium"}` on the
Cloudflare Worker with the orchestrator key.

## 5. Connect it to the app

1. Cloudflare Worker: set `PREMIUM_RUNPOD_LB_ENDPOINT_ID=<endpoint-id>` and
   `PREMIUM_FALLBACK_TO_STANDARD=false` once Premium works end to end.
2. App: raise the Premium capture size in
   `src/components/ai-studio/live-cam/liveCamTier.ts` after checking quality.

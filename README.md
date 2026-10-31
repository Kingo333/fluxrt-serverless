# fluxrt-serverless

> **Two engines, one image:** `Dockerfile.livecam` builds FluxRT + JoyAI-Video-Edit; set
> `LIVECAM_ENGINE=fluxrt|joyai` on the RunPod endpoint. See [LIVECAM_ENGINES.md](LIVECAM_ENGINES.md).

Clean RunPod Serverless Load Balancer wrapper for TensorForger FluxRT.

Reference:
- https://github.com/tensorforger/FluxRT

This repo runs FluxRT as a FastAPI + WebSocket service on RunPod Load Balancer mode.

## Important

This is NOT queue-based RunPod serverless.

Do not use:

```python
runpod.serverless.start(...)
```

This service uses:

uvicorn server.py on port 8765

## Endpoints

### GET /ping

Always returns HTTP 200 as soon as uvicorn is running.

### GET /health

Same as /ping.

### GET /debug/startup

Non-secret diagnostics.

Use:

```
/debug/startup?deep=true
```

to probe torch/cv2.

### POST /warmup

Starts FluxRT model download/loading in the background.

Use:

```
/warmup?wait=true
```

to wait until ready or error.

### WS /ws?token=...

Live WebSocket frame processing.

Client messages:

```json
{"type":"set_prompt","prompt":"..."}
{"type":"set_reference_image","image_b64":"..."}
{"type":"frame","frame_b64":"..."}
{"type":"set_param","name":"steps","value":2}
```

Server messages:

```json
{"type":"warming","state":{...}}
{"type":"ready"}
{"type":"frame","frame_b64":"..."}
{"type":"error","message":"..."}
```

## Required RunPod settings

Endpoint type:
- Serverless Load Balancer / HTTP WebSocket

Docker:
- Branch: main
- Dockerfile path: Dockerfile
- Build context: .
- Container start command override: empty

Ports:
- Expose only: 8765
- PORT=8765
- PORT_HEALTH=8765

Recommended workers:
- Min workers: 1
- Max workers: 1 first, then scale later
- Idle timeout: 300 seconds or more while testing

Recommended GPU:
- ADA_24 / RTX 4090 minimum
- 48GB GPU preferred

Recommended disk:
- 80GB minimum
- 100GB preferred

## Required env vars

```
PORT=8765
PORT_HEALTH=8765
HF_TOKEN=<huggingface token if required>
SESSION_SIGNING_SECRET=<random secret>
AUTO_WARMUP=false
ENABLE_INT8=true
ENABLE_REFERENCE_IMAGE=true
FLUXRT_WIDTH=576
FLUXRT_HEIGHT=320
FLUXRT_STEPS=2
WARMUP_TIMEOUT_S=900
# FluxRT 32206e8 extensions, on by default (see LIVECAM_ENGINES.md)
ENABLE_TINY_VAE=true
ENABLE_FLOW_UPSCALER=true
FLUXRT_REF_WIDTH=384
FLUXRT_REF_HEIGHT=512
FLUXRT_OUTPUT_JPEG_QUALITY=82
FLUXRT_CLIENT_HINTS=true
FLUXRT_INPUT_FPS=10
FLUXRT_INPUT_JPEG_QUALITY=0.8
```

## Token format

```
<payload>.<signature>
```

Signature:

```
HMAC-SHA256(SESSION_SIGNING_SECRET, payload)
```

Payload may be:

```
userId.garmentId.exp_epoch
```

## Testing order

1. Deploy with AUTO_WARMUP=false.
2. Confirm /ping returns 200.
3. Confirm /debug/startup works.
4. Call /debug/startup?deep=true.
5. Call /warmup?wait=true.
6. Only after warmup succeeds, test /ws.

## Notes

FluxRT model loading is intentionally not done during container startup.
This prevents RunPod health checks from killing the worker before uvicorn
can answer /ping.


<!-- trigger rebuild: bake HF model weights (f93c64a) -->

<!-- trigger rebuild: bake HF model weights (f93c64a) -->

## Auth and session cap

- The server refuses to start if `SESSION_SIGNING_SECRET` is unset.
- `/warmup`, `/api/load`, `/api/predict`, `/debug/startup` need a session token in
  `X-Session-Token` or `?token=` (RunPod's gateway uses `Authorization`). `/ping`, `/health` stay public.
- Tokens: `userId.garmentId.tier.maxSessionSeconds.exp.<hmac>` (current) or `userId.garmentId.exp.<hmac>`
  (legacy, treated as Standard). The expiry segment is required. See `session_token.py`.
- `ACCEPTED_TIERS` (default `standard,premium`) and `MAX_SESSION_SECONDS` (default 300): after
  `ready`, a session streams for at most that long, then gets `{"type":"status","status":"session_limit"}`.

# fluxrt-serverless

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

### GET /debug/startup  (auth required)

Non-secret diagnostics. Needs a valid session token (see "Auth" below).

Use:

```
/debug/startup?deep=true
```

to probe torch/cv2.

### POST /warmup, POST /api/load  (auth required)

Starts FluxRT model download/loading in the background.

Use:

```
/warmup?wait=true
```

to wait until ready or error.

### POST /api/predict  (auth required)

Single-frame HTTP predict.

### WS /ws?token=...

Live WebSocket frame processing. Each session streams for at most
`MAX_SESSION_SECONDS` (default 300) after `ready`; the server then sends
`{"type":"status","status":"session_limit"}` and closes the socket.
Reconnecting with the same token resumes the same clock.

## Auth

The server refuses to start if `SESSION_SIGNING_SECRET` is unset.

Admin routes take the session token in an `X-Session-Token` header or a
`?token=` query parameter. The `Authorization` header is not used, because
RunPod's gateway consumes it for the RunPod API key. Only Standard-tier tokens
(and legacy 3-part tokens) are accepted.

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
{"type":"status","status":"session_limit"}
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
SESSION_SIGNING_SECRET=<random secret, same value as the Cloudflare Worker>
MAX_SESSION_SECONDS=300
AUTO_WARMUP=false
ENABLE_INT8=true
ENABLE_REFERENCE_IMAGE=true
FLUXRT_WIDTH=576
FLUXRT_HEIGHT=320
FLUXRT_STEPS=2
WARMUP_TIMEOUT_S=900
```

## Token format

```
<payload>.<hex HMAC-SHA256(SESSION_SIGNING_SECRET, payload)>
```

Payload (current, issued by the Worker):

```
userId.garmentId.tier.maxSessionSeconds.expEpochSeconds
```

Legacy payload (still accepted, treated as Standard):

```
userId.garmentId.expEpochSeconds
```

The expiry segment is required in both. See `session_token.py`.

## Testing order

1. Deploy with AUTO_WARMUP=false.
2. Confirm /ping returns 200.
3. Confirm /debug/startup returns 401 without a token, 200 with one.
4. Call /debug/startup?deep=true with a token.
5. Call /warmup?wait=true with a token.
6. Only after warmup succeeds, test /ws.

## Notes

FluxRT model loading is intentionally not done during container startup.
This prevents RunPod health checks from killing the worker before uvicorn
can answer /ping.


<!-- trigger rebuild: bake HF model weights (f93c64a) -->

<!-- trigger rebuild: bake HF model weights (f93c64a) -->

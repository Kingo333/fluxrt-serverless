# fluxrt-serverless

FluxRT FastAPI/WebSocket worker for RunPod Serverless Load Balancing endpoints.
Powers Azyah Shopping's Live Cam virtual try-on.

## What's inside

- `Dockerfile` — CUDA 12.8 base, installs FluxRT and bakes in FLUX.2-klein-4B + RIFE model weights (~16 GB) so cold starts are short.
- `server.py` — FastAPI app exposing `GET /ping` (health) and `WS /ws` (FluxRT pipeline). Verifies an HMAC-SHA256 signed token minted by the Cloudflare orchestrator Worker.
- `.dockerignore` — skip irrelevant files in the build context.

## Build & push

Requires a build machine with Docker, ~30 GB free disk, decent bandwidth.

```bash
docker login                                          # to Docker Hub (or your registry)
docker build -t YOUR_DH_USER/fluxrt-serverless:latest .
docker push YOUR_DH_USER/fluxrt-serverless:latest
```

To pin a specific upstream FluxRT commit instead of `main`:

```bash
docker build --build-arg FLUXRT_REF=<commit_sha> -t YOUR_DH_USER/fluxrt-serverless:<tag> .
```

First build takes 30–60 min (model weight downloads). Subsequent builds are layer-cached.

## Deploy on RunPod Serverless (Load Balancing)

1. RunPod console → Serverless → New Endpoint → choose **Load Balancing** type.
2. Image: `YOUR_DH_USER/fluxrt-serverless:latest`.
3. GPU: 4090 PRO (24 GB). Add A6000 / L40 as fallback if multi-GPU select is available.
4. Workers: min 0, max 2 to start. Idle timeout 30s. **FlashBoot: ON**.
5. Expose HTTP Ports: `8765`.
6. Env vars:
   - `PORT=8765`
   - `PORT_HEALTH=8765`
   - `SESSION_SIGNING_SECRET=<random hex, same value as the Cloudflare Worker secret>`
7. Save. Copy the endpoint ID — paste into the Cloudflare Worker secret `RUNPOD_LB_ENDPOINT_ID`.

## Health checks

The load balancer polls `/ping` on the same port. Returns:

- 200 once `StreamProcessor.is_ready()` is true.
- 204 while warming up.

Cold-start time is measured between first 204 and first 200.

## Client protocol (JSON over WebSocket)

Client connects to `wss://{ENDPOINT_ID}.api.runpod.ai/ws?token={signed_token}`.

Client → Server:

```json
{"type":"set_prompt", "prompt":"…"}
{"type":"set_reference_image", "image_b64":"…"}
{"type":"frame", "frame_b64":"…"}
{"type":"set_param", "name":"…", "value": "…"}
```

Server → Client:

```json
{"type":"warming"}
{"type":"ready"}
{"type":"frame", "frame_b64":"…"}
{"type":"error", "message":"…"}
```

## Token verification

The Cloudflare orchestrator Worker mints a token shaped:

```
{userId}.{garmentId}.{exp_epoch}.{hex_hmac_sha256_signature}
```

The server HMACs the `userId.garmentId.exp` payload with `SESSION_SIGNING_SECRET` and constant-time compares. Tokens expire 5 minutes after issuance. Connections without a valid token are closed with code 1008.

## Security notes

- No persistent volume is mounted. Container disk is wiped when the Serverless worker scales down.
- No frames are written to disk by `server.py`. They live in shared-memory tensors inside FluxRT only.
- Bearer tokens for RunPod and Docker are never used at runtime by the worker — only at build/deploy time on your machine.

"""
FluxRT WebSocket server for RunPod Serverless Load Balancing.

Endpoints:
  GET  /ping  -> health (200 always; body says warming|ready)
  WS   /ws    -> live cam pipeline (token query param required)

Client protocol (JSON over WS):
  Client -> Server:
    {"type":"set_prompt", "prompt":"..."}
    {"type":"set_reference_image", "image_b64":"..."}  # PNG/JPEG bytes, base64
    {"type":"frame", "frame_b64":"..."}                # JPEG bytes, base64
    {"type":"set_param", "name":"...", "value": ...}
  Server -> Client:
    {"type":"warming"}                                  # workers initializing
    {"type":"ready"}                                    # warmup complete
    {"type":"frame", "frame_b64":"..."}                 # JPEG output
    {"type":"error", "message":"..."}
"""

import asyncio
import base64
import hmac
import hashlib
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, status
from fastapi.responses import Response
from PIL import Image

# Heavy fluxrt + torch imports are deferred to warmup() so uvicorn can bind port 8765
# in <2s and answer LB /ping probes during the (potentially long) first import.
StreamProcessor = None  # lazy-loaded in warmup()
crop_maximal_rectangle = None  # lazy-loaded in warmup()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fluxrt-ws")

CONFIG_PATH = os.environ.get("FLUXRT_CONFIG", "/workspace/FluxRT/configs/stream_processor_config.json")
SESSION_SIGNING_SECRET = os.environ.get("SESSION_SIGNING_SECRET", "")
PORT = int(os.environ.get("PORT", "8765"))

# Model weights downloaded at runtime (not baked into Docker image to keep build < 30 min)
FLUXRT_DIR = os.environ.get("FLUXRT_DIR", "/workspace/FluxRT")
MODELS_DIR = os.environ.get("FLUXRT_MODELS_DIR", "/workspace/models")
FLUX_REPO_ID = os.environ.get("FLUX_REPO_ID", "black-forest-labs/FLUX.2-klein-4B")
RIFE_REPO_ID = os.environ.get("RIFE_REPO_ID", "TensorForger/RIFE-safetensors")

# --- Module-level state (must exist BEFORE lifespan / warmup / handlers run) ---
processor = None
ready_event = asyncio.Event()

if not SESSION_SIGNING_SECRET:
    log.warning(
        "SESSION_SIGNING_SECRET is empty - running in DEV MODE: verify_token() will accept "
        "ALL /ws connections. Set SESSION_SIGNING_SECRET in the RunPod endpoint env vars before production."
    )

def verify_token(token: str) -> bool:
    """Verify a session token of the form '<session_id>.<sig>' where
    sig = HMAC-SHA256(SESSION_SIGNING_SECRET, session_id) hex-encoded.
    If SESSION_SIGNING_SECRET is empty, accept all tokens (dev mode)."""
    if not SESSION_SIGNING_SECRET:
        return True
    if not token or "." not in token:
        return False
    try:
        session_id, sig = token.split(".", 1)
        expected = hmac.new(
            SESSION_SIGNING_SECRET.encode("utf-8"),
            session_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False

def b64_to_bgr(b64_str: str) -> "np.ndarray":
    """Decode a base64 PNG/JPEG image into a BGR HxWx3 uint8 numpy array (OpenCV order)."""
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img)  # RGB
    return arr[:, :, ::-1].copy()  # BGR

def bgr_to_b64_jpeg(bgr: "np.ndarray", quality: int = 85) -> str:
    """Encode a BGR HxWx3 uint8 numpy array as base64-encoded JPEG."""
    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(rgb.astype(np.uint8) if rgb.dtype != np.uint8 else rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _ensure_models():
    """Download FLUX.2-klein-4B and RIFE weights into the local FluxRT working dir
    on first boot. Skips if a sentinel file already exists AND the directory contains
    a sane number of files (so a previously-interrupted download is re-attempted).
    Uses hf_transfer for multi-stream parallel downloads (HF_HUB_ENABLE_HF_TRANSFER=1).
    Revisions can be pinned via FLUX_REV / RIFE_REV env vars (default "main").
    """
    import os as _os
    from huggingface_hub import snapshot_download

    flux_local = _os.path.join(FLUXRT_DIR, "FLUX.2-klein-4B")
    rife_local = _os.path.join(FLUXRT_DIR, "RIFE-safetensors")
    flux_sentinel = _os.path.join(flux_local, ".downloaded")
    rife_sentinel = _os.path.join(rife_local, ".downloaded")
    flux_rev = _os.environ.get("FLUX_REV", "main")
    rife_rev = _os.environ.get("RIFE_REV", "main")

    def _has_enough_files(d, minimum):
        if not _os.path.isdir(d):
            return False
        n = 0
        for _root, _dirs, _files in _os.walk(d):
            n += len(_files)
            if n >= minimum:
                return True
        return False

    # FLUX.2-klein-4B has many shard files; require at least 5 to consider it complete.
    if _os.path.exists(flux_sentinel) and _has_enough_files(flux_local, 5):
        log.info("FLUX weights already present (rev=%s), skipping download", flux_rev)
    else:
        log.info("Downloading FLUX weights (rev=%s) to %s ...", flux_rev, flux_local)
        _os.makedirs(flux_local, exist_ok=True)
        snapshot_download(
            repo_id=FLUX_REPO_ID,
            revision=flux_rev,
            local_dir=flux_local,
            local_dir_use_symlinks=False,
            max_workers=8,
        )
        with open(flux_sentinel, "w") as f:
            f.write(flux_rev)
        log.info("FLUX weights ready")

    if _os.path.exists(rife_sentinel) and _has_enough_files(rife_local, 1):
        log.info("RIFE weights already present (rev=%s), skipping download", rife_rev)
    else:
        log.info("Downloading RIFE weights (rev=%s) to %s ...", rife_rev, rife_local)
        _os.makedirs(rife_local, exist_ok=True)
        snapshot_download(
            repo_id=RIFE_REPO_ID,
            revision=rife_rev,
            local_dir=rife_local,
            local_dir_use_symlinks=False,
            max_workers=8,
        )
        with open(rife_sentinel, "w") as f:
            f.write(rife_rev)
        log.info("RIFE weights ready")

@asynccontextmanager
async def lifespan(app):
    """Non-blocking lifespan: start FastAPI immediately, warm up FluxRT in background.
    /ping returns 200 always (body says warming|ready) so RunPod's load balancer marks
    the worker healthy as soon as uvicorn is up, even before the model finishes loading.
    On unrecoverable warmup failure we os._exit(1) so RunPod restarts the worker cleanly
    rather than leaving it stuck in 'warming' forever.
    """
    global processor

    async def warmup():
        # Lazy import: fluxrt pulls torch+CUDA which is slow (20-60s on cold container).
        # Doing it here keeps uvicorn responsive on /ping during the warmup window.
        global StreamProcessor, crop_maximal_rectangle
        from fluxrt import StreamProcessor as _SP
        from fluxrt.utils import crop_maximal_rectangle as _crm
        StreamProcessor = _SP
        crop_maximal_rectangle = _crm
        log.info("FluxRT imports loaded in warmup")
        global processor
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, _ensure_models)
            log.info("Loading FluxRT StreamProcessor from %s", CONFIG_PATH)
            processor = await loop.run_in_executor(None, StreamProcessor, CONFIG_PATH)
            await loop.run_in_executor(None, processor.start)
            for i in range(600):
                try:
                    if processor.is_ready():
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            ready_event.set()
            log.info("StreamProcessor ready after warmup")
        except Exception as e:
            log.exception("Warmup failed, exiting so RunPod restarts the worker: %s", e)
            # Give logs a moment to flush, then exit so the platform restarts us cleanly.
            await asyncio.sleep(1)
            os._exit(1)

    task = asyncio.create_task(warmup())
    try:
        yield
    finally:
        task.cancel()
        try:
            if processor is not None:
                processor.stop()
        except Exception:
            pass

app = FastAPI(lifespan=lifespan)

@app.get("/ping")
async def ping():
    """Health endpoint. Always returns HTTP 200 so the RunPod load balancer marks the
    worker healthy as soon as uvicorn binds the port; the JSON body distinguishes
    'warming' (model still loading) from 'ready' (can serve frames)."""
    if ready_event.is_set():
        return {"status": "ready"}
    return {"status": "warming"}

@app.websocket("/ws")
async def ws(websocket: WebSocket, token: str = Query(default="")):
    if not verify_token(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    if not ready_event.is_set():
        await websocket.send_json({"type": "warming"})
    await ready_event.wait()
    await websocket.send_json({"type": "ready"})

    assert processor is not None
    input_t = processor.get_input_tensor()
    output_t = processor.get_output_tensor()
    res = processor.get_resolution()
    H, W = int(res["height"]), int(res["width"])

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                await websocket.send_json({"type": "error", "message": "invalid_json"})
                continue

            t = data.get("type")
            if t == "set_prompt":
                processor.set_prompt(str(data.get("prompt", "")))
            elif t == "set_reference_image":
                try:
                    ref = b64_to_bgr(data["image_b64"])
                    processor.set_reference_image(ref)
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"set_ref_failed:{e}"})
            elif t == "set_param":
                try:
                    processor.set_param(data["name"], data["value"])
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"set_param_failed:{e}"})
            elif t == "frame":
                try:
                    frame = b64_to_bgr(data["frame_b64"])
                    resized = crop_maximal_rectangle(frame, H, W)
                    input_t.copy_from(resized)
                    out = output_t.to_numpy()
                    await websocket.send_json({"type": "frame", "frame_b64": bgr_to_b64_jpeg(out)})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"frame_failed:{e}"})
            else:
                await websocket.send_json({"type": "error", "message": f"unknown_type:{t}"})
    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception as e:
        log.exception("ws error: %s", e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

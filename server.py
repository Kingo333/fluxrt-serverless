"""
FluxRT WebSocket server for RunPod Serverless Load Balancing.

Endpoints:
  GET  /ping            -> health (always HTTP 200; body says warming|ready)
  GET  /debug/startup   -> non-secret diagnostic snapshot
  WS   /ws              -> live cam pipeline (token query param required)

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

Session token format (verified by verify_token):
    "<payload>.<sig>" where payload may itself contain dots, e.g.
    "<userId>.<garmentId>.<exp_epoch>" or just "<session_id>".
    sig = HMAC-SHA256(SESSION_SIGNING_SECRET, payload) hex-encoded.
    If SESSION_SIGNING_SECRET is empty, all tokens are accepted (dev mode).
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
from PIL import Image

# Heavy fluxrt + torch + cv2 imports are deferred to a synchronous warmup_sync()
# that runs entirely in a worker thread (run_in_executor). This keeps uvicorn able
# to bind port 8765 and answer LB /ping probes during the (potentially long) cold
# import / model-download / CUDA-init phase.
StreamProcessor = None  # lazy-loaded in warmup_sync()
crop_maximal_rectangle = None  # lazy-loaded in warmup_sync()

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
startup_error = None  # last warmup failure message (for /debug/startup)

if not SESSION_SIGNING_SECRET:
    log.warning(
        "SESSION_SIGNING_SECRET is empty - running in DEV MODE: verify_token() will accept "
        "ALL /ws connections. Set SESSION_SIGNING_SECRET in the RunPod endpoint env vars before production."
    )

def verify_token(token: str) -> bool:
    """Verify a session token of the form '<payload>.<sig>' where payload may itself
    contain dots (e.g. '<userId>.<garmentId>.<exp_epoch>') and
    sig = HMAC-SHA256(SESSION_SIGNING_SECRET, payload) hex-encoded.
    Uses rsplit so any number of dots in the payload is fine. Validates exp_epoch if
    the payload has a numeric last segment that looks like a unix timestamp.
    If SESSION_SIGNING_SECRET is empty, accept all tokens (dev mode)."""
    if not SESSION_SIGNING_SECRET:
        return True
    if not token or "." not in token:
        return False
    try:
        payload, sig = token.rsplit(".", 1)
        expected = hmac.new(
            SESSION_SIGNING_SECRET.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return False
        parts = payload.split(".")
        if parts:
            last = parts[-1]
            if last.isdigit() and len(last) >= 9:
                try:
                    if int(last) < int(time.time()):
                        return False
                except Exception:
                    pass
        return True
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
    """Download FLUX.2-klein-4B and RIFE weights into the local FluxRT working dir."""
    import os as _os
    from huggingface_hub import snapshot_download

    flux_local = _os.path.join(FLUXRT_DIR, "FLUX.2-klein-4B")
    rife_local = _os.path.join(FLUXRT_DIR, "RIFE-safetensors")
    flux_sentinel = _os.path.join(flux_local, ".downloaded")
    rife_sentinel = _os.path.join(rife_local, ".downloaded")
    flux_rev = _os.environ.get("FLUX_REV", "main")
    rife_rev = _os.environ.get("RIFE_REV", "main")
    hf_token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def _has_enough_files(d, minimum):
        if not _os.path.isdir(d):
            return False
        n = 0
        for _root, _dirs, _files in _os.walk(d):
            n += len(_files)
            if n >= minimum:
                return True
        return False

    if _os.path.exists(flux_sentinel) and _has_enough_files(flux_local, 5):
        log.info("FLUX weights already present (rev=%s), skipping download", flux_rev)
    else:
        log.info("Downloading FLUX weights (rev=%s) to %s ...", flux_rev, flux_local)
        _os.makedirs(flux_local, exist_ok=True)
        snapshot_download(repo_id=FLUX_REPO_ID, revision=flux_rev, local_dir=flux_local,
            local_dir_use_symlinks=False, max_workers=8, token=hf_token)
        with open(flux_sentinel, "w") as f:
            f.write(flux_rev)
        log.info("FLUX weights ready")

    if _os.path.exists(rife_sentinel) and _has_enough_files(rife_local, 1):
        log.info("RIFE weights already present (rev=%s), skipping download", rife_rev)
    else:
        log.info("Downloading RIFE weights (rev=%s) to %s ...", rife_rev, rife_local)
        _os.makedirs(rife_local, exist_ok=True)
        snapshot_download(repo_id=RIFE_REPO_ID, revision=rife_rev, local_dir=rife_local,
            local_dir_use_symlinks=False, max_workers=8, token=hf_token)
        with open(rife_sentinel, "w") as f:
            f.write(rife_rev)
        log.info("RIFE weights ready")

def warmup_sync():
    """Synchronous warmup: import heavy deps (fluxrt -> cv2/torch), download weights,
    construct the StreamProcessor, and start it. Runs entirely in a worker thread via
    run_in_executor so the asyncio event loop (and /ping) stays responsive.
    Returns (processor, StreamProcessor_class, crop_maximal_rectangle) or raises.
    """
    log.info("warmup_sync: importing fluxrt (this pulls cv2 + torch)")
    from fluxrt import StreamProcessor as _SP
    from fluxrt.utils import crop_maximal_rectangle as _crm
    log.info("warmup_sync: fluxrt imports OK")
    _ensure_models()
    log.info("warmup_sync: building StreamProcessor from %s", CONFIG_PATH)
    proc = _SP(CONFIG_PATH)
    proc.start()
    for _ in range(600):
        try:
            if proc.is_ready():
                break
        except Exception:
            pass
        time.sleep(1)
    log.info("warmup_sync: StreamProcessor ready")
    return proc, _SP, _crm

@asynccontextmanager
async def lifespan(app):
    """Non-blocking lifespan: start FastAPI immediately, warm up FluxRT in background."""
    global processor, StreamProcessor, crop_maximal_rectangle, startup_error

    async def warmup():
        global processor, StreamProcessor, crop_maximal_rectangle, startup_error
        loop = asyncio.get_event_loop()
        try:
            proc, _SP, _crm = await loop.run_in_executor(None, warmup_sync)
            StreamProcessor = _SP
            crop_maximal_rectangle = _crm
            processor = proc
            ready_event.set()
            log.info("warmup: ready_event set, worker can serve frames")
        except Exception as e:
            startup_error = f"{type(e).__name__}: {e}"
            log.exception("warmup failed, exiting so RunPod restarts the worker: %s", e)
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
    """Health endpoint. Always returns HTTP 200."""
    if ready_event.is_set():
        return {"status": "ready"}
    return {"status": "warming"}

@app.get("/debug/startup")
async def debug_startup():
    """Non-secret diagnostic snapshot of worker startup state."""
    info = {
        "port": PORT,
        "config_path": CONFIG_PATH,
        "config_path_exists": os.path.exists(CONFIG_PATH),
        "session_secret_configured": bool(SESSION_SIGNING_SECRET),
        "hf_token_configured": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "ready_event": ready_event.is_set(),
        "processor_loaded": processor is not None,
        "startup_error": startup_error,
        "fluxrt_dir": FLUXRT_DIR,
        "fluxrt_dir_exists": os.path.isdir(FLUXRT_DIR),
        "flux_weights_dir_exists": os.path.isdir(os.path.join(FLUXRT_DIR, "FLUX.2-klein-4B")),
        "rife_weights_dir_exists": os.path.isdir(os.path.join(FLUXRT_DIR, "RIFE-safetensors")),
    }
    try:
        import torch  # type: ignore
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:
        info["torch_import_error"] = f"{type(e).__name__}: {e}"
    try:
        import cv2  # type: ignore
        info["cv2_version"] = cv2.__version__
    except Exception as e:
        info["cv2_import_error"] = f"{type(e).__name__}: {e}"
    return info

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

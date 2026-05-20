import asyncio
import base64
import hashlib
import hmac
import io
import json
import logging
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("fluxrt-lb")

PORT = int(os.getenv("PORT", "8765"))

FLUXRT_ROOT = Path(os.getenv("FLUXRT_ROOT", "/workspace/FluxRT"))
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", str(FLUXRT_ROOT)))

FLUX_REPO_ID = os.getenv("FLUX_REPO_ID", "black-forest-labs/FLUX.2-klein-4B")
RIFE_REPO_ID = os.getenv("RIFE_REPO_ID", "TensorForger/RIFE-safetensors")
INT8_REPO_ID = os.getenv("INT8_REPO_ID", "aydin99/FLUX.2-klein-4B-int8")

ENABLE_INT8 = os.getenv("ENABLE_INT8", "true").strip().lower() in ("1", "true", "yes", "on")
ENABLE_REFERENCE_IMAGE = os.getenv("ENABLE_REFERENCE_IMAGE", "true").strip().lower() in ("1", "true", "yes", "on")

WIDTH = int(os.getenv("FLUXRT_WIDTH", "576"))
HEIGHT = int(os.getenv("FLUXRT_HEIGHT", "320"))
DEFAULT_STEPS = int(os.getenv("FLUXRT_STEPS", "2"))
DEFAULT_SEED = int(os.getenv("FLUXRT_SEED", "52"))
DEFAULT_PROMPT = os.getenv("DEFAULT_PROMPT", "Transform this live camera frame into a realistic fashion try-on preview.")
WARMUP_TIMEOUT_S = int(os.getenv("WARMUP_TIMEOUT_S", "900"))

SESSION_SIGNING_SECRET = os.getenv("SESSION_SIGNING_SECRET", "")

STATE_LOCK = threading.RLock()
STATE: Dict[str, Any] = {
    "status": "cold",
    "error": None,
    "processor": None,
    "crop_fn": None,
    "warmup_started_at": None,
    "ready_at": None,
    "warmup_thread": None,
}


def verify_token(token: str) -> bool:
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
            maybe_exp = parts[-1]
            if maybe_exp.isdigit() and len(maybe_exp) >= 9:
                if int(maybe_exp) < int(time.time()):
                    return False

        return True
    except Exception:
        return False


def _strip_data_url(value: str) -> str:
    if "," in value and value.strip().lower().startswith("data:"):
        return value.split(",", 1)[1]
    return value


def b64_to_bgr(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(_strip_data_url(b64_str))
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img)
    return arr[:, :, ::-1].copy()


def bgr_to_b64_jpeg(bgr: np.ndarray, quality: int = 85) -> str:
    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(rgb.astype(np.uint8) if rgb.dtype != np.uint8 else rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _download_model(repo_id: str, local_dir: Path, min_files: int) -> None:
    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    sentinel = local_dir / ".downloaded"

    # Fast path: sentinel present with enough files.
    if sentinel.exists() and local_dir.exists():
        count = sum(1 for p in local_dir.rglob("*") if p.is_file())
        if count >= min_files:
            log.info("model already present (sentinel): %s", local_dir)
            return

    # Second fast path: image-baked models. Folder exists with enough files
    # even though the sentinel was not written. Trust the baked content and
    # write the sentinel so future starts hit the fast path above.
    if local_dir.exists() and local_dir.is_dir():
        try:
            count = sum(1 for p in local_dir.rglob("*") if p.is_file())
        except Exception:
            count = 0
        if count >= min_files:
            log.info(
                "model already present (baked, %d files): %s",
                count,
                local_dir,
            )
            try:
                sentinel.write_text(str(time.time()))
            except Exception as e:
                log.warning("could not write sentinel for %s: %s", local_dir, e)
            return

    log.info("downloading %s to %s", repo_id, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
        max_workers=8,
    )
    sentinel.write_text(str(time.time()))
    log.info("download complete: %s", local_dir)


def ensure_models() -> None:
    _download_model(FLUX_REPO_ID, FLUXRT_ROOT / "FLUX.2-klein-4B", min_files=5)
    _download_model(RIFE_REPO_ID, FLUXRT_ROOT / "RIFE-safetensors", min_files=1)

    if ENABLE_INT8:
        _download_model(INT8_REPO_ID, FLUXRT_ROOT / "FLUX.2-klein-4B-int8", min_files=3)


def write_runtime_config() -> Path:
    config = {
        "default_prompt": DEFAULT_PROMPT,
        "default_steps": DEFAULT_STEPS,
        "default_seed": DEFAULT_SEED,
        "models_path": "FLUX.2-klein-4B",
        "int8_models_path": "FLUX.2-klein-4B-int8",
        "resolution": {
            "height": HEIGHT,
            "width": WIDTH,
        },
        "compile_models": False,
        "enable_spatial_cache": True,
        "enable_int8_quantization": ENABLE_INT8,
        "target_fps": None,
        "interpolation_exp": 2,
        "use_reference_image": ENABLE_REFERENCE_IMAGE,
        "reference_image_resolution": {
            "height": HEIGHT,
            "width": WIDTH,
        },
        "logging": True,
    }

    path = FLUXRT_ROOT / "config.server.runtime.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def warmup_blocking() -> None:
    with STATE_LOCK:
        if STATE["status"] == "ready":
            return
        if STATE["status"] == "loading":
            return
        STATE["status"] = "loading"
        STATE["error"] = None
        STATE["warmup_started_at"] = time.time()

    try:
        os.chdir(str(FLUXRT_ROOT))
        log.info("warmup: cwd=%s", os.getcwd())

        from fluxrt import StreamProcessor
        from fluxrt.utils import crop_maximal_rectangle

        log.info("warmup: fluxrt imports ok")

        ensure_models()

        config_path = write_runtime_config()
        log.info("warmup: config written to %s", config_path)

        proc = StreamProcessor(str(config_path))
        proc.start()
        log.info("warmup: processor started")

        start = time.time()
        while time.time() - start < WARMUP_TIMEOUT_S:
            model_proc = getattr(proc.model_inference_subprocess, "process", None)
            if model_proc is not None and not model_proc.is_alive():
                raise RuntimeError("FluxRT model_inference_subprocess exited during warmup")

            sched_proc = getattr(proc.output_scheduler_subprocess, "process", None)
            if sched_proc is not None and not sched_proc.is_alive():
                raise RuntimeError("FluxRT output_scheduler_subprocess exited during warmup")

            try:
                if proc.is_ready():
                    with STATE_LOCK:
                        STATE["processor"] = proc
                        STATE["crop_fn"] = crop_maximal_rectangle
                        STATE["status"] = "ready"
                        STATE["ready_at"] = time.time()
                    log.info("warmup: ready")
                    return
            except Exception:
                pass

            time.sleep(1)

        raise TimeoutError(f"FluxRT warmup timed out after {WARMUP_TIMEOUT_S}s")

    except Exception as e:
        log.exception("warmup failed: %s", e)
        with STATE_LOCK:
            STATE["status"] = "error"
            STATE["error"] = f"{type(e).__name__}: {e}"
        raise


def start_warmup_background() -> None:
    with STATE_LOCK:
        if STATE["status"] in ("loading", "ready"):
            return
        t = threading.Thread(target=warmup_blocking, name="fluxrt-warmup", daemon=True)
        STATE["warmup_thread"] = t
        t.start()


def public_state() -> Dict[str, Any]:
    with STATE_LOCK:
        processor = STATE.get("processor")
        return {
            "status": STATE["status"],
            "error": STATE["error"],
            "warmup_started_at": STATE["warmup_started_at"],
            "ready_at": STATE["ready_at"],
            "processor_loaded": processor is not None,
            "enable_int8": ENABLE_INT8,
            "enable_reference_image": ENABLE_REFERENCE_IMAGE,
            "resolution": {"width": WIDTH, "height": HEIGHT},
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("AUTO_WARMUP", "false").strip().lower() in ("1", "true", "yes", "on"):
        log.warning("AUTO_WARMUP enabled: starting FluxRT warmup in background")
        start_warmup_background()

    yield

    with STATE_LOCK:
        proc = STATE.get("processor")
    if proc is not None:
        try:
            proc.stop()
        except Exception:
            pass


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping():
    return JSONResponse({"ok": True, **public_state()})


@app.get("/health")
async def health():
    return JSONResponse({"ok": True, **public_state()})


@app.get("/debug/startup")
async def debug_startup(deep: bool = Query(False)):
    total, used, free = shutil.disk_usage("/")
    info = {
        "ok": True,
        "port": PORT,
        "cwd": os.getcwd(),
        "fluxrt_root": str(FLUXRT_ROOT),
        "fluxrt_root_exists": FLUXRT_ROOT.exists(),
        "hf_token_configured": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")),
        "session_secret_configured": bool(SESSION_SIGNING_SECRET),
        "env_names": sorted(os.environ.keys()),
        "disk": {"total": total, "used": used, "free": free},
        **public_state(),
    }

    if deep:
        try:
            import torch
            info["torch"] = {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": torch.cuda.device_count(),
            }
        except Exception as e:
            info["torch_error"] = f"{type(e).__name__}: {e}"

        try:
            import cv2
            info["cv2_version"] = cv2.__version__
        except Exception as e:
            info["cv2_error"] = f"{type(e).__name__}: {e}"

    return JSONResponse(info)


@app.post("/warmup")
async def warmup(wait: bool = Query(False)):
    start_warmup_background()

    if wait:
        for _ in range(WARMUP_TIMEOUT_S):
            state = public_state()
            if state["status"] in ("ready", "error"):
                return JSONResponse(state)
            await asyncio.sleep(1)

    return JSONResponse(public_state())


@app.websocket("/ws")
async def ws(websocket: WebSocket, token: str = Query(default="")):
    if not verify_token(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    state = public_state()
    if state["status"] != "ready":
        await websocket.send_json({"type": "warming", "state": state})
        start_warmup_background()

        while True:
            state = public_state()
            if state["status"] == "ready":
                break
            if state["status"] == "error":
                await websocket.send_json({"type": "error", "message": state["error"]})
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                return
            await asyncio.sleep(1)

    await websocket.send_json({"type": "ready"})

    with STATE_LOCK:
        proc = STATE["processor"]
        crop_fn = STATE["crop_fn"]

    input_t = proc.get_input_tensor()
    output_t = proc.get_output_tensor()
    res = proc.get_resolution()
    h, w = int(res["height"]), int(res["width"])

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                await websocket.send_json({"type": "error", "message": "invalid_json"})
                continue

            msg_type = data.get("type")

            if msg_type == "set_prompt":
                proc.set_prompt(str(data.get("prompt", "")))
                await websocket.send_json({"type": "ack", "name": "set_prompt"})

            elif msg_type == "set_param":
                try:
                    proc.set_param(str(data["name"]), data["value"])
                    await websocket.send_json({"type": "ack", "name": "set_param"})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"set_param_failed:{e}"})

            elif msg_type == "set_reference_image":
                try:
                    ref = b64_to_bgr(data["image_b64"])
                    proc.set_reference_image(ref)
                    await websocket.send_json({"type": "ack", "name": "set_reference_image"})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"set_reference_image_failed:{e}"})

            elif msg_type == "frame":
                try:
                    frame = b64_to_bgr(data["frame_b64"])
                    resized = crop_fn(frame, h, w)
                    input_t.copy_from(resized)
                    output = output_t.to_numpy()
                    await websocket.send_json({
                        "type": "frame",
                        "frame_b64": bgr_to_b64_jpeg(output),
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"frame_failed:{e}"})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "state": public_state()})

            else:
                await websocket.send_json({"type": "error", "message": f"unknown_type:{msg_type}"})

    except WebSocketDisconnect:
        log.info("websocket disconnected")
    except Exception as e:
        log.exception("websocket failure: %s", e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))

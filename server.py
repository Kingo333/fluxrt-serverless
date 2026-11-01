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
from fastapi import FastAPI, Header, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from PIL import Image

from session_token import SessionClock, SessionToken, parse_and_verify

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
# FluxRT 32206e8 extensions (on by default). Revisions are pinned so the
# weights can't change under the pinned FluxRT code.
TAEF2_REPO_ID = os.getenv("TAEF2_REPO_ID", "madebyollin/taef2")  # MIT
TAEF2_REVISION = os.getenv("TAEF2_REVISION", "bd244ebfe4398c84fbf312a2f1c868c676b500e3")
FLOW_UPSCALER_REPO_ID = os.getenv("FLOW_UPSCALER_REPO_ID", "TensorForger/FlowUpscaler")  # Unlicense
FLOW_UPSCALER_REVISION = os.getenv("FLOW_UPSCALER_REVISION", "ae334296526769d7fb5065465e364049f6c45c2b")

ENABLE_INT8 = os.getenv("ENABLE_INT8", "true").strip().lower() in ("1", "true", "yes", "on")
ENABLE_REFERENCE_IMAGE = os.getenv("ENABLE_REFERENCE_IMAGE", "true").strip().lower() in ("1", "true", "yes", "on")
# TAEF2: faster tiny decoder. Flow Upscaler: 2x latent upscale of every frame
# (FluxRT recommends using it together with TAEF2). Set false only to roll back.
ENABLE_TINY_VAE = os.getenv("ENABLE_TINY_VAE", "true").strip().lower() in ("1", "true", "yes", "on")
ENABLE_FLOW_UPSCALER = os.getenv("ENABLE_FLOW_UPSCALER", "true").strip().lower() in ("1", "true", "yes", "on")

WIDTH = int(os.getenv("FLUXRT_WIDTH", "576"))
HEIGHT = int(os.getenv("FLUXRT_HEIGHT", "320"))


def _mult16(v: int) -> int:
    return max(16, (int(v) // 16) * 16)


# Garment reference size. FluxRT's own default is 256x256, too small for
# logos and prints. 384x512 (3:4) matches most product photos; cost is
# (w/16)*(h/16) extra tokens (768 here vs 576 for a 288x512 camera frame).
REF_WIDTH = _mult16(int(os.getenv("FLUXRT_REF_WIDTH", "384")))
REF_HEIGHT = _mult16(int(os.getenv("FLUXRT_REF_HEIGHT", "512")))
UPSCALE = 2 if ENABLE_FLOW_UPSCALER else 1
OUT_WIDTH = WIDTH * UPSCALE
OUT_HEIGHT = HEIGHT * UPSCALE
OUTPUT_JPEG_QUALITY = int(os.getenv("FLUXRT_OUTPUT_JPEG_QUALITY", "82"))
# Sent to the app in "ready" so it captures at the model's size and a rate
# FluxRT can keep up with (it renders ~5-12 base frames/s on a 5090; extra
# frames are dropped). The upscaler magnifies JPEG blocks, so ask for 0.8.
# Optional garment LoRA (training/fluxrt_lora). Off unless a path is set. A LoRA
# whose selection.json says it is not cleared for commercial use (e.g. trained
# with ViViD) loads only with FLUXRT_ALLOW_RESEARCH_MODEL=1, like the JoyAI gate.
LORA_PATH = os.getenv("FLUXRT_LORA_PATH", "").strip()
ALLOW_RESEARCH_MODEL = os.getenv("FLUXRT_ALLOW_RESEARCH_MODEL", "").strip() in ("1", "true", "yes")
CLIENT_HINTS = os.getenv("FLUXRT_CLIENT_HINTS", "true").strip().lower() in ("1", "true", "yes", "on")
INPUT_FPS = float(os.getenv("FLUXRT_INPUT_FPS", "10"))
INPUT_JPEG_QUALITY = float(os.getenv("FLUXRT_INPUT_JPEG_QUALITY", "0.8"))
# Typical FluxRT warm-up, shown on the app's wait screen (weights download into the container).
EXPECTED_WARMUP_S = int(os.getenv("FLUXRT_EXPECTED_WARMUP_S", "120"))
DEFAULT_STEPS = int(os.getenv("FLUXRT_STEPS", "2"))
DEFAULT_SEED = int(os.getenv("FLUXRT_SEED", "52"))
DEFAULT_PROMPT = os.getenv("DEFAULT_PROMPT", "Transform this live camera frame into a realistic fashion try-on preview.")
WARMUP_TIMEOUT_S = int(os.getenv("WARMUP_TIMEOUT_S", "900"))

ENABLE_PLAYBACK_SCHEDULER = os.getenv("ENABLE_PLAYBACK_SCHEDULER", "false").strip().lower() in ("1", "true", "yes", "on")
PLAYBACK_TARGET_FPS = float(os.getenv("PLAYBACK_TARGET_FPS", "12"))
PLAYBACK_MAX_JITTER_MS = float(os.getenv("PLAYBACK_MAX_JITTER_MS", "40"))
REPEAT_LAST_FRAME_WHEN_IDLE = os.getenv("REPEAT_LAST_FRAME_WHEN_IDLE", "false").strip().lower() in ("1", "true", "yes", "on")
ENABLE_LIVECAM_TIMING_LOG = os.getenv("ENABLE_LIVECAM_TIMING_LOG", "true").strip().lower() in ("1", "true", "yes", "on")
LIVECAM_LOG_EVERY = max(1, int(os.getenv("LIVECAM_LOG_EVERY", "30")))

SESSION_SIGNING_SECRET = os.getenv("SESSION_SIGNING_SECRET", "")
# Streaming cap per session (D4). The token can lower it, never raise it.
MAX_SESSION_SECONDS = int(os.getenv("MAX_SESSION_SECONDS", "300"))
# Token tiers this worker serves. Both by default: with LIVECAM_ENGINE one
# endpoint can serve either tier; the Cloudflare Worker decides routing.
ACCEPTED_TIERS = {t.strip() for t in os.getenv("ACCEPTED_TIERS", "standard,premium").split(",") if t.strip()}

# Fail closed: without a signing secret every token check would be meaningless.
# bootstrap.py re-raises SystemExit, so the container exits instead of serving.
if not SESSION_SIGNING_SECRET:
    raise SystemExit("SESSION_SIGNING_SECRET is not set; refusing to start")

SESSION_CLOCK = SessionClock()

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


def verify_token(token: str) -> Optional[SessionToken]:
    """Valid, unexpired token for an accepted tier, else None. Expiry segment is required."""
    tok = parse_and_verify(token, SESSION_SIGNING_SECRET)
    if tok is None or tok.tier not in ACCEPTED_TIERS:
        return None
    return tok


def unauthorized() -> JSONResponse:
    return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)


def request_token(header_token: Optional[str], query_token: Optional[str]) -> Optional[SessionToken]:
    """Admin HTTP routes take the session token in X-Session-Token or ?token=.

    The Authorization header is not used: RunPod's gateway consumes it for the
    RunPod API key.
    """
    return verify_token(header_token or query_token or "")


def _strip_data_url(value: str) -> str:
    if "," in value and value.strip().lower().startswith("data:"):
        return value.split(",", 1)[1]
    return value


def b64_to_bgr(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(_strip_data_url(b64_str))
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img)
    return arr[:, :, ::-1].copy()


def b64_to_reference_rgb(b64_str: str) -> np.ndarray:
    """Garment image for FluxRT's set_reference_image.

    FluxRT wants RGB here (camera frames are BGR). The image is padded with
    white to the reference aspect ratio first, because FluxRT stretches it
    to REF_WIDTH x REF_HEIGHT and a stretched garment loses its shape.
    """
    raw = base64.b64decode(_strip_data_url(b64_str))
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    scale = min(REF_WIDTH / img.width, REF_HEIGHT / img.height)
    w = max(1, round(img.width * scale))
    h = max(1, round(img.height * scale))
    img = img.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (REF_WIDTH, REF_HEIGHT), (255, 255, 255))
    canvas.paste(img, ((REF_WIDTH - w) // 2, (REF_HEIGHT - h) // 2))
    return np.asarray(canvas).copy()


def warming_message(state: Dict[str, Any]) -> Dict[str, Any]:
    """Progress for the app's wait screen (phase, seconds since warm-up began, typical total)."""
    started = state.get("warmup_started_at")
    elapsed = max(0.0, time.time() - started) if started else 0.0
    return {"type": "warming", "state": state, "engine": "fluxrt", "phase": state.get("status"),
            "elapsed_s": round(elapsed, 1), "expected_s": EXPECTED_WARMUP_S}


def ready_message(deadline: float) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"type": "ready", "session_deadline": int(deadline)}
    if CLIENT_HINTS:
        msg.update({
            "engine": "fluxrt",
            "input_width": WIDTH,
            "input_height": HEIGHT,
            "output_width": OUT_WIDTH,
            "output_height": OUT_HEIGHT,
            "input_fps": INPUT_FPS,
            "input_jpeg_quality": INPUT_JPEG_QUALITY,
        })
    return msg


def bgr_to_b64_jpeg(bgr: np.ndarray, quality: int = OUTPUT_JPEG_QUALITY) -> str:
    rgb = bgr[:, :, ::-1]
    img = Image.fromarray(rgb.astype(np.uint8) if rgb.dtype != np.uint8 else rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _download_model(repo_id: str, local_dir: Path, min_files: int, revision: Optional[str] = None) -> None:
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
        revision=revision,
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

    # FluxRT loads these from fixed relative paths: taef2/taef2.safetensors
    # and FlowUpscaler/flow_upscaler.safetensors.
    if ENABLE_TINY_VAE:
        _download_model(TAEF2_REPO_ID, FLUXRT_ROOT / "taef2", min_files=1, revision=TAEF2_REVISION)
    if ENABLE_FLOW_UPSCALER:
        _download_model(FLOW_UPSCALER_REPO_ID, FLUXRT_ROOT / "FlowUpscaler", min_files=1, revision=FLOW_UPSCALER_REVISION)
        if not ENABLE_TINY_VAE:
            log.warning("Flow Upscaler without TAEF2: the full VAE decodes 2x frames, expect lower fps")


def check_lora(path: str) -> str:
    """Fail closed: the file must exist and its provenance must allow serving."""
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"FLUXRT_LORA_PATH not found: {path}")
    sel = p.parent / "selection.json"
    if not sel.is_file():
        raise RuntimeError(f"no selection.json next to {path}; provenance is required")
    info = json.loads(sel.read_text())
    if info.get("commercial_use_cleared") is not True and not ALLOW_RESEARCH_MODEL:
        raise RuntimeError(f"LoRA {path} is not cleared for commercial use ({info.get('notice', '')}); "
                           "set FLUXRT_ALLOW_RESEARCH_MODEL=1 for internal testing only")
    return str(p)


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
        "enable_tiny_vae": ENABLE_TINY_VAE,
        "enable_flow_upscaler": ENABLE_FLOW_UPSCALER,
        "target_fps": None,
        "interpolation_exp": 2,
        "use_reference_image": ENABLE_REFERENCE_IMAGE,
        "reference_image_resolution": {
            "height": REF_HEIGHT,
            "width": REF_WIDTH,
        },
        "logging": True,
    }
    if LORA_PATH:
        config["use_lora"] = True
        config["lora_weights_path"] = check_lora(LORA_PATH)

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
            "output_resolution": {"width": OUT_WIDTH, "height": OUT_HEIGHT},
            "reference_resolution": {"width": REF_WIDTH, "height": REF_HEIGHT},
            "enable_tiny_vae": ENABLE_TINY_VAE,
            "enable_flow_upscaler": ENABLE_FLOW_UPSCALER,
            "lora": Path(LORA_PATH).parent.name if LORA_PATH else None,
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


@app.get("/")
async def root():
    return JSONResponse({"ok": True, "service": "fluxrt-lb", **public_state()})


@app.get("/ping")
async def ping():
    return JSONResponse({"ok": True, **public_state()})


@app.get("/health")
async def health():
    return JSONResponse({"ok": True, **public_state()})


@app.get("/debug/startup")
async def debug_startup(
    deep: bool = Query(False),
    token: Optional[str] = Query(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    if request_token(x_session_token, token) is None:
        return unauthorized()
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
async def warmup(
    wait: bool = Query(False),
    token: Optional[str] = Query(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    if request_token(x_session_token, token) is None:
        return unauthorized()
    start_warmup_background()

    if wait:
        for _ in range(WARMUP_TIMEOUT_S):
            state = public_state()
            if state["status"] in ("ready", "error"):
                return JSONResponse(state)
            await asyncio.sleep(1)

    return JSONResponse(public_state())


@app.post("/api/load")
async def api_load(
    wait: bool = Query(False),
    token: Optional[str] = Query(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    """Alias for /warmup. Triggers background model loading. Owen-style HTTP-first."""
    if request_token(x_session_token, token) is None:
        return unauthorized()
    start_warmup_background()
    if wait:
        for _ in range(WARMUP_TIMEOUT_S):
            state = public_state()
            if state["status"] in ("ready", "error"):
                return JSONResponse(state)
            await asyncio.sleep(1)
    return JSONResponse(public_state())


@app.post("/api/predict")
async def api_predict(
    payload: Dict[str, Any],
    token: Optional[str] = Query(default=None),
    x_session_token: Optional[str] = Header(default=None),
):
    """Single-frame HTTP predict endpoint (Owen-style).
    Body:
      {"base64_image": "...", "prompt": "...", "reference_image_b64": "optional"}
    Returns:
      {"base64_image": "..."} on success, or {"error": "model_not_ready"} if
      warmup has not completed yet.
    """
    if request_token(x_session_token, token) is None:
        return unauthorized()

    state = public_state()
    if state["status"] != "ready":
        return JSONResponse({"error": "model_not_ready", "state": state}, status_code=503)

    try:
        b64 = payload.get("base64_image")
        if not isinstance(b64, str) or not b64:
            return JSONResponse({"error": "missing_base64_image"}, status_code=400)

        prompt = payload.get("prompt")
        ref_b64 = payload.get("reference_image_b64")

        with STATE_LOCK:
            proc = STATE["processor"]
            crop_fn = STATE["crop_fn"]

        if proc is None or crop_fn is None:
            return JSONResponse({"error": "model_not_ready"}, status_code=503)

        if isinstance(prompt, str) and prompt:
            try:
                proc.set_prompt(prompt)
            except Exception as e:
                log.warning("set_prompt failed: %s", e)

        if isinstance(ref_b64, str) and ref_b64:
            try:
                ref = b64_to_reference_rgb(ref_b64)
                proc.set_reference_image(ref)
            except Exception as e:
                log.warning("set_reference_image failed: %s", e)

        frame = b64_to_bgr(b64)
        res = proc.get_resolution()
        h, w = int(res["height"]), int(res["width"])
        resized = crop_fn(frame, h, w)

        input_t = proc.get_input_tensor()
        output_t = proc.get_output_tensor()
        input_t.copy_from(resized)
        output = output_t.to_numpy()

        return JSONResponse({"base64_image": bgr_to_b64_jpeg(output)})
    except Exception as e:
        log.exception("api_predict failed")
        return JSONResponse(
            {"error": "predict_failed", "message": f"{type(e).__name__}: {e}"},
            status_code=500,
        )


@app.websocket("/ws")
async def ws(websocket: WebSocket, token: str = Query(default="")):
    session_tok = verify_token(token)
    if session_tok is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    state = public_state()
    if state["status"] != "ready":
        start_warmup_background()
        last_sent = 0.0
        while True:
            state = public_state()
            if state["status"] == "ready":
                break
            if state["status"] == "error":
                await websocket.send_json({"type": "error", "message": state["error"]})
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                return
            if time.monotonic() - last_sent >= 3.0:
                await websocket.send_json(warming_message(state))
                last_sent = time.monotonic()
            await asyncio.sleep(1)

    deadline = SESSION_CLOCK.deadline(session_tok, MAX_SESSION_SECONDS)
    if time.time() >= deadline:
        await websocket.send_json({"type": "status", "status": "session_limit"})
        await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
        return

    await websocket.send_json(ready_message(deadline))

    with STATE_LOCK:
        proc = STATE["processor"]
        crop_fn = STATE["crop_fn"]

    input_t = proc.get_input_tensor()
    output_t = proc.get_output_tensor()
    res = proc.get_resolution()
    h, w = int(res["height"]), int(res["width"])

    # ------------------------------------------------------------------
    # Latest-frame-wins live-cam pipeline.
    #
    # Receiver task: continuously reads WebSocket messages. set_prompt /
    # set_reference_image / set_param / ping are handled inline. For
    # "frame" messages we keep only the NEWEST pending frame; if a
    # previous pending frame is still unprocessed we DROP it (and bump
    # dropped_frame_count). Max backlog is 1.
    #
    # Processor task: loops, takes the latest pending frame (if any),
    # runs it through FluxRT, publishes output to sender hand-off state.
    #
    # Sender task: waits on output_event and sends frames to client,
    # either immediately (scheduler off) or on cadence (scheduler on).
    #
    # An asyncio.Event coordinates clean shutdown when either side
    # finishes (disconnect / error). No base64 is ever logged.
    # ------------------------------------------------------------------

    pending_lock = asyncio.Lock()
    pending_frame_b64: Optional[str] = None
    pending_recv_ts: float = 0.0
    dropped_frame_count = 0
    stop_event = asyncio.Event()

    # Track B: sender hand-off state.
    latest_output_b64: Optional[str] = None
    latest_output_seq: int = 0
    latest_output_ts: float = 0.0
    last_sent_seq: int = 0
    output_event = asyncio.Event()

    async def receiver():
        nonlocal pending_frame_b64, pending_recv_ts, dropped_frame_count
        try:
            while not stop_event.is_set():
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "invalid_json"})
                    continue

                msg_type = data.get("type")

                if msg_type == "set_prompt":
                    try:
                        proc.set_prompt(str(data.get("prompt", "")))
                        await websocket.send_json({"type": "ack", "name": "set_prompt"})
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": f"set_prompt_failed:{e}"})

                elif msg_type == "set_param":
                    try:
                        proc.set_param(str(data["name"]), data["value"])
                        await websocket.send_json({"type": "ack", "name": "set_param"})
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": f"set_param_failed:{e}"})

                elif msg_type == "set_reference_image":
                    try:
                        ref = b64_to_reference_rgb(data["image_b64"])
                        proc.set_reference_image(ref)
                        await websocket.send_json({"type": "ack", "name": "set_reference_image"})
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": f"set_reference_image_failed:{e}"})

                elif msg_type == "frame":
                    # Latest-frame-wins: overwrite any pending frame.
                    frame_b64 = data.get("frame_b64")
                    if isinstance(frame_b64, str) and frame_b64:
                        async with pending_lock:
                            if pending_frame_b64 is not None:
                                dropped_frame_count += 1
                            pending_frame_b64 = frame_b64
                            pending_recv_ts = time.time()

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong", "state": public_state()})

                else:
                    await websocket.send_json({"type": "error", "message": f"unknown_type:{msg_type}"})
        except WebSocketDisconnect:
            log.info("livecam receiver: websocket disconnected")
        except Exception as e:
            log.exception("livecam receiver failure: %s", e)
        finally:
            stop_event.set()

    async def processor_loop():
        nonlocal pending_frame_b64, pending_recv_ts, latest_output_b64, latest_output_seq, latest_output_ts
        processed = 0
        try:
            while not stop_event.is_set():
                async with pending_lock:
                    frame_b64 = pending_frame_b64
                    recv_ts = pending_recv_ts
                    pending_frame_b64 = None

                if frame_b64 is None:
                    # No work; brief idle to avoid busy loop.
                    await asyncio.sleep(0.01)
                    continue

                t_proc_start = time.time()
                try:
                    frame = b64_to_bgr(frame_b64)
                    resized = crop_fn(frame, h, w)
                    input_t.copy_from(resized)
                    output = output_t.to_numpy()
                    out_b64 = bgr_to_b64_jpeg(output)
                except Exception as e:
                    try:
                        await websocket.send_json({"type": "error", "message": f"frame_failed:{e}"})
                    except Exception:
                        pass
                    continue
                t_proc_end = time.time()

                # Publish to sender hand-off.
                latest_output_b64 = out_b64
                latest_output_seq += 1
                latest_output_ts = t_proc_end
                output_event.set()

                processed += 1

                if ENABLE_LIVECAM_TIMING_LOG and processed % LIVECAM_LOG_EVERY == 0:
                    proc_ms = (t_proc_end - t_proc_start) * 1000.0
                    pending_age_ms = (t_proc_start - recv_ts) * 1000.0
                    log.info(
                        "livecam timing: proc_ms=%.1f pending_age_ms=%.1f "
                        "dropped=%d processed=%d latest_seq=%d",
                        proc_ms,
                        pending_age_ms,
                        dropped_frame_count,
                        processed,
                        latest_output_seq,
                    )
        except WebSocketDisconnect:
            log.info("livecam processor: websocket disconnected")
        except Exception as e:
            log.exception("livecam processor failure: %s", e)
        finally:
            stop_event.set()
            output_event.set()

    async def sender_loop():
        nonlocal latest_output_b64, latest_output_seq, last_sent_seq

        try:
            unique_frames_sent = 0
            repeated_frames_sent = 0
            last_send_ts = 0.0

            if not ENABLE_PLAYBACK_SCHEDULER:
                # Scheduler OFF: send immediately on new output_event.
                while not stop_event.is_set():
                    try:
                        await asyncio.wait_for(output_event.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    output_event.clear()

                    if latest_output_b64 is not None and latest_output_seq > last_sent_seq:
                        try:
                            await websocket.send_json({"type": "frame", "frame_b64": latest_output_b64})
                            last_send_ts = time.time()
                            last_sent_seq = latest_output_seq
                            unique_frames_sent += 1
                        except Exception as e:
                            log.warning("livecam sender: send failed: %s", e)
                            break

                        if ENABLE_LIVECAM_TIMING_LOG and (unique_frames_sent + repeated_frames_sent) % LIVECAM_LOG_EVERY == 0:
                            sender_interval_ms = (time.time() - last_send_ts) * 1000.0 if last_send_ts > 0 else 0.0
                            log.info(
                                "livecam sender: scheduler_enabled=False sender_interval_ms=%.1f "
                                "unique_frames_sent=%d repeated_frames_sent=%d latest_seq=%d last_sent_seq=%d",
                                sender_interval_ms,
                                unique_frames_sent,
                                repeated_frames_sent,
                                latest_output_seq,
                                last_sent_seq,
                            )
            else:
                # Scheduler ON: cadence-based with jitter handling.
                period_s = 1.0 / PLAYBACK_TARGET_FPS if PLAYBACK_TARGET_FPS > 0 else 0.0
                max_jitter_s = PLAYBACK_MAX_JITTER_MS / 1000.0
                next_tick_ts = time.time()

                while not stop_event.is_set():
                    cadence_deadline = next_tick_ts

                    # Check if new frame available.
                    have_new_frame = latest_output_b64 is not None and latest_output_seq > last_sent_seq

                    if have_new_frame:
                        freshness_deadline = latest_output_ts + max_jitter_s
                        emit_deadline = min(cadence_deadline, freshness_deadline)
                    else:
                        emit_deadline = cadence_deadline

                    # Wait until emit_deadline or output_event.
                    wait_time = max(0, emit_deadline - time.time())
                    try:
                        await asyncio.wait_for(output_event.wait(), timeout=wait_time)
                        # Early wake: clear event and recompute, do NOT send immediately.
                        output_event.clear()
                        continue
                    except asyncio.TimeoutError:
                        pass

                    if stop_event.is_set():
                        break

                    # emit_deadline reached.
                    if have_new_frame:
                        try:
                            await websocket.send_json({"type": "frame", "frame_b64": latest_output_b64})
                            last_send_ts = time.time()
                            last_sent_seq = latest_output_seq
                            unique_frames_sent += 1
                        except Exception as e:
                            log.warning("livecam sender: send failed: %s", e)
                            break
                    elif REPEAT_LAST_FRAME_WHEN_IDLE and latest_output_b64 is not None:
                        try:
                            await websocket.send_json({"type": "frame", "frame_b64": latest_output_b64})
                            last_send_ts = time.time()
                            repeated_frames_sent += 1
                        except Exception as e:
                            log.warning("livecam sender: send failed: %s", e)
                            break

                    # Advance cadence.
                    next_tick_ts += period_s

                    # Log periodically.
                    total_sent = unique_frames_sent + repeated_frames_sent
                    if ENABLE_LIVECAM_TIMING_LOG and total_sent > 0 and total_sent % LIVECAM_LOG_EVERY == 0:
                        sender_interval_ms = (time.time() - last_send_ts) * 1000.0 if last_send_ts > 0 else 0.0
                        log.info(
                            "livecam sender: scheduler_enabled=True sender_interval_ms=%.1f "
                            "unique_frames_sent=%d repeated_frames_sent=%d latest_seq=%d last_sent_seq=%d",
                            sender_interval_ms,
                            unique_frames_sent,
                            repeated_frames_sent,
                            latest_output_seq,
                            last_sent_seq,
                        )
        except WebSocketDisconnect:
            log.info("livecam sender: websocket disconnected")
        except Exception as e:
            log.exception("livecam sender failure: %s", e)
        finally:
            stop_event.set()

    async def session_cap():
        # D4: end the session at the deadline (5 minutes by default).
        remaining = deadline - time.time()
        if remaining > 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=remaining)
                return  # session ended on its own
            except asyncio.TimeoutError:
                pass
        log.info("livecam: session time limit reached")
        stop_event.set()
        try:
            await websocket.send_json({"type": "status", "status": "session_limit"})
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
        except Exception:
            pass

    tasks = [asyncio.ensure_future(c) for c in (receiver(), processor_loop(), sender_loop(), session_cap())]
    try:
        await asyncio.gather(*tasks)
    except WebSocketDisconnect:
        log.info("websocket disconnected")
    except Exception as e:
        log.exception("websocket failure: %s", e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # gather() does not stop the other loops when one fails (e.g. the receiver
        # after the session cap closes the socket); stop and wait for them so the
        # handler never returns with tasks still running.
        stop_event.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))

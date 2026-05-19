"""
FluxRT WebSocket server for RunPod Serverless Load Balancing.

Endpoints:
  GET  /ping  -> health (200 healthy, 204 initializing, other unhealthy)
  WS   /ws    -> live cam pipeline (token query param required)

Client protocol (JSON over WS):
  Client -> Server:
    {"type":"set_prompt", "prompt":"..."}
    {"type":"set_reference_image", "image_b64":"..."}   # PNG/JPEG bytes, base64
    {"type":"frame", "frame_b64":"..."}                  # JPEG bytes, base64
    {"type":"set_param", "name":"...", "value": ...}
  Server -> Client:
    {"type":"warming"}                                   # workers initializing
    {"type":"ready"}                                     # warmup complete
    {"type":"frame", "frame_b64":"..."}                  # JPEG output
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

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fluxrt-ws")

CONFIG_PATH = os.environ.get("FLUXRT_CONFIG", "/workspace/FluxRT/configs/stream_processor_config.json")
SESSION_SIGNING_SECRET = os.environ.get("SESSION_SIGNING_SECRET", "")
PORT = int(os.environ.get("PORT", "8765"))

# Global processor - loaded once per worker
processor = None
ready_event = asyncio.Event()


def verify_token(token):
    """Verify HMAC-SHA256 signed token minted by the Cloudflare Worker.
    Token format: userId.garmentId.expEpoch.hexSig
    """
    if not token or not SESSION_SIGNING_SECRET:
        return False
    try:
        last_dot = token.rfind(".")
        payload, sig = token[:last_dot], token[last_dot + 1:]
        parts = payload.split(".")
        if len(parts) != 3:
            return False
        exp = int(parts[2])
        if exp < int(time.time()):
            return False
        expected = hmac.new(
            SESSION_SIGNING_SECRET.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, sig)
    except Exception:
        return False


def b64_to_bgr(b64):
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1].copy()


def bgr_to_b64_jpeg(arr, quality=80):
    rgb = arr[:, :, ::-1]
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@asynccontextmanager
async def lifespan(app):
    """Non-blocking lifespan: start FastAPI immediately, warm up FluxRT in background.
    /ping returns 204 while loading, 200 once ready. This lets RunPod's load balancer
    keep the worker alive during the (potentially long) model load.
    """
    global processor

    async def warmup():
        global processor
        loop = asyncio.get_event_loop()
        try:
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
            log.exception("Warmup failed: %s", e)

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
    if ready_event.is_set():
        return Response(status_code=200)
    return Response(status_code=204)


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

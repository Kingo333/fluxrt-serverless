"""Azyah Premium HD adapter: FastAPI on :8765 in front of unmodified JoyAI on 127.0.0.1:8080.

GET  /ping            204 while loading, 200 when ready, 503 on error (RunPod LB health)
GET  /health          state as JSON (always 200)
POST /warmup          start loading (auth)            ?wait=true blocks until ready/error
POST /api/load        alias of /warmup (auth)
GET  /debug/startup   non-secret diagnostics (auth)
WS   /ws?token=...    FluxRT-compatible message contract (Premium tokens only)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Header, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, Response

from .bridge import Bridge
from .config import Settings
from session_token import SessionClock, SessionToken, parse_and_verify
from .upstream import Upstream

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("joyai-adapter")

def create_app(settings: Optional[Settings] = None) -> FastAPI:
    s = settings or Settings()
    # Fail closed: refuse to start without the signing secret.
    if not s.session_signing_secret:
        raise SystemExit("SESSION_SIGNING_SECRET is not set; refusing to start")

    upstream = Upstream(s)
    clock = SessionClock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if s.auto_load:
            await upstream.ensure_started()
        yield
        await upstream.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.upstream = upstream
    app.state.settings = s

    def verify(token: str) -> Optional[SessionToken]:
        tok = parse_and_verify(token, s.session_signing_secret)
        return tok if tok is not None and tok.tier in s.accepted_tiers else None

    def authorized(header_token: Optional[str], query_token: Optional[str]) -> bool:
        # Authorization is taken by RunPod's gateway, so the token comes in a custom header or ?token=.
        return verify(header_token or query_token or "") is not None

    def unauthorized() -> JSONResponse:
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

    @app.get("/ping")
    async def ping() -> Response:
        if upstream.status == "ready":
            return JSONResponse({"status": "ready"}, status_code=200)
        if upstream.status == "error":
            return JSONResponse({"status": "error"}, status_code=503)
        return Response(status_code=204)  # initializing

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": upstream.status != "error", **upstream.public_state()})

    async def _warmup(wait: bool) -> JSONResponse:
        await upstream.ensure_started()
        if wait:
            await asyncio.wait_for(upstream.wait_ready(), timeout=s.load_timeout_s + s.upstream_boot_timeout_s)
        return JSONResponse(upstream.public_state())

    @app.post("/warmup")
    async def warmup(
        wait: bool = Query(False),
        token: Optional[str] = Query(default=None),
        x_session_token: Optional[str] = Header(default=None),
    ):
        if not authorized(x_session_token, token):
            return unauthorized()
        return await _warmup(wait)

    @app.post("/api/load")
    async def api_load(
        wait: bool = Query(False),
        token: Optional[str] = Query(default=None),
        x_session_token: Optional[str] = Header(default=None),
    ):
        if not authorized(x_session_token, token):
            return unauthorized()
        return await _warmup(wait)

    @app.get("/debug/startup")
    async def debug_startup(
        token: Optional[str] = Query(default=None),
        x_session_token: Optional[str] = Header(default=None),
    ):
        if not authorized(x_session_token, token):
            return unauthorized()
        total, used, free = shutil.disk_usage("/")
        vol = "/runpod-volume"
        info: dict[str, Any] = {
            "ok": True,
            **upstream.public_state(),
            "upstream_health": await upstream.upstream_health(),
            "env_names": sorted(os.environ.keys()),
            "session_secret_configured": True,
            "disk": {"total": total, "used": used, "free": free},
            "runpod_volume_exists": os.path.isdir(vol),
            "checkpoints": {
                name: os.path.exists(os.path.join(os.getenv("JOYOMNI_CKPT_ROOT", f"{vol}/joyai/checkpoints"), name))
                for name in (
                    "JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth",
                    "JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors",
                    "MiMo-VL-7B-RL-2508",
                    "face_detection_yunet_2023mar.onnx",
                    "yolov8n.onnx",
                )
            },
        }
        return JSONResponse(info)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket, token: str = Query(default="")) -> None:
        tok = verify(token)
        if tok is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await websocket.accept()

        send_lock = asyncio.Lock()

        async def send_client(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_text(json.dumps(payload))

        # Wait for the model; keep the socket open like FluxRT does.
        if upstream.status != "ready":
            await send_client({"type": "warming", "state": upstream.public_state()})
            await upstream.ensure_started()
            while upstream.status not in ("ready", "error"):
                await asyncio.sleep(1.0)
            if upstream.status == "error":
                await send_client({"type": "error", "message": upstream.error or "upstream_error"})
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                return

        deadline = clock.deadline(tok, s.max_session_seconds)
        if time.time() >= deadline:
            await send_client({"type": "status", "status": "session_limit"})
            await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
            return
        await send_client({"type": "ready", "session_deadline": int(deadline)})

        bridge = Bridge(s, send_client)
        try:
            while True:
                if bridge.failed:
                    await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                    return
                remaining = deadline - time.time()
                if remaining <= 0:
                    await send_client({"type": "status", "status": "session_limit"})
                    await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                    return
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=min(1.0, remaining))
                except asyncio.TimeoutError:
                    continue
                try:
                    data = json.loads(raw)
                except ValueError:
                    await send_client({"type": "error", "message": "invalid_json"})
                    continue
                await bridge.handle_client_message(data)
        except WebSocketDisconnect:
            log.info("client disconnected")
        except Exception as e:  # noqa: BLE001
            log.exception("ws failure")
            try:
                await send_client({"type": "error", "message": f"{type(e).__name__}"})
            except Exception:  # noqa: BLE001
                pass
        finally:
            await bridge.close()

    return app


if __name__ == "__main__":
    import uvicorn

    settings = Settings()
    uvicorn.run(create_app(settings), host="0.0.0.0", port=settings.port, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))

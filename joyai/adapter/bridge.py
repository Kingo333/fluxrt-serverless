"""Translates the app's FluxRT message contract to JoyAI's WebSocket protocol.

Client (app)                       JoyAI upstream (127.0.0.1:8080/ws)
--------------------------------   -----------------------------------------------
set_reference_image {image_b64} -> stored; sent as start.ref_image
set_prompt {prompt}             -> start {prompt, ref_image, 480x840, mjpeg, ...}
                                   (sent again if prompt/reference/params change)
frame {frame_b64}               -> binary JPEG (dropped until upstream "started")
set_param {name, value}         -> stored; applied with a fresh start
ping                            -> answered locally: pong
                                <- output_frame {...} + binary JPEG -> frame {frame_b64}
                                <- waiting_face / no_person / session_reset -> status
                                <- queue_position -> status "queued"
                                <- session_timeout / unexpected close -> reconnect + start
                                <- error -> error
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import websockets

from .config import Settings

log = logging.getLogger("joyai-adapter.bridge")

ClientSend = Callable[[dict[str, Any]], Awaitable[None]]

# set_param names the app may send -> JoyAI start fields.
PARAM_MAP = {
    "steps": "num_inference_steps",
    "num_inference_steps": "num_inference_steps",
    "seed": "seed",
    "kv_reset_frames": "kv_reset_frames",
    "output_quality": "output_quality",
}

STATUS_PASSTHROUGH = {"waiting_face", "no_person", "session_reset"}
MAX_RECONNECTS = 3


class Bridge:
    def __init__(self, settings: Settings, send_client: ClientSend) -> None:
        self.s = settings
        self.send_client = send_client
        self.ref_b64: Optional[str] = None
        self.prompt: Optional[str] = None
        # Short edit instruction from the app (Lucy-style); preferred over the
        # long FluxRT-oriented prompt when present.
        self.instruction: Optional[str] = None
        self._restart_task: Optional[asyncio.Task] = None
        self.params: dict[str, Any] = {}
        self.upstream: Optional[Any] = None  # websockets connection
        self.upstream_started = False
        self.reader_task: Optional[asyncio.Task] = None
        self.reconnects = 0
        self.frames_in = 0
        self.frames_out = 0
        self._closing = False
        # Set when the upstream can't be recovered; the WS handler then closes.
        self.failed = False
        self._send_lock = asyncio.Lock()

    # ---- start payload -------------------------------------------------
    def start_payload(self) -> dict[str, Any]:
        text = self.instruction or self.prompt or ""
        prompt = self.s.prompt_template.replace("{prompt}", text)
        payload: dict[str, Any] = {
            "type": "start",
            "prompt": prompt,
            "width": self.s.width,
            "height": self.s.height,
            "fps": self.s.fps,
            "num_inference_steps": self.s.num_inference_steps,
            "kv_reset_frames": self.s.kv_reset_frames,
            "output_quality": self.s.output_quality,
            "input_codec": "mjpeg",
            "output_codec": "mjpeg",
            "use_pe": False,  # D3: prompt enhancement off
            "gate_enabled": self.s.gate_enabled,
            "max_temporal_ids": self.s.max_temporal_ids,
            "source": "camera",
        }
        if self.ref_b64:
            payload["ref_image"] = self.ref_b64
        for name, value in self.params.items():
            payload[PARAM_MAP[name]] = value
        return payload

    # ---- client -> upstream ----------------------------------------------
    async def handle_client_message(self, data: dict[str, Any]) -> None:
        t = data.get("type")
        if t == "set_reference_image":
            b64 = _strip_data_url(str(data.get("image_b64") or ""))
            try:
                base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError):
                await self.send_client({"type": "error", "message": "set_reference_image_failed:invalid_base64"})
                return
            self.ref_b64 = b64
            await self.send_client({"type": "ack", "name": "set_reference_image"})
            if self.upstream is not None:
                self._schedule_restart()  # garment swap: one restart for image + prompt
        elif t == "set_prompt":
            self.prompt = str(data.get("prompt") or "")
            instruction = data.get("instruction")
            self.instruction = str(instruction).strip() if isinstance(instruction, str) and instruction.strip() else None
            await self.send_client({"type": "ack", "name": "set_prompt"})
            if self.upstream is None:
                await self._ensure_upstream_and_start()
            else:
                self._schedule_restart()
        elif t == "set_param":
            name = str(data.get("name") or "")
            if name not in PARAM_MAP:
                await self.send_client({"type": "error", "message": f"set_param_failed:unknown_param:{name}"})
                return
            try:
                self.params[name] = int(data.get("value"))
            except (TypeError, ValueError):
                await self.send_client({"type": "error", "message": f"set_param_failed:invalid_value:{name}"})
                return
            await self.send_client({"type": "ack", "name": "set_param"})
            if self.upstream is not None:
                if name == "output_quality":
                    await self._upstream_send_json({"type": "set_output_quality", "value": self.params[name]})
                else:
                    self._schedule_restart()
        elif t == "frame":
            await self._forward_frame(str(data.get("frame_b64") or ""))
        elif t == "ping":
            await self.send_client({"type": "pong", "frames_in": self.frames_in, "frames_out": self.frames_out})
        else:
            await self.send_client({"type": "error", "message": f"unknown_type:{t}"})

    async def _forward_frame(self, frame_b64: str) -> None:
        if self.upstream is None or not self.upstream_started:
            return  # not editing yet; drop
        try:
            jpeg = base64.b64decode(_strip_data_url(frame_b64))
        except (binascii.Error, ValueError):
            return
        self.frames_in += 1
        try:
            async with self._send_lock:
                await self.upstream.send(json.dumps({"type": "frame_meta", "seq": self.frames_in,
                                                     "t_capture_ms": time.time() * 1000.0}))
                await self.upstream.send(jpeg)
        except websockets.ConnectionClosed:
            pass  # reader task handles reconnect

    # ---- upstream connection --------------------------------------------
    def _schedule_restart(self, delay_s: float = 0.25) -> None:
        """Coalesce back-to-back changes (reference image + prompt on a garment
        swap) into a single JoyAI restart on the same upstream socket."""
        if self._restart_task is not None and not self._restart_task.done():
            self._restart_task.cancel()

        async def _later() -> None:
            await asyncio.sleep(delay_s)
            await self._send_start()

        self._restart_task = asyncio.create_task(_later(), name="joyai-restart")

    async def _ensure_upstream_and_start(self) -> None:
        if self.upstream is None:
            await self._connect()
        await self._send_start()

    async def _connect(self) -> None:
        self.upstream = await websockets.connect(self.s.upstream_ws, max_size=None, ping_interval=20)
        self.upstream_started = False
        self.reader_task = asyncio.create_task(self._read_upstream(), name="joyai-upstream-reader")

    async def _send_start(self) -> None:
        self.upstream_started = False
        await self._upstream_send_json(self.start_payload())

    async def _upstream_send_json(self, payload: dict[str, Any]) -> None:
        if self.upstream is None:
            return
        try:
            async with self._send_lock:
                await self.upstream.send(json.dumps(payload))
        except websockets.ConnectionClosed:
            pass

    async def _read_upstream(self) -> None:
        conn = self.upstream
        pending_header: Optional[dict[str, Any]] = None
        try:
            async for message in conn:
                if isinstance(message, (bytes, bytearray)):
                    pending_header = None
                    self.frames_out += 1
                    await self.send_client({"type": "frame", "frame_b64": base64.b64encode(message).decode("ascii")})
                    continue
                try:
                    data = json.loads(message)
                except ValueError:
                    continue
                t = data.get("type")
                if t == "output_frame":
                    pending_header = data
                elif t == "started":
                    self.upstream_started = True
                elif t in STATUS_PASSTHROUGH:
                    status = {"type": "status", "status": t}
                    if data.get("reason") is not None:
                        status["reason"] = data["reason"]
                    await self.send_client(status)
                elif t == "queue_position":
                    await self.send_client({"type": "status", "status": "queued", "position": data.get("position")})
                elif t == "session_timeout":
                    log.info("upstream released the session (idle); reconnecting")
                    break
                elif t == "error":
                    await self.send_client({"type": "error", "message": str(data.get("message") or "upstream_error")})
                # chunk_start / chunk_done / accepted / pong / flow_drop / session_granted: ignored
        except websockets.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            raise
        finally:
            del pending_header
        if not self._closing:
            await self._reconnect()

    async def _reconnect(self) -> None:
        self.upstream = None
        self.upstream_started = False
        if self.reconnects >= MAX_RECONNECTS:
            await self.send_client({"type": "error", "message": "upstream_unavailable"})
            self.failed = True
            return
        self.reconnects += 1
        await self.send_client({"type": "status", "status": "session_reset"})
        await asyncio.sleep(0.5)
        try:
            await self._connect()
            await self._send_start()
        except OSError as e:
            await self.send_client({"type": "error", "message": f"upstream_unavailable:{type(e).__name__}"})
            self.failed = True

    async def close(self) -> None:
        self._closing = True
        if self._restart_task is not None and not self._restart_task.done():
            self._restart_task.cancel()
        if self.reader_task is not None and not self.reader_task.done():
            self.reader_task.cancel()
        if self.upstream is not None:
            try:
                await self.upstream.close()  # releases JoyAI's single session slot
            except Exception:  # noqa: BLE001
                pass
        self.upstream = None


def _strip_data_url(value: str) -> str:
    return value.split(",", 1)[1] if value.startswith("data:") and "," in value else value

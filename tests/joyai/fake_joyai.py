"""Minimal stand-in for JoyAI's server (deploy/xvideo/serving/serve_joyomni_streaming.py).

Mirrors the parts of the protocol the adapter relies on:
GET /health, POST /load, WS /ws with session_granted, start->started,
binary JPEG in -> output_frame header + binary out, waiting_face, session_timeout.
"""

import asyncio
import json
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse


class FakeJoyAI:
    def __init__(self, load_ok=True, load_delay=0.2):
        self.loaded = False
        self.load_ok = load_ok
        self.load_delay = load_delay
        self.starts = []  # start payloads received
        self.connections = 0
        self.timeout_after_frames = None  # send session_timeout after N frames (first connection only)
        self.emit_waiting_face = False
        app = FastAPI()

        @app.get("/health")
        def health():
            return JSONResponse({"ok": True, "error": None, "runtime_loaded": self.loaded})

        @app.post("/load")
        def load():
            time.sleep(self.load_delay)
            if not self.load_ok:
                return JSONResponse({"ok": False}, status_code=500)
            self.loaded = True
            return JSONResponse({"ok": True})

        @app.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            self.connections += 1
            conn_no = self.connections
            await websocket.send_json({"type": "session_granted"})
            frames = 0
            last_seq = None
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        return
                    if msg.get("text"):
                        data = json.loads(msg["text"])
                        if data["type"] == "frame_meta":
                            last_seq = data.get("seq")
                        if data["type"] == "start":
                            self.starts.append(data)
                            await websocket.send_json({"type": "started", "width": data["width"], "height": data["height"]})
                            if self.emit_waiting_face:
                                await websocket.send_json({"type": "waiting_face", "reason": "show_full_face"})
                    elif msg.get("bytes") is not None:
                        frames += 1
                        await websocket.send_json({"type": "output_frame", "index": 0, "count": 1,
                                                   "source_seq": last_seq})
                        await websocket.send_bytes(b"OUT:" + msg["bytes"])
                        if conn_no == 1 and self.timeout_after_frames and frames >= self.timeout_after_frames:
                            await websocket.send_json({"type": "session_timeout", "message": "idle"})
                            await websocket.close()
                            return
            except WebSocketDisconnect:
                return

        self.app = app
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)

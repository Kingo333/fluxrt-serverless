"""Runs unmodified JoyAI on loopback and tracks whether its model is loaded.

States: idle -> booting -> loading -> ready, or error at any point.
  idle     upstream not started yet (AUTO_LOAD=false and nobody called /warmup)
  booting  JoyAI process started, waiting for its /health to answer
  loading  POST /load in flight (JoyAI started with --no-preload)
  ready    model loaded; sessions can start
  error    process exited, /health never came up, or /load failed
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from typing import Any, Optional

import httpx

from .config import Settings

log = logging.getLogger("joyai-adapter.upstream")


class Upstream:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.status = "idle"
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.ready_at: Optional[float] = None
        self.process: Optional[asyncio.subprocess.Process] = None
        self._task: Optional[asyncio.Task] = None
        self._watch: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def public_state(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "upstream_pid": self.process.pid if self.process else None,
        }

    async def ensure_started(self) -> None:
        """Start boot+load once. Safe to call repeatedly."""
        async with self._lock:
            if self._task is not None or self.status in ("ready", "error"):
                return
            self.started_at = time.time()
            self._task = asyncio.create_task(self._run(), name="joyai-upstream")

    async def wait_ready(self, poll_s: float = 1.0) -> str:
        while self.status not in ("ready", "error"):
            await asyncio.sleep(poll_s)
        return self.status

    async def _run(self) -> None:
        try:
            if self.s.upstream_command:
                await self._spawn()
            self.status = "booting"
            await self._wait_for_health()
            self.status = "loading"
            await self._load()
            self.status = "ready"
            self.ready_at = time.time()
            log.info("JoyAI ready after %.0fs", self.ready_at - (self.started_at or self.ready_at))
        except Exception as e:  # noqa: BLE001
            self._fail(f"{type(e).__name__}: {e}")

    def _fail(self, message: str) -> None:
        if self.status != "error":
            log.error("upstream error: %s", message)
        self.status = "error"
        self.error = message

    async def _spawn(self) -> None:
        argv = shlex.split(self.s.upstream_command)
        log.info("starting JoyAI: %s", argv)
        self.process = await asyncio.create_subprocess_exec(*argv)
        self._watch = asyncio.create_task(self._watch_process(), name="joyai-upstream-watch")

    async def _watch_process(self) -> None:
        assert self.process is not None
        code = await self.process.wait()
        self._fail(f"JoyAI process exited with code {code}")

    async def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self.s.upstream_boot_timeout_s
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self.status == "error":
                    raise RuntimeError(self.error or "upstream failed")
                try:
                    r = await client.get(f"{self.s.upstream_http}/health")
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2.0)
        raise TimeoutError(f"JoyAI /health not reachable after {self.s.upstream_boot_timeout_s}s")

    async def _load(self) -> None:
        # JoyAI's POST /load blocks until the model is loaded (several minutes cold).
        async with httpx.AsyncClient(timeout=self.s.load_timeout_s) as client:
            r = await client.post(f"{self.s.upstream_http}/load")
            if r.status_code != 200:
                raise RuntimeError(f"/load returned {r.status_code}: {r.text[:300]}")
            health = (await client.get(f"{self.s.upstream_http}/health")).json()
            if not health.get("runtime_loaded") or health.get("error"):
                raise RuntimeError(f"JoyAI not loaded after /load: {health.get('error')}")

    async def upstream_health(self) -> Optional[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.s.upstream_http}/health")
                return r.json() if r.status_code == 200 else {"status_code": r.status_code}
        except Exception as e:  # noqa: BLE001
            return {"unreachable": f"{type(e).__name__}"}

    async def shutdown(self) -> None:
        for t in (self._task, self._watch):
            if t is not None and not t.done():
                t.cancel()
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()

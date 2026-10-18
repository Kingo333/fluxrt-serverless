"""
bootstrap.py -- failsafe launcher for the FluxRT load-balancer worker.

Goal: keep the HTTP server on ${PORT:-8765} alive no matter what happens
inside server.py. If importing or running server.app raises, this process
catches it and starts a minimal FastAPI fallback on the same port so that
RunPod's health probe (/ping) and /debug/startup can surface the real
exception instead of the worker silently dying.
"""

import os
import sys
import time
import traceback
from pathlib import Path

# --- 1. early diagnostics -----------------------------------------------------

PORT = int(os.getenv("PORT", "8765"))
APP_DIR = "/app"
SERVER_PATH = Path(APP_DIR) / "server.py"
FLUXRT_ROOT = Path(os.getenv("FLUXRT_ROOT", "/workspace/FluxRT"))

print(f"[bootstrap] python {sys.version}", flush=True)
print(f"[bootstrap] cwd={os.getcwd()}", flush=True)
print(f"[bootstrap] PORT={PORT}", flush=True)
print(f"[bootstrap] APP_DIR={APP_DIR} exists={Path(APP_DIR).exists()}", flush=True)
print(f"[bootstrap] SERVER_PATH={SERVER_PATH} exists={SERVER_PATH.exists()}", flush=True)
print(f"[bootstrap] FLUXRT_ROOT={FLUXRT_ROOT} exists={FLUXRT_ROOT.exists()}", flush=True)

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
    print(f"[bootstrap] inserted {APP_DIR} into sys.path", flush=True)


def _redacted_env_names():
    """Return only the names of env vars, never their values."""
    return sorted(os.environ.keys())


# --- 2. try the real server ---------------------------------------------------

def _run_real_server():
    print("[bootstrap] importing server...", flush=True)
    import server  # noqa: F401  -- side-effect: defines server.app
    print("[bootstrap] server imported ok; starting uvicorn on 0.0.0.0:%d" % PORT, flush=True)
    import uvicorn
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=PORT,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )


# --- 3. fallback minimal app --------------------------------------------------

def _run_fallback(exc: BaseException):
    tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print("[bootstrap] !!! server.py failed to import/run; starting fallback", flush=True)
    print(tb_text, flush=True)

    from typing import Optional

    from fastapi import FastAPI, Header, Query
    from fastapi.responses import JSONResponse
    import uvicorn as _uv

    fb = FastAPI()

    def _authorized(header_token: Optional[str], query_token: Optional[str]) -> bool:
        # Diagnostics include tracebacks, so they need a valid session token.
        # Fails closed if the secret or the token module is missing.
        try:
            from session_token import parse_and_verify
        except Exception:
            return False
        secret = os.getenv("SESSION_SIGNING_SECRET", "")
        return parse_and_verify(header_token or query_token or "", secret) is not None

    def _diag():
        return {
            "ok": False,
            "mode": "fallback",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": tb_text,
            "cwd": os.getcwd(),
            "env_names": _redacted_env_names(),
            "python_version": sys.version,
            "server_py_exists": SERVER_PATH.exists(),
            "fluxrt_root_exists": FLUXRT_ROOT.exists(),
            "port": PORT,
            "bootstrap_started_at": _BOOT_TS,
        }

    @fb.get("/")
    async def root():
        return JSONResponse({"ok": False, "mode": "fallback"})

    @fb.get("/ping")
    async def ping():
        # 200 so RunPod's health probe at least sees us alive in fallback mode.
        return JSONResponse({"ok": True, "mode": "fallback"})

    @fb.get("/health")
    async def health():
        return JSONResponse({"ok": True, "mode": "fallback"})

    @fb.get("/debug/startup")
    async def debug_startup(
        token: Optional[str] = Query(default=None),
        x_session_token: Optional[str] = Header(default=None),
    ):
        if not _authorized(x_session_token, token):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
        return JSONResponse(_diag())

    _uv.run(
        fb,
        host="0.0.0.0",
        port=PORT,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )


# --- 4. main ------------------------------------------------------------------

_BOOT_TS = time.time()

if __name__ == "__main__":
    try:
        _run_real_server()
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 -- intentional: we want to catch everything
        try:
            _run_fallback(e)
        except BaseException as e2:  # noqa: BLE001
            # Last-resort log; container will exit and RunPod will restart us.
            print(
                "[bootstrap] FATAL: fallback itself failed: "
                f"{type(e2).__name__}: {e2}",
                flush=True,
            )
            traceback.print_exc()
            sys.exit(1)

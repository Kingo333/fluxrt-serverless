"""Adapter settings, all from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Public port RunPod routes to (adapter). JoyAI itself stays on loopback.
    port: int = field(default_factory=lambda: _int("PORT", 8765))
    upstream_host: str = field(default_factory=lambda: os.getenv("JOYAI_UPSTREAM_HOST", "127.0.0.1"))
    upstream_port: int = field(default_factory=lambda: _int("JOYAI_UPSTREAM_PORT", 8080))

    # Auth (same HMAC token as the Cloudflare Worker issues).
    session_signing_secret: str = field(default_factory=lambda: os.getenv("SESSION_SIGNING_SECRET", ""))
    max_session_seconds: int = field(default_factory=lambda: _int("MAX_SESSION_SECONDS", 300))
    # Token tiers accepted. Both by default: with LIVECAM_ENGINE one endpoint
    # may serve either tier; the Cloudflare Worker decides routing.
    accepted_tiers: frozenset = field(default_factory=lambda: frozenset(
        t.strip() for t in os.getenv("ACCEPTED_TIERS", "standard,premium").split(",") if t.strip()))

    # Upstream process. Empty command = don't spawn (tests / external upstream).
    upstream_command: str = field(
        default_factory=lambda: os.getenv("JOYAI_UPSTREAM_COMMAND", "/app/joyai/start_upstream.sh")
    )
    auto_load: bool = field(default_factory=lambda: _bool("AUTO_LOAD", True))
    upstream_boot_timeout_s: int = field(default_factory=lambda: _int("UPSTREAM_BOOT_TIMEOUT_S", 300))
    load_timeout_s: int = field(default_factory=lambda: _int("LOAD_TIMEOUT_S", 3600))
    # RunPod stops a load-balancing worker whose /ping still says 204 after ~8 minutes,
    # and a first JoyAI start (weights + kernel compile) takes longer. When true, /ping
    # answers 200 while loading; sessions that arrive early get "warming" and wait.
    ping_ready_while_loading: bool = field(default_factory=lambda: _bool("PING_READY_WHILE_LOADING", False))

    # Session `start` sent to JoyAI (480x840 portrait, mjpeg both ways, gates on).
    width: int = field(default_factory=lambda: _int("JOYAI_SESSION_WIDTH", 480))
    height: int = field(default_factory=lambda: _int("JOYAI_SESSION_HEIGHT", 840))
    fps: int = field(default_factory=lambda: _int("JOYOMNI_FPS", 24))
    num_inference_steps: int = field(default_factory=lambda: _int("JOYAI_STEPS", 2))
    kv_reset_frames: int = field(default_factory=lambda: _int("JOYAI_KV_RESET_FRAMES", 600))
    # Recent-chunk memory (JoyAI's max_temporal_ids). Smaller = faster, larger = steadier.
    max_temporal_ids: int = field(default_factory=lambda: _int("JOYAI_MAX_TEMPORAL_IDS", 8))
    # 90 keeps prints, logos and small text; 70 visibly blurred them.
    output_quality: int = field(default_factory=lambda: _int("JOYAI_OUTPUT_QUALITY", 90))
    # Camera JPEG quality the app should use (sent to the app in "ready").
    input_jpeg_quality: float = field(default_factory=lambda: _int("JOYAI_INPUT_JPEG_QUALITY", 85) / 100.0)
    # Per-chunk JoyAI stage timings passed to the app with each frame (benchmarking only).
    profile_timings: bool = field(default_factory=lambda: _bool("JOYAI_PROFILE_TIMINGS", False))
    # "full" (default) passes the app's instruction as is; "short" keeps only its first
    # sentence + the keep-the-person sentence, matching a model trained on short captions.
    instruction_style: str = field(default_factory=lambda: os.getenv("JOYAI_INSTRUCTION_STYLE", "full").strip().lower() or "full")
    # Send the app JoyAI's frame size/fps/quality in "ready" (#18). False reproduces the
    # pre-#18 behaviour (app keeps its tier settings) for before/after benchmarks (review R3).
    client_hints: bool = field(default_factory=lambda: _bool("JOYAI_CLIENT_HINTS", True))
    # Garment swaps restart the JoyAI session, which re-arms its face gate: by default
    # 24 still, centred frames (1 s at 24 fps, 2 s at 12) before editing resumes. The
    # person was already verified, so swaps ask for a shorter streak. 0 = upstream default.
    swap_stable_frames: int = field(default_factory=lambda: _int("JOYAI_SWAP_STABLE_FRAMES", 6))
    # Which trained model this worker serves (start_upstream.sh resolves it); reported to the app.
    model_version: str = field(default_factory=lambda: os.getenv("JOYAI_MODEL_VERSION", "joyai-baseline") or "joyai-baseline")
    gate_enabled: bool = field(default_factory=lambda: _bool("JOYAI_GATE_ENABLED", True))
    # Optional wrapper around the app's prompt, e.g.
    # "Replace the subject's outfit with the garment from the reference image. {prompt}"
    prompt_template: str = field(default_factory=lambda: os.getenv("JOYAI_PROMPT_TEMPLATE", "{prompt}"))

    @property
    def upstream_http(self) -> str:
        return f"http://{self.upstream_host}:{self.upstream_port}"

    @property
    def upstream_ws(self) -> str:
        return f"ws://{self.upstream_host}:{self.upstream_port}/ws"

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
    load_timeout_s: int = field(default_factory=lambda: _int("LOAD_TIMEOUT_S", 2400))

    # Session `start` sent to JoyAI (480x840 portrait, mjpeg both ways, gates on).
    width: int = field(default_factory=lambda: _int("JOYAI_SESSION_WIDTH", 480))
    height: int = field(default_factory=lambda: _int("JOYAI_SESSION_HEIGHT", 840))
    fps: int = field(default_factory=lambda: _int("JOYOMNI_FPS", 24))
    num_inference_steps: int = field(default_factory=lambda: _int("JOYAI_STEPS", 2))
    kv_reset_frames: int = field(default_factory=lambda: _int("JOYAI_KV_RESET_FRAMES", 600))
    output_quality: int = field(default_factory=lambda: _int("JOYAI_OUTPUT_QUALITY", 70))
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

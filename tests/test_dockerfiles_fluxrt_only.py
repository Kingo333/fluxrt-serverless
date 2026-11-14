"""The FluxRT-only image (Dockerfile.livecam-fluxrt) is the dual image (Dockerfile.livecam) without JoyAI.

The shared steps must stay word for word identical, so the FluxRT engine is the same in both images and JoyAI
can still be built from Dockerfile.livecam.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUAL = (ROOT / "Dockerfile.livecam").read_text()
ONLY = (ROOT / "Dockerfile.livecam-fluxrt").read_text()
BAR = "# ---------------------------------------------------------------------------\n"


def _section(text: str, title: str, next_title: str) -> str:
    start = text.index(BAR + title)
    return text[start:text.index(BAR + next_title, start)]


def test_fluxrt_steps_are_identical_to_the_dual_image():
    shared = _section(DUAL, "# FluxRT engine", "# JoyAI engine").rstrip("\n")
    assert shared in ONLY
    assert "COPY fluxrt/patches /tmp/fluxrt-patches" in shared and "-c /tmp/fluxrt-constraints.txt" in shared
    assert "--target /opt/azyah-torchao torchao==0.14.1" in shared


def test_base_system_and_start_are_identical():
    first = lambda t, prefix: next(l for l in t.splitlines() if l.startswith(prefix))
    for prefix in ("FROM ", "ARG FLUXRT_REF=", 'CMD ["/usr/bin/python3.12"', "EXPOSE "):
        assert first(ONLY, prefix) == first(DUAL, prefix), prefix
    # the ENV block and the system packages (apt + deadsnakes Python 3.12)
    env_block = DUAL[DUAL.index("ENV DEBIAN_FRONTEND"):DUAL.index(BAR + "# System packages")]
    apt = DUAL[DUAL.index("RUN apt-get update"):DUAL.index(BAR + "# FluxRT engine")]
    assert env_block in ONLY and apt in ONLY


def test_no_joyai_in_the_fluxrt_only_image():
    steps = "\n".join(l for l in ONLY.splitlines() if not l.lstrip().startswith("#"))
    for word in ("joyai", "JoyAI", "SageAttention", "cutlass", "joyomni_ops", "JOYAI_", "JOYOMNI_"):
        assert word not in steps, word
    assert "ENV LIVECAM_ENGINE=fluxrt" in ONLY
    assert "COPY server.py bootstrap.py session_token.py /app/" in ONLY and "COPY livecam /app/livecam" in ONLY
    assert "import server; print('fluxrt server import ok')" in ONLY and "assert webrtc.available()" in ONLY


def test_bootstrap_refuses_joyai_clearly_on_this_image():
    boot = (ROOT / "bootstrap.py").read_text()
    assert "LIVECAM_ENGINE=joyai but {py} is missing (use the livecam-dual image)" in boot


def test_workflow_builds_only_fluxrt_tags():
    wf = (ROOT / ".github/workflows/build-livecam-fluxrt.yml").read_text()
    assert "file: ./Dockerfile.livecam-fluxrt" in wf
    assert "livecam-fluxrt-v1-${{ github.sha }}" in wf and "fluxrt-serverless:livecam-dual" not in wf
    assert '"refs/heads/main"' in wf                                  # the moving tag only follows main
    for path in ("Dockerfile.livecam-fluxrt", "fluxrt/patches/**", "fluxrt/constraints.txt", "server.py", "livecam/**"):
        assert f"- {path}" in wf, path

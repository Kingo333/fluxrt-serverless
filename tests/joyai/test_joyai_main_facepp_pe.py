"""JoyAI track (2026-11): upstream main pin, face++ checkpoint, prompt enhancement, wording.

Every switch defaults to today's behaviour: 0811, PE off (keys unset), the app's wording,
the near face gate. These tests pin that, and the on-states, without a GPU."""
import base64
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from joyai.adapter.app import create_app
from joyai.adapter.bridge import reword_reference
from joyai.adapter.config import Settings
from tests.joyai.fake_joyai import FakeJoyAI
from tests.joyai.test_adapter import SECRET, handshake, settings, tok, wait_for
from tests.joyai.test_start_upstream import run

ROOT = Path(__file__).resolve().parents[2]
JOYAI_MAIN = Path("/tmp/claude-0/joyai-main")

# The app's exact instruction strings (azyah-shopping buildTryOnPrompt.ts INSTRUCTIONS).
APP_INSTRUCTIONS = {
    "tops": "Substitute the current top with the garment in the reference image.",
    "bottoms": "Substitute the current pants or skirt with the garment in the reference image.",
    "dresses": "Substitute the current outfit with the dress in the reference image.",
    "outerwear": "Put the outerwear from the reference image on the person, over their current outfit.",
    "shoes": "Substitute the current shoes with the shoes in the reference image.",
    "bags": "Have the person carry the bag from the reference image.",
    "accessories": "Put the accessory from the reference image on the person.",
    "fallback": "Substitute the matching clothing item with the garment in the reference image.",
}
KEEP = " Keep the person's face, body, pose and the background unchanged."


# ---------- start_upstream.sh ----------

def test_pe_off_by_default_and_keys_unset(tmp_path):
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31", OPENAI_API_KEY="sk-leak", OPENAI_BASE_URL="https://x", PE_MODEL="m")
    out = env["_stdout"]
    assert "--no-use-pe --pe-backend none" in out
    assert "sk-leak" not in out and "OPENAI_API_KEY" not in out
    assert "--full-body-detection" not in out


def test_pe_on_passes_api_backend_and_never_logs_the_key(tmp_path):
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31", JOYAI_USE_PE="1", OPENAI_API_KEY="sk-secret-123",
              OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai", PE_MODEL="gemini-x")
    out = env["_stdout"]
    assert "--use-pe --pe-backend api --pe-model gemini-x --pe-timeout-s 20" in out
    assert "OPENAI_API_KEY=sk-secret-123" in out          # reaches JoyAI's environment
    assert "sk-secret-123" not in out.replace("OPENAI_API_KEY=sk-secret-123", "")  # never printed by us
    assert "prompt enhancement ON" in out and "(key: set)" in out
    env2 = run(tmp_path, "NVIDIA GeForce RTX 5090|31", JOYAI_USE_PE="1", OPENAI_API_KEY="k",
               OPENAI_BASE_URL="https://x", PE_MODEL="m", JOYAI_PE_TIMEOUT_S="45", JOYOMNI_PE_BACKEND="api")
    assert "--pe-timeout-s 45" in env2["_stdout"]


def test_pe_requested_but_not_configured_starts_off(tmp_path):
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31", JOYAI_USE_PE="1", OPENAI_API_KEY="k")
    out = env["_stdout"]
    assert "--no-use-pe --pe-backend none" in out and "OPENAI_API_KEY" not in out


def test_full_body_gate_flag_and_upstream_main_env(tmp_path):
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31", JOYAI_FULL_BODY_GATE="1")
    assert "--full-body-detection" in env["_stdout"]
    assert env["JOYOMNI_FA4_ATTN"] == "0"                       # DEPLOYMENT.md: 0 on 5090 / PRO 6000
    assert env["JOYOMNI_LOAD_WARMUP_TIMEOUT_S"] == "900"        # replaces the dropped timeouts patch
    env2 = run(tmp_path, "NVIDIA RTX PRO 6000 Blackwell Server Edition|94", JOYOMNI_FA4_ATTN="1",
               JOYOMNI_LOAD_WARMUP_TIMEOUT_S="1200")
    assert env2["JOYOMNI_FA4_ATTN"] == "1" and env2["JOYOMNI_LOAD_WARMUP_TIMEOUT_S"] == "1200"


def test_checkpoint_default_0811_and_facepp_override(tmp_path):
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31")
    assert "joyai_video_edit_dit_0811.pth (1 bytes)" in env["_stdout"]
    facepp = tmp_path / "vol" / "checkpoints" / "JoyAI-Video-Edit" / "dit" / "joyai_video_edit_dit_1101_face++.pth"
    facepp.write_text("face++")
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31", JOYOMNI_DIT_CKPT=str(facepp))
    assert "joyai_video_edit_dit_1101_face++.pth (6 bytes)" in env["_stdout"]


def test_timeouts_patch_dropped_and_remaining_patches_apply_to_main():
    patches = sorted((ROOT / "joyai" / "patches").glob("*.patch"))
    assert [p.name for p in patches] == ["joyai-warmup-orientation.patch"]
    df = (ROOT / "Dockerfile.livecam").read_text()
    assert "ARG JOYAI_COMMIT=45ff66d97095640b62861198602e3d5fb64226e8" in df
    if not (JOYAI_MAIN / "deploy").is_dir() or shutil.which("git") is None:
        pytest.skip("JoyAI main checkout not present")
    for p in patches:
        r = subprocess.run(["git", "-C", str(JOYAI_MAIN), "apply", "--check", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{p.name} no longer applies to JoyAI main: {r.stderr}"
    main_serve = (JOYAI_MAIN / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py").read_text()
    for flag in ("--pe-backend", "--full-body-detection", "--pe-timeout-s", "--use-pe"):
        assert flag in main_serve
    assert "JOYOMNI_LOAD_WARMUP_TIMEOUT_S" in (JOYAI_MAIN / "deploy" / "xvideo" / "serving" / "joyomni_streaming.py").read_text()


# ---------- populate_volume.sh ----------

def test_populate_facepp_refuses_without_space(tmp_path):
    script = ROOT / "joyai" / "scripts" / "populate_volume.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "hf").write_text("#!/usr/bin/env bash\necho HF_CALLED \"$@\"\n")
    (fake_bin / "python3").write_text("#!/usr/bin/env bash\nexit 0\n")
    # A fake df reporting 10 GB free, so the test does not depend on the machine's real disk
    # (GitHub's runners have about 86 GB free, enough for face++).
    (fake_bin / "df").write_text("#!/usr/bin/env bash\n"
                                 "echo 'Filesystem 1-blocks Used Available Capacity Mounted on'\n"
                                 "echo 'fake 107374182400 96636764160 10737418240 90% /vol'\n")
    for f in fake_bin.iterdir():
        f.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", VOLUME_ROOT=str(tmp_path / "vol"), FACEPP="1")
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    # 10 GB free < 31 GB: the script must refuse, name the shortfall, and not download.
    assert r.returncode == 3, r.stdout + r.stderr
    assert "NOT ENOUGH SPACE" in r.stderr and "short by 22 GB" in r.stderr and "Do NOT delete 0811" in r.stderr
    assert "HF_CALLED" not in r.stdout
    sh = script.read_text()
    assert "joyai_video_edit_dit_1101_face++.pth" in sh and "HF_HUB_ENABLE_HF_TRANSFER=0" in sh
    assert "joyai_video_edit_dit_0811.pth" in sh  # the default download is unchanged


# ---------- check_dit_checkpoint.py ----------

def _load_checker():
    spec = importlib.util.spec_from_file_location("check_dit_checkpoint", ROOT / "joyai" / "scripts" / "check_dit_checkpoint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _T:
    def __init__(self, *shape):
        self.shape = shape


def test_checkpoint_checker_compares_layouts(tmp_path):
    mod = _load_checker()
    a = tmp_path / "a.pth"
    b = tmp_path / "b.pth"
    a.write_bytes(b"a" * 10)
    b.write_bytes(b"b" * 12)
    fake = {str(a): {"x.weight": _T(4, 4), "y.bias": _T(4)},
            str(b): {"state_dict": {"x.weight": _T(4, 4), "y.bias": _T(4)}}}
    sa, sb = mod.summarize(str(a), fake.get), mod.summarize(str(b), fake.get)
    assert sa["tensors"] == 2 and sa["params"] == 20 and sa["bytes"] == 10 and sb["tensors"] == 2
    assert mod.compare(sa, sb) == []
    fake[str(b)] = {"x.weight": _T(4, 8)}
    diffs = mod.compare(sa, mod.summarize(str(b), fake.get))
    assert any("shape differs" in d for d in diffs) and any("only in" in d for d in diffs)
    assert mod.main([str(tmp_path / "missing.pth")]) == 2


# ---------- adapter: wording + use_pe + detection_mode ----------

def test_reference_wording_rewrite_for_every_app_category():
    expect = {
        "tops": "Substitute the current top with the garment from Image 1.",
        "bottoms": "Substitute the current pants or skirt with the garment from Image 1.",
        "dresses": "Substitute the current outfit with the dress from Image 1.",
        "outerwear": "Put the outerwear from Image 1 on the person, over their current outfit.",
        "shoes": "Substitute the current shoes with the shoes from Image 1.",
        "bags": "Have the person carry the bag from Image 1.",
        "accessories": "Put the accessory from Image 1 on the person.",
        "fallback": "Substitute the matching clothing item with the garment from Image 1.",
    }
    for cat, text in APP_INSTRUCTIONS.items():
        assert reword_reference(text + KEEP, "image1") == expect[cat] + KEEP, cat
        assert reword_reference(text + KEEP, "reference") == text + KEEP
    full = (APP_INSTRUCTIONS["tops"] + " The reference garment is the Blue Tee: cotton. "
            "Match its exact color, pattern, print, logo, fabric and construction details." + KEEP)
    out = reword_reference(full, "image1")
    assert out.startswith("Substitute the current top with the garment from Image 1.")
    assert "The reference garment is the Blue Tee" in out    # details sentence untouched
    assert reword_reference("", "image1") == ""


def _start_with(fake, **kw):
    with TestClient(create_app(settings(fake.port, **kw))) as c:
        with c.websocket_connect(f"/ws?token={tok()}") as ws:
            while ws.receive_json()["type"] != "ready":
                pass
            ws.send_json({"type": "set_reference_image", "image_b64": base64.b64encode(b"TOP1").decode()})
            assert ws.receive_json()["name"] == "set_reference_image"
            ws.send_json({"type": "set_prompt", "prompt": "long", "instruction": APP_INSTRUCTIONS["tops"] + KEEP})
            assert ws.receive_json()["name"] == "set_prompt"
            assert wait_for(lambda: len(fake.starts) == 1)
    return fake.starts[0]


def test_start_payload_defaults_match_today():
    with FakeJoyAI() as fake:
        start = _start_with(fake)
    assert start["use_pe"] is False
    assert start["gate_enabled"] is True and start["detection_mode"] == "near"
    assert start["prompt"].startswith("Substitute the current top with the garment in the reference image.")


def test_start_payload_switches_on():
    with FakeJoyAI() as fake:
        start = _start_with(fake, use_pe=True, ref_wording="image1", detection_mode="full")
    assert start["use_pe"] is True
    assert start["detection_mode"] == "full" and start["gate_enabled"] is True
    assert start["prompt"].startswith("Substitute the current top with the garment from Image 1.")


def test_face_gate_off_turns_detection_off_too():
    with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port, detection_mode="full"))) as c:
        with c.websocket_connect(f"/ws?token={tok()}") as ws:
            while ws.receive_json()["type"] != "ready":
                pass
            ws.send_json({"type": "set_prompt", "prompt": "shoes", "instruction": APP_INSTRUCTIONS["shoes"], "face_gate": False})
            assert ws.receive_json()["name"] == "set_prompt"
            assert wait_for(lambda: len(fake.starts) == 1)
    start = fake.starts[0]
    assert start["gate_enabled"] is False and start["detection_mode"] == "off"


def test_settings_env_defaults(monkeypatch):
    for k in ("JOYAI_USE_PE", "JOYAI_REF_WORDING", "JOYAI_DETECTION_MODE"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(session_signing_secret="x")
    assert s.use_pe is False and s.ref_wording == "reference" and s.detection_mode == "near"
    monkeypatch.setenv("JOYAI_USE_PE", "1")
    monkeypatch.setenv("JOYAI_REF_WORDING", "IMAGE1")
    monkeypatch.setenv("JOYAI_DETECTION_MODE", "full")
    s = Settings(session_signing_secret="x")
    assert s.use_pe is True and s.ref_wording == "image1" and s.detection_mode == "full"


# ---------- no-face try-ons (shoes, bags, accessories, trousers, back camera) keep working ----------

def test_no_face_tryon_turns_every_gate_off_on_main_and_on_our_pin(tmp_path):
    """The app sends face_gate=false for shoes/bottoms/back camera. Our pinned JoyAI read
    gate_enabled; main reads detection_mode, where "off" also disables the full-body entry gate
    and the mid-session presence hold (serve: entry_gate = detection_mode != "off"; gate_on = entry_gate)."""
    for mode in ("near", "full"):
        with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port, detection_mode=mode))) as c:
            with c.websocket_connect(f"/ws?token={tok()}") as ws:
                while ws.receive_json()["type"] != "ready":
                    pass
                ws.send_json({"type": "set_reference_image", "image_b64": base64.b64encode(b"SHOE").decode()})
                assert ws.receive_json()["name"] == "set_reference_image"
                ws.send_json({"type": "set_prompt", "prompt": "shoes", "instruction": APP_INSTRUCTIONS["shoes"] + KEEP,
                              "face_gate": False, "category": "shoes"})
                assert ws.receive_json()["name"] == "set_prompt"
                assert wait_for(lambda: len(fake.starts) == 1)
        start = fake.starts[0]
        assert start["gate_enabled"] is False and start["detection_mode"] == "off", mode
    # start_upstream.sh never asks for full-body detection unless the owner sets the switch.
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31")
    assert "--full-body-detection" not in env["_stdout"]
    if (JOYAI_MAIN / "deploy").is_dir():
        serve = (JOYAI_MAIN / "deploy" / "xvideo" / "serving" / "serve_joyomni_streaming.py").read_text()
        assert 'entry_gate = detection_mode != "off"' in serve and "gate_on = entry_gate" in serve


# ---------- face-check tuning pass-through (7 Nov 2026: cap shading the face) ----------

def test_face_check_tuning_off_by_default(tmp_path):
    out = run(tmp_path, "NVIDIA GeForce RTX 5090|31")["_stdout"]
    for flag in ("--face-gate-score", "--presence-absent-frames", "--presence-return-frames",
                 "--person-count-change-frames", "--face-present-min-ratio"):
        assert flag not in out


def test_face_check_tuning_passes_numbers_and_ignores_typos(tmp_path):
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31", JOYAI_FACE_GATE_SCORE="0.25",
              JOYAI_PRESENCE_ABSENT_FRAMES="36", JOYAI_PERSON_COUNT_CHANGE_FRAMES="72",
              JOYAI_FACE_PRESENT_MIN_RATIO="abc")
    out = env["_stdout"]
    assert "--face-gate-score 0.25" in out
    assert "--presence-absent-frames 36" in out
    assert "--person-count-change-frames 72" in out
    assert "--face-present-min-ratio" not in out                 # typo: JoyAI's default, no crash
    assert "--presence-return-frames" not in out                 # unset: JoyAI's default


def test_populate_facepp_without_pip_uses_curl(tmp_path):
    script = ROOT / "joyai" / "scripts" / "populate_volume.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # python3 without pip (runpod/base), no hf CLI, plenty of space, and a curl that records its call.
    (fake_bin / "python3").write_text("#!/usr/bin/env bash\nexit 1\n")
    (fake_bin / "df").write_text("#!/usr/bin/env bash\n"
                                 "echo 'Filesystem 1-blocks Used Available Capacity Mounted on'\n"
                                 "echo 'fake 161061273600 53687091200 107374182400 33% /vol'\n")
    (fake_bin / "curl").write_text("#!/usr/bin/env bash\necho CURL_CALLED \"$@\"\nexit 0\n")
    for f in fake_bin.iterdir():
        f.chmod(0o755)
    path = f"{fake_bin}:/usr/bin:/bin"
    env = dict(os.environ, PATH=path, VOLUME_ROOT=str(tmp_path / "vol"), FACEPP="1")
    r = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert "CURL_CALLED" in r.stdout and "face%2B%2B.pth" in r.stdout and "-C -" in r.stdout
    assert "no pip / hf CLI" in r.stdout
    # The fake curl wrote nothing, so the size check must report it as incomplete (never "Done").
    assert r.returncode != 0 and "Done." not in r.stdout

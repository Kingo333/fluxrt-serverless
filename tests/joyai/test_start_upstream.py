"""Runs joyai/start_upstream.sh with a fake GPU query and a fake JoyAI server."""
import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "joyai" / "start_upstream.sh"


def run(tmp_path: Path, gpu: str, **env_overrides) -> dict:
    vol = tmp_path / "vol"
    ckpt = vol / "checkpoints"
    for p in ("JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth", "JoyAI-Video-Edit/vae/x",
              "MiMo-VL-7B-RL-2508/config.json"):
        (ckpt / p).parent.mkdir(parents=True, exist_ok=True)
        (ckpt / p).write_text("x")
    joyai_dir = tmp_path / "deploy"
    (joyai_dir / "xvideo" / "serving").mkdir(parents=True)
    (joyai_dir / "xvideo" / "serving" / "serve_joyomni_streaming.py").write_text("")
    # Fake python: the GPU query prints `gpu`; running the server dumps the env it got.
    fake_py = tmp_path / "python"
    fake_py.write_text(
        "#!/usr/bin/env bash\n"
        f'if [ "$1" = "-c" ]; then echo "{gpu}"; exit 0; fi\n'
        'case "$1" in *resolve_model.py) exec python3 "$@";; esac\n'
        "env | grep -E '^(JOYOMNI_|TORCHINDUCTOR_CACHE_DIR)' \n"
    )
    fake_py.chmod(fake_py.stat().st_mode | stat.S_IEXEC)
    env = {k: v for k, v in os.environ.items() if not k.startswith("JOYOMNI_")}
    env.update(JOYAI_PYTHON=str(fake_py), JOYAI_DIR=str(joyai_dir), JOYAI_VOLUME_ROOT=str(vol))
    env.update(env_overrides)
    proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    out = proc.stdout
    env_out = dict(line.split("=", 1) for line in out.splitlines() if "=" in line and not line.startswith("["))
    env_out["_stdout"] = out
    return env_out


@pytest.mark.parametrize("gpu,expected,cache", [
    ("NVIDIA GeForce RTX 5090|31", "1", "cache_rtx5090"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition|94", "0", "cache_full"),
    ("unknown|0", "1", "cache_rtx5090"),
])
def test_profile_by_gpu(tmp_path, gpu, expected, cache):
    env = run(tmp_path, gpu)
    assert env["JOYOMNI_SAGE_ATTN"] == expected
    assert env["JOYOMNI_FP8_FAST_ACCUM"] == expected
    assert env["JOYOMNI_LOW_VRAM"] == expected
    assert env["TORCHINDUCTOR_CACHE_DIR"].endswith(f"{cache}/torchinductor")


def test_endpoint_env_overrides_profile(tmp_path):
    env = run(tmp_path, "NVIDIA RTX PRO 6000 Blackwell Server Edition|94", JOYOMNI_SAGE_ATTN="1")
    assert env["JOYOMNI_SAGE_ATTN"] == "1"
    assert env["JOYOMNI_LOW_VRAM"] == "0"


def test_model_version_baseline_by_default(tmp_path):
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31")
    assert "model version: joyai-baseline" in env["_stdout"]
    assert "joyai_video_edit_dit_0811.pth" in env["_stdout"]


def test_model_version_from_registry(tmp_path):
    import json
    models = tmp_path / "vol" / "models"
    (models / "exp1").mkdir(parents=True)
    (models / "exp1" / "dit.pth").write_text("x")
    (models / "registry.json").write_text(json.dumps({"azyah-vton-experimental": {"dit": "exp1/dit.pth"}}))
    env = run(tmp_path, "NVIDIA GeForce RTX 5090|31", JOYAI_MODEL_VERSION="azyah-vton-experimental")
    assert f"-> {models}/exp1/dit.pth" in env["_stdout"]


def test_unknown_model_version_stops_the_worker(tmp_path):
    with pytest.raises(RuntimeError, match="no registry|not in"):
        run(tmp_path, "NVIDIA GeForce RTX 5090|31", JOYAI_MODEL_VERSION="azyah-vton-candidate")

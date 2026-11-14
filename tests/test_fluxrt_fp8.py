"""FP8 favourite #2 (FLUXRT_PRECISION=fp8) and the optional compile switch, worker side.

Off by default: unset = today's INT8 model and no compile. The FluxRT side (fluxrt/azyah_fuse_lora.py and the
patch) is covered in test_fuse_lora_patch.py.
"""
import collections
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

COMPILER_ENV = ("TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE",
                "TRITON_CACHE_DIR")
ENV = ("FLUXRT_PRECISION", "FLUXRT_COMPILE", "FLUXRT_COMPILE_CACHE_DIR", "ENABLE_INT8", "FLUXRT_LORA_PATH",
       "FLUXRT_LORA_FUSE", "FLUXRT_ALLOW_RESEARCH_MODEL", "FLUXRT_ROOT", "MODEL_ROOT", "RUNPOD_VOLUME_ROOT",
       "FLUXRT_CACHE_MASK", "FLUXRT_ANCHOR", "FLUXRT_COMPOSITE", "FLUXRT_REF_CLEAN", "FLUXRT_FACE_PASTE")
Usage = collections.namedtuple("Usage", "total used free")


def load(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("SESSION_SIGNING_SECRET", "s")
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    for k in COMPILER_ENV:  # set-then-delete so monkeypatch also removes what the server sets at import
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    monkeypatch.setenv("FLUXRT_ROOT", str(tmp_path / "fluxrt"))
    (tmp_path / "fluxrt").mkdir(exist_ok=True)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("server", None)
    return importlib.import_module("server")


def config(s):
    return json.loads(s.write_runtime_config().read_text())


def cleared_lora(tmp_path):
    lora = tmp_path / "fluxrt_lora" / "blend05s065" / "pytorch_lora_weights.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"x" * 2_000_000)
    (lora.parent / "selection.json").write_text(json.dumps({"commercial_use_cleared": True}))
    return lora


def test_default_is_todays_int8_model_without_compile(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path)
    assert s.PRECISION == "int8" and s.COMPILE is False
    cfg = config(s)
    assert cfg["precision"] == "int8" and cfg["compile_models"] is False
    assert cfg["precision_marker"] == str(tmp_path / "fluxrt" / "azyah_precision.json")
    sw = s.live_switches(False)
    assert (sw["precision"], sw["precision_active"], sw["compile"], sw["compile_cache"]) == ("int8", None, False, "off")
    import os
    assert not any(k in os.environ for k in COMPILER_ENV)          # compile caches untouched when off


def test_fp8_switch_and_its_guards(monkeypatch, tmp_path):
    assert config(load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp8"))["precision"] == "fp8"
    assert load(monkeypatch, tmp_path, FLUXRT_PRECISION=" FP8 ").PRECISION == "fp8"
    assert load(monkeypatch, tmp_path, FLUXRT_PRECISION="bf16").PRECISION == "int8"     # typo: today's model
    assert load(monkeypatch, tmp_path, FLUXRT_PRECISION="").PRECISION == "int8"
    # FP8 is built by FluxRT's quantized loader, so it needs ENABLE_INT8 (on by default)
    assert load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp8", ENABLE_INT8="false").PRECISION == "int8"
    # the LoRA must be fused into the weights before FP8
    lora = cleared_lora(tmp_path)
    s = load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp8", FLUXRT_LORA_PATH=str(lora), FLUXRT_LORA_FUSE="0")
    assert s.PRECISION == "int8"
    s = load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp8", FLUXRT_LORA_PATH=str(lora))
    cfg = config(s)
    assert cfg["precision"] == "fp8" and cfg["use_lora"] is True and cfg["fuse_lora"] is True
    assert cfg["lora_weights_path"] == str(lora) and cfg["lora_scale"] == 1.0  # favourite #2 unchanged


def test_compile_switch(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path, FLUXRT_COMPILE="1")
    assert config(s)["compile_models"] is True and s.live_switches(False)["compile"] is True
    assert s.COMPILE_CACHE_USED is False and "inside this worker only" in s.COMPILE_CACHE_STATUS
    s = load(monkeypatch, tmp_path, FLUXRT_COMPILE="1", FLUXRT_COMPILE_CACHE_DIR=str(tmp_path / "cc"))
    assert (tmp_path / "cc").is_dir() and s.COMPILE_CACHE_STATUS.startswith(("kept at", "not kept"))


def test_compile_cache_only_with_room(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path)
    import os
    monkeypatch.setattr(s.shutil, "disk_usage", lambda p: Usage(150e9, 140e9, 10e9))
    used, status = s.setup_compile_cache(str(tmp_path / "cc"))
    assert used is True and status == f"kept at {tmp_path / 'cc'} (10 GB free)"
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "cc" / "inductor")
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "cc" / "triton")
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1" and os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"
    for k in COMPILER_ENV:
        monkeypatch.delenv(k)
    monkeypatch.setattr(s.shutil, "disk_usage", lambda p: Usage(150e9, 149e9, 1e9))
    used, status = s.setup_compile_cache(str(tmp_path / "cc"))
    assert used is False and status.startswith("not kept: only 1.0 GB free")
    assert not any(k in os.environ for k in COMPILER_ENV)          # a full volume never stops the worker
    blocker = tmp_path / "file"
    blocker.write_text("x")
    used, status = s.setup_compile_cache(str(blocker / "cc"))
    assert used is False and "not usable" in status


def test_storage_report_names_models_disks_and_lora(monkeypatch, tmp_path):
    lora = cleared_lora(tmp_path)
    vol = tmp_path / "volume"
    vol.mkdir()
    s = load(monkeypatch, tmp_path, FLUXRT_LORA_PATH=str(lora), RUNPOD_VOLUME_ROOT=str(vol))
    root = tmp_path / "fluxrt"
    (root / "FLUX.2-klein-4B" / "transformer").mkdir(parents=True)
    (root / "FLUX.2-klein-4B" / "transformer" / "w.safetensors").write_bytes(b"x" * 300_000_000)
    (root / "FLUX.2-klein-4B-int8").mkdir()
    lines = s.storage_report()
    assert lines[0] == (f"model full precision: {root / 'FLUX.2-klein-4B'} (0.3 GB, transformer folder present; "
                        "downloaded from Hugging Face black-forest-labs/FLUX.2-klein-4B at start-up)")
    assert lines[1].startswith(f"model int8: {root / 'FLUX.2-klein-4B-int8'} (0.0 GB;")
    assert lines[2].startswith(f"worker disk {root}: ") and "GB free of" in lines[2]
    assert lines[3].startswith(f"network volume {vol}: ") and "GB free of" in lines[3]
    assert lines[4] == f"LoRA {lora} (2 MB)"
    assert lines[5] == "transformer precision asked: int8; compile: off"
    s = load(monkeypatch, tmp_path, RUNPOD_VOLUME_ROOT=str(tmp_path / "none"), FLUXRT_PRECISION="fp8",
             FLUXRT_COMPILE="1")
    lines = s.storage_report()
    assert "MISSING" in lines[0] or "transformer folder present" in lines[0]
    assert any(l == f"network volume {tmp_path / 'none'}: not mounted" for l in lines)
    assert lines[-1].startswith("transformer precision asked: fp8; compile: on, cache inside this worker only")


def test_precision_marker_is_read_as_one_line(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path)
    m = tmp_path / "m.json"
    assert s.read_precision_marker(m) == "unknown"                  # older FluxRT patch: nothing written
    m.write_text("{not json")
    assert s.read_precision_marker(m) == "unknown"
    m.write_text(json.dumps({"precision": "fp8", "lora_fused": True, "fp8_layers": 196}))
    assert s.read_precision_marker(m) == "fp8 (196 layers)"
    m.write_text(json.dumps({"precision": "int8", "lora_fused": True}))
    assert s.read_precision_marker(m) == "int8"
    m.write_text(json.dumps({"precision": "int8", "lora_fused": True, "fp8_error": "FP8 unavailable: no CUDA GPU"}))
    assert s.read_precision_marker(m) == "int8 (FP8 unavailable: FP8 unavailable: no CUDA GPU)"


class _FakeStreamProcessor:
    """FluxRT's StreamProcessor at warm-up: its model process writes the marker named in the config."""
    writes = None

    def __init__(self, config_path):
        self.cfg = json.loads(Path(config_path).read_text())
        self.model_inference_subprocess = types.SimpleNamespace(process=None)
        self.output_scheduler_subprocess = types.SimpleNamespace(process=None)

    def start(self):
        if self.writes is not None:
            Path(self.cfg["precision_marker"]).write_text(json.dumps(self.writes))

    def is_ready(self):
        return True


@pytest.mark.parametrize("writes, expected", [
    ({"precision": "fp8", "lora_fused": True, "fp8_layers": 196}, "fp8 (196 layers)"),
    ({"precision": "int8", "lora_fused": True, "fp8_error": "FP8 unavailable: compute capability 8.6 has no FP8 "
      "tensor cores"}, "int8 (FP8 unavailable: FP8 unavailable: compute capability 8.6 has no FP8 tensor cores)"),
    (None, "unknown"),                                              # a stale marker from an earlier start is ignored
])
def test_warmup_logs_where_models_are_and_what_was_built(monkeypatch, tmp_path, caplog, writes, expected):
    s = load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp8")
    monkeypatch.chdir(tmp_path)                                     # warm-up changes into FLUXRT_ROOT
    fake = type("SP", (_FakeStreamProcessor,), {"writes": writes})
    monkeypatch.setitem(sys.modules, "fluxrt", types.SimpleNamespace(StreamProcessor=fake))
    monkeypatch.setitem(sys.modules, "fluxrt.utils", types.SimpleNamespace(crop_maximal_rectangle=lambda *a: None))
    monkeypatch.setattr(s, "ensure_models", lambda: None)
    s.PRECISION_MARKER.write_text(json.dumps({"precision": "fp8", "fp8_layers": 1}))   # left by an earlier start
    with caplog.at_level("INFO", logger="fluxrt-lb"):
        s.warmup_blocking()
    assert s.STATE["status"] == "ready" and s.STATE["precision_active"] == expected
    text = caplog.text
    assert "warmup: model full precision: " in text and "warmup: worker disk " in text
    assert "warmup: network volume " in text and "warmup: transformer precision asked: fp8; compile: off" in text
    assert f"warmup: transformer built in {expected}" in text
    assert f"'precision_active': '{expected}'" in text              # also in the 'ready; live switches' line


ROOT = Path(__file__).resolve().parents[1]


def test_full_image_pins_packages_and_keeps_torchao_out_of_peft_sight():
    df = (ROOT / "Dockerfile.livecam").read_text()
    # every pip install of the FluxRT engine after torch uses today's exact versions
    assert df.count("-m pip install -c /tmp/fluxrt-constraints.txt") >= 5
    assert "COPY fluxrt/constraints.txt /tmp/fluxrt-constraints.txt" in df
    assert "torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128" in df
    # torchao only in its own folder (PEFT 0.21 refuses a visible torchao < 0.16), checked at build time
    assert "pip install --no-deps --target /opt/azyah-torchao torchao==0.14.1" in df
    assert df.count("torchao==") == 1 and "PEFT must not see torchao" in df and "import_torchao()" in df
    helper = (ROOT / "fluxrt" / "azyah_fuse_lora.py").read_text()
    assert 'os.environ.get("AZYAH_TORCHAO_PATH", "/opt/azyah-torchao")' in helper
    # WebRTC in the full image too, checked like the thin image does
    assert "-r /tmp/livecam-webrtc-requirements.txt" in df
    assert "assert webrtc.available(), webrtc.unavailable_reason()" in df
    pins = {}
    for line in (ROOT / "fluxrt" / "constraints.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, ver = line.split("==")
            pins[name.lower().replace("_", "-")] = ver
    assert pins["diffusers"] == "0.42.1" and pins["transformers"] == "5.22.2" and pins["peft"] == "0.21.2"
    assert pins["optimum-quanto"] == "0.2.7" and pins["accelerate"] == "1.15.0" and pins["aiortc"] == "1.16.0"
    assert pins["numpy"] == "2.2.6" and pins["opencv-python"] == "4.12.0.88"
    # torch and its CUDA packages come from the pinned torch install, torchao never goes in site-packages
    assert not [n for n in pins if n in ("torch", "torchvision", "triton", "torchao", "fluxrt") or n.startswith("nvidia-")]
    req = (ROOT / "livecam" / "requirements-webrtc.txt").read_text()
    assert "aiortc==" + pins["aiortc"] in req
    workflow = (ROOT / ".github" / "workflows" / "build-livecam-dual.yml").read_text()
    assert "- fluxrt/constraints.txt" in workflow

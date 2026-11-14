"""FluxRT 32206e8 engine settings: TAEF2 + Flow Upscaler on, reference image prep, client hints."""
import base64
import importlib
import io
import json
import sys

import numpy as np
from PIL import Image


def load(monkeypatch, **env):
    monkeypatch.setenv("SESSION_SIGNING_SECRET", "s")
    for k in ("ENABLE_TINY_VAE", "ENABLE_FLOW_UPSCALER", "FLUXRT_WIDTH", "FLUXRT_HEIGHT",
              "FLUXRT_REF_WIDTH", "FLUXRT_REF_HEIGHT", "FLUXRT_CLIENT_HINTS",
              "FLUXRT_LORA_PATH", "FLUXRT_ALLOW_RESEARCH_MODEL",
              "FLUXRT_LORA_FUSE", "FLUXRT_LORA_SCALE", "FLUXRT_SHARPEN",
              "FLUXRT_PRECISION", "FLUXRT_COMPILE", "FLUXRT_COMPILE_CACHE_DIR"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("server", None)
    return importlib.import_module("server")


def png_b64(w, h, rgb):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_upgrades_on_by_default(monkeypatch, tmp_path):
    s = load(monkeypatch, FLUXRT_WIDTH="288", FLUXRT_HEIGHT="512")
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    cfg = json.loads(s.write_runtime_config().read_text())
    assert cfg["enable_tiny_vae"] is True
    assert cfg["enable_flow_upscaler"] is True
    assert cfg["reference_image_resolution"] == {"width": 384, "height": 512}
    assert (s.OUT_WIDTH, s.OUT_HEIGHT) == (576, 1024)


def test_rollback_switches(monkeypatch, tmp_path):
    s = load(monkeypatch, ENABLE_TINY_VAE="false", ENABLE_FLOW_UPSCALER="false", FLUXRT_WIDTH="288", FLUXRT_HEIGHT="512")
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    cfg = json.loads(s.write_runtime_config().read_text())
    assert cfg["enable_tiny_vae"] is False and cfg["enable_flow_upscaler"] is False
    assert (s.OUT_WIDTH, s.OUT_HEIGHT) == (288, 512)


def test_downloads_pinned_extensions(monkeypatch, tmp_path):
    s = load(monkeypatch)
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(s, "_download_model", lambda repo, d, min_files, revision=None: calls.append((repo, d.name, revision)))
    s.ensure_models()
    names = {c[1]: c for c in calls}
    assert names["taef2"] == ("madebyollin/taef2", "taef2", s.TAEF2_REVISION)
    assert names["FlowUpscaler"] == ("TensorForger/FlowUpscaler", "FlowUpscaler", s.FLOW_UPSCALER_REVISION)
    assert len(s.TAEF2_REVISION) == 40 and len(s.FLOW_UPSCALER_REVISION) == 40


def test_reference_is_rgb_and_padded_not_stretched(monkeypatch):
    s = load(monkeypatch)
    # Wide red image: must stay red (RGB, not BGR) and be padded with white.
    arr = s.b64_to_reference_rgb(png_b64(400, 100, (255, 0, 0)))
    assert arr.shape == (512, 384, 3)
    centre = arr[256, 192]
    assert centre[0] > 200 and centre[2] < 50
    assert (arr[0, 0] == 255).all()


def test_ref_size_rounded_to_16(monkeypatch):
    s = load(monkeypatch, FLUXRT_REF_WIDTH="390", FLUXRT_REF_HEIGHT="500")
    assert (s.REF_WIDTH, s.REF_HEIGHT) == (384, 496)


def test_ready_hints(monkeypatch):
    s = load(monkeypatch, FLUXRT_WIDTH="288", FLUXRT_HEIGHT="512")
    m = s.ready_message(123.9)
    assert m["type"] == "ready" and m["session_deadline"] == 123
    assert m["engine"] == "fluxrt"
    assert (m["input_width"], m["input_height"], m["output_width"], m["output_height"]) == (288, 512, 576, 1024)
    s = load(monkeypatch, FLUXRT_CLIENT_HINTS="false")
    assert "engine" not in s.ready_message(1)


def test_lora_gate(monkeypatch, tmp_path):
    lora = tmp_path / "exp1" / "pytorch_lora_weights.safetensors"
    lora.parent.mkdir()
    lora.write_bytes(b"x")
    s = load(monkeypatch, FLUXRT_LORA_PATH=str(lora))
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    import pytest
    with pytest.raises(RuntimeError, match="provenance"):
        s.write_runtime_config()
    (lora.parent / "selection.json").write_text(json.dumps({"commercial_use_cleared": False, "notice": "ViViD"}))
    with pytest.raises(RuntimeError, match="not cleared"):
        s.write_runtime_config()
    s = load(monkeypatch, FLUXRT_LORA_PATH=str(lora), FLUXRT_ALLOW_RESEARCH_MODEL="1")
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    cfg = json.loads(s.write_runtime_config().read_text())
    assert cfg["use_lora"] is True and cfg["lora_weights_path"] == str(lora)


def test_no_lora_by_default(monkeypatch, tmp_path):
    s = load(monkeypatch)
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    assert "use_lora" not in json.loads(s.write_runtime_config().read_text())


def test_favourite_2_switches_default_to_plain_fluxrt(monkeypatch, tmp_path):
    s = load(monkeypatch)
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    assert s.LORA_FUSE is True and s.LORA_SCALE == 1.0 and s.SHARPEN == 0.0
    assert "fuse_lora" not in json.loads(s.write_runtime_config().read_text())   # no LoRA: nothing added
    lora = tmp_path / "blend05s065" / "pytorch_lora_weights.safetensors"
    lora.parent.mkdir()
    lora.write_bytes(b"x")
    (lora.parent / "selection.json").write_text(json.dumps({"commercial_use_cleared": True}))
    s = load(monkeypatch, FLUXRT_LORA_PATH=str(lora))
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    cfg = json.loads(s.write_runtime_config().read_text())
    assert cfg["use_lora"] is True and cfg["fuse_lora"] is True and cfg["lora_scale"] == 1.0
    s = load(monkeypatch, FLUXRT_LORA_PATH=str(lora), FLUXRT_LORA_FUSE="0", FLUXRT_LORA_SCALE="0.7")
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    cfg = json.loads(s.write_runtime_config().read_text())
    assert cfg["fuse_lora"] is False and cfg["lora_scale"] == 0.7
    s = load(monkeypatch, FLUXRT_SHARPEN="1.7")
    assert s.SHARPEN == 1.0                                                        # clamped
    s = load(monkeypatch, FLUXRT_SHARPEN="abc")
    assert s.SHARPEN == 0.0                                                        # typo = off


def test_sharpen_is_off_at_zero_and_adds_edge_contrast_otherwise(monkeypatch):
    from PIL import Image
    import numpy as np
    s = load(monkeypatch)
    rng = np.random.default_rng(0)
    base = np.zeros((64, 64, 3), np.uint8)
    base[:, 32:] = 180
    base = np.clip(base + rng.integers(0, 20, base.shape), 0, 255).astype(np.uint8)   # an edge plus texture
    img = Image.fromarray(base)
    assert s.sharpen_rgb(img, 0.0) is img
    out = np.asarray(s.sharpen_rgb(img, 0.3)).astype(int)
    edge = lambda a: np.abs(np.diff(a[:, :, 0].astype(int), axis=1)).mean()  # noqa: E731
    assert edge(out) > edge(base) * 1.05
    assert out.shape == base.shape
    # the encoder applies it only when FLUXRT_SHARPEN is set
    s2 = load(monkeypatch, FLUXRT_SHARPEN="0.3")
    plain = s.bgr_to_b64_jpeg(base[:, :, ::-1].copy())
    sharp = s2.bgr_to_b64_jpeg(base[:, :, ::-1].copy())
    assert plain != sharp and s.bgr_to_b64_jpeg(base[:, :, ::-1].copy()) == plain

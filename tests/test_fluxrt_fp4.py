"""FP4 favourite #2 (FLUXRT_PRECISION=fp4) and the tighter compile switches (owner, 2026-11-15).

FluxRT side: fluxrt/azyah_fuse_lora.py (FP4 build, edge blocks in FP8, test multiply against full precision) and the
two patches (FP4 -> FP8 -> INT8 chain; compile mode and tuning). Worker side: server.py settings and logging.
"""
import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fluxrt_fp8 import cleared_lora, config, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TORCHAO = os.environ.get("AZYAH_TORCHAO_PATH", "")


def _helper():
    spec = importlib.util.spec_from_file_location("azyah_fuse_lora_fp4", ROOT / "fluxrt" / "azyah_fuse_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny(torch, diffusers):
    """Three double and three single blocks, every block layer a multiple of 64 wide (FP4 sizes)."""
    try:
        return diffusers.Flux2Transformer2DModel(
            patch_size=1, in_channels=8, out_channels=8, num_layers=3, num_single_layers=3, attention_head_dim=32,
            num_attention_heads=2, joint_attention_dim=64, timestep_guidance_channels=16, mlp_ratio=2.0,
            axes_dims_rope=(8, 8, 8, 8), guidance_embeds=False).eval()
    except TypeError as e:
        pytest.skip(f"tiny Flux2 config not constructible here: {e}")


# ---------------------------------------------------------------- helper (no GPU)
def test_block_index_and_edge_blocks():
    h = _helper()
    assert h.block_index("transformer_blocks.12.attn.to_q") == ("double", 12)
    assert h.block_index("single_transformer_blocks.3.proj_mlp") == ("single", 3)
    assert h.block_index("x_embedder") is None and h.block_index("transformer_blocks.x.y") is None
    names = [f"transformer_blocks.{i}.ff" for i in range(5)] + [f"single_transformer_blocks.{i}.ff" for i in range(20)]
    assert h.edge_blocks(names, 1) == {("double", 0), ("double", 4), ("single", 0), ("single", 19)}
    assert h.edge_blocks(names, 2) == {("double", 0), ("double", 1), ("double", 3), ("double", 4),
                                       ("single", 0), ("single", 1), ("single", 18), ("single", 19)}
    assert h.edge_blocks(names, 0) == set()


def test_fp4_layer_choice_and_gpu_refusals(monkeypatch):
    torch = pytest.importorskip("torch")
    h = _helper()
    keep = {("double", 0)}
    assert h.fp4_layer("transformer_blocks.1.attn.to_q", torch.nn.Linear(128, 64), keep)
    assert not h.fp4_layer("transformer_blocks.0.attn.to_q", torch.nn.Linear(128, 64), keep)    # kept edge block
    assert not h.fp4_layer("transformer_blocks.1.attn.to_q", torch.nn.Linear(48, 64), keep)     # 48: not a multiple of 64
    assert not h.fp4_layer("proj_out", torch.nn.Linear(128, 64), keep)                         # outside the blocks
    assert not h.is_fp4_weight(torch.nn.Linear(16, 16).weight)
    assert h.fp4_unavailable("cpu") == "no CUDA GPU"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 9))           # RTX 4090
    assert "8.9 has no FP4" in h.fp4_unavailable("cuda:0")
    for cap in ((10, 0), (12, 0)):                                                              # B200, RTX 5090 / PRO 6000
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None, c=cap: c)
        assert h.fp4_unavailable("cuda:0") is None


def test_fp4_build_refuses_without_a_blackwell_gpu(tmp_path):
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    h = _helper()
    with pytest.raises(RuntimeError, match="FP4 unavailable: no CUDA GPU"):                    # the caller falls back to FP8
        h.build_fp4_transformer(diffusers.Flux2Transformer2DModel, str(tmp_path / "missing"), None, 1.0, "cpu",
                                torch.bfloat16)


def _torchao(monkeypatch, h, torch):
    """torchao's own folder, as in the image; FP4 multiplies emulated on the CPU (weights FP4, maths 16-bit)."""
    if importlib.util.find_spec("torchao") is None and not os.path.isdir(os.path.join(TORCHAO, "torchao")):
        pytest.skip("torchao not installed (point AZYAH_TORCHAO_PATH at its folder)")
    monkeypatch.setattr(h, "TORCHAO_PATH", TORCHAO)
    monkeypatch.setattr(sys, "path", list(sys.path))
    real_config, real_mm = h.import_torchao_fp4()
    # The GPU path multiplies in FP4 (DYNAMIC); the CPU can only emulate it, so the test uses WEIGHT_ONLY.
    monkeypatch.setattr(h, "import_torchao_fp4", lambda: (real_config, types.SimpleNamespace(DYNAMIC=real_mm.WEIGHT_ONLY)))
    monkeypatch.setattr(h, "fp4_unavailable", lambda d: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=None: "cpu (test)")
    import torchao.float8.inference as ao_inference
    import torchao.quantization.quant_api as ao_api
    monkeypatch.setattr(ao_api, "is_sm_at_least_89", lambda: True)
    monkeypatch.setattr(ao_inference, "is_sm_at_least_89", lambda: True)
    try:
        one = torch.tensor(1.0)
        a = torch.randn(32, 32).to(torch.float8_e4m3fn)
        torch._scaled_mm(a, a.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"this torch has no CPU FP8 matmul (the edge blocks): {e}")


def test_fp4_build_end_to_end_on_cpu(tmp_path, monkeypatch):
    """Favourite #2's blocks: the middle blocks in FP4, the first and last of each kind in FP8, nothing outside the
    blocks touched, the test multiply passes and the whole model stays close to full precision."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    diffusers = pytest.importorskip("diffusers")
    h = _helper()
    _torchao(monkeypatch, h, torch)
    M = diffusers.Flux2Transformer2DModel
    torch.manual_seed(0)
    model = _tiny(torch, diffusers)
    model.save_pretrained(tmp_path / "transformer")
    ref = h.load_fused_full_precision(M, str(tmp_path / "transformer"), None, 1.0, torch.bfloat16).eval()
    fp4 = h.build_fp4_transformer(M, str(tmp_path / "transformer"), None, 1.0, "cpu", torch.bfloat16, keep_edge_blocks=1)
    mods = dict(fp4.named_modules())
    block = [n for n, m in mods.items() if h.fp8_layer(n, m)]
    in4 = [n for n in block if h.is_fp4_weight(mods[n].weight)]
    in8 = [n for n in block if h.is_fp8_weight(mods[n].weight)]
    assert in4 and in8 and len(in4) + len(in8) == len(block)                 # nothing left in 16 bits
    assert all(h.block_index(n) in {("double", 1), ("single", 1)} for n in in4)   # only the middle blocks in FP4
    assert all(h.block_index(n)[1] in (0, 2) for n in in8)
    assert fp4.azyah_fp4_layers == len(in4) and fp4.azyah_fp8_layers == len(in8)
    assert 0 < fp4.azyah_fp4_error < h.FP4_MAX_RELATIVE_ERROR
    for name in ("proj_out", "x_embedder", "context_embedder"):
        assert not h.is_fp4_weight(mods[name].weight) and not h.is_fp8_weight(mods[name].weight)
    n_img, n_txt = 23, 5
    kw = dict(hidden_states=torch.randn(1, n_img, 8, dtype=torch.bfloat16),
              encoder_hidden_states=torch.randn(1, n_txt, 64, dtype=torch.bfloat16),
              timestep=torch.tensor([0.5], dtype=torch.bfloat16), img_ids=torch.zeros(1, n_img, 4),
              txt_ids=torch.zeros(1, n_txt, 4), return_dict=False)
    kw["img_ids"][..., 1] = torch.arange(n_img)
    with torch.no_grad():
        want, got = ref(**kw)[0].float(), fp4(**kw)[0].float()
    assert torch.isfinite(got).all()
    assert ((got - want).norm() / want.norm()).item() < 0.3


def test_fp4_build_raises_when_the_test_multiply_is_wrong(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    diffusers = pytest.importorskip("diffusers")
    h = _helper()
    _torchao(monkeypatch, h, torch)
    _tiny(torch, diffusers).save_pretrained(tmp_path / "transformer")
    monkeypatch.setattr(h, "relative_error", lambda out, ref: 1.7)             # a broken kernel
    with pytest.raises(RuntimeError, match="FP4: the test multiply .* is 1.70 off full precision"):
        h.build_fp4_transformer(diffusers.Flux2Transformer2DModel, str(tmp_path / "transformer"), None, 1.0, "cpu",
                                torch.bfloat16)
    with pytest.raises(RuntimeError, match="no layer suits FP4"):                # every block kept in FP8
        h.build_fp4_transformer(diffusers.Flux2Transformer2DModel, str(tmp_path / "transformer"), None, 1.0, "cpu",
                                torch.bfloat16, keep_edge_blocks=4)


# ---------------------------------------------------------------- the patches
def test_patches_try_fp4_then_fp8_then_int8_and_carry_the_compile_switches():
    fuse = (ROOT / "fluxrt/patches/fluxrt-fuse-lora.patch").read_text()
    speed = (ROOT / "fluxrt/patches/fluxrt-speed.patch").read_text()
    for must in ('chain = {"fp4": ("fp4", "fp8"), "fp8": ("fp8",)}.get(asked, ())', "build_fp4_transformer(",
                 "build_fp8_transformer(", 'keep_edge_blocks=int(self.config.get("fp4_keep_edge_blocks", 1))',
                 'precision_info[f"{want}_error"]', 'nxt = "FP8" if want == "fp4" else "INT8"',
                 '"fp4_test_error"', "write_precision_marker("):
        assert must in fuse, must
    for must in ('self.config.get("compile_mode", "default")', 'azyah_compile_kwargs["mode"] = azyah_mode',
                 "coordinate_descent_tuning = True", "**azyah_compile_kwargs", "AZYAH_FAST_MASK"):
        assert must in speed, must
    names = sorted(p.name for p in (ROOT / "fluxrt/patches").glob("*.patch"))
    assert names == ["fluxrt-anchor-reference.patch", "fluxrt-fuse-lora.patch", "fluxrt-speed.patch"]


# ---------------------------------------------------------------- the worker
def test_fp4_switch_and_its_guards(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp4")
    cfg = config(s)
    assert s.PRECISION == "fp4" and cfg["precision"] == "fp4" and cfg["fp4_keep_edge_blocks"] == 1
    assert load(monkeypatch, tmp_path, FLUXRT_PRECISION=" FP4 ").PRECISION == "fp4"
    assert load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp4", ENABLE_INT8="false").PRECISION == "int8"
    lora = cleared_lora(tmp_path)
    assert load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp4", FLUXRT_LORA_PATH=str(lora), FLUXRT_LORA_FUSE="0").PRECISION == "int8"
    cfg = config(load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp4", FLUXRT_LORA_PATH=str(lora)))
    assert cfg["use_lora"] is True and cfg["fuse_lora"] is True and cfg["lora_scale"] == 1.0   # favourite #2 fused first
    monkeypatch.setenv("FLUXRT_FP4_KEEP_EDGE_BLOCKS", "9")
    assert config(load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp4"))["fp4_keep_edge_blocks"] == 4   # kept in range
    monkeypatch.delenv("FLUXRT_FP4_KEEP_EDGE_BLOCKS")
    s = load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp8")                       # release #29: unchanged
    assert config(s)["precision"] == "fp8" and s.live_switches(False)["fp4_keep_edge_blocks"] is None


def test_compile_mode_and_tuning_are_off_unless_set(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path, FLUXRT_COMPILE="1")
    cfg = config(s)
    assert (cfg["compile_mode"], cfg["compile_tune"]) == ("default", False)       # today's compile
    sw = s.live_switches(False)
    assert (sw["compile_mode"], sw["compile_tune"]) == ("default", False)
    monkeypatch.setenv("FLUXRT_COMPILE_MODE", "max-autotune-no-cudagraphs")
    monkeypatch.setenv("FLUXRT_COMPILE_TUNE", "1")
    s = load(monkeypatch, tmp_path, FLUXRT_COMPILE="1")
    assert (config(s)["compile_mode"], config(s)["compile_tune"]) == ("max-autotune-no-cudagraphs", True)
    assert any("mode max-autotune-no-cudagraphs, tuning on" in line for line in s.storage_report())
    monkeypatch.setenv("FLUXRT_COMPILE_MODE", "reduce-overhead")                  # CUDA graphs: not offered
    assert load(monkeypatch, tmp_path, FLUXRT_COMPILE="1").COMPILE_MODE == "default"
    monkeypatch.delenv("FLUXRT_COMPILE_MODE")
    monkeypatch.delenv("FLUXRT_COMPILE_TUNE")
    assert load(monkeypatch, tmp_path).live_switches(False)["compile_mode"] is None   # no compile: nothing to show


def test_precision_marker_names_fp4_and_every_fallback(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path)
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"precision": "fp4", "fp4_layers": 180, "fp8_layers": 16, "fp4_test_error": 0.091}))
    assert s.read_precision_marker(m) == "fp4 (180 layers FP4, 16 FP8, test error 0.091)"
    m.write_text(json.dumps({"precision": "fp8", "fp8_layers": 196,
                             "fp4_error": "FP4 unavailable: compute capability 8.9 has no FP4 tensor cores"}))
    assert s.read_precision_marker(m) == ("fp8 (196 layers) (FP4 unavailable: FP4 unavailable: compute capability 8.9 "
                                          "has no FP4 tensor cores)")
    m.write_text(json.dumps({"precision": "int8", "fp4_error": "a", "fp8_error": "b"}))
    assert s.read_precision_marker(m) == "int8 (FP4 unavailable: a; FP8 unavailable: b)"
    m.write_text(json.dumps({"precision": "fp8", "fp8_layers": 196}))
    assert s.read_precision_marker(m) == "fp8 (196 layers)"                       # release #29's line, unchanged

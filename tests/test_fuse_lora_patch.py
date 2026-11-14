"""fluxrt-fuse-lora: the LoRA is folded into the weights before INT8, with the pre-quantized model's scheme."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _helper():
    spec = importlib.util.spec_from_file_location("azyah_fuse_lora", ROOT / "fluxrt" / "azyah_fuse_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_patch_carries_the_same_helper_as_the_repo_and_touches_the_right_places():
    patch = (ROOT / "fluxrt/patches/fluxrt-fuse-lora.patch").read_text()
    helper = (ROOT / "fluxrt/azyah_fuse_lora.py").read_text()
    section = patch.split("+++ b/src/fluxrt/stream_processor/azyah_fuse_lora.py", 1)[1].split("\ndiff --git", 1)[0]
    added = "\n".join(l[1:] for l in section.splitlines() if l.startswith("+") and not l.startswith("+++"))
    assert added.strip() == helper.strip()                      # the two copies cannot drift apart
    assert "fuse_lora_into_int8(" in patch and 'self.config["use_lora"] = False' in patch
    assert 'adapter_name="garment"' in patch and "set_adapters" in patch
    # FP8 (FLUXRT_PRECISION=fp8): built first when asked, INT8 when it cannot be, and the worker is told which
    assert 'self.config.get("precision", "int8")).lower() == "fp8"' in patch and "build_fp8_transformer(" in patch
    assert "falling back to INT8" in patch and 'precision_info["fp8_error"]' in patch
    assert 'self.config.get("precision_marker")' in patch and "write_precision_marker(" in patch
    # Dockerfile.livecam applies fluxrt/patches/*.patch in name order: anchor, this one, then the speed patch.
    names = sorted(p.name for p in (ROOT / "fluxrt/patches").glob("*.patch"))
    assert names == ["fluxrt-anchor-reference.patch", "fluxrt-fuse-lora.patch", "fluxrt-speed.patch"]
    assert "fluxrt/patches" in (ROOT / "Dockerfile.livecam").read_text()


def test_qmap_scheme_reads_one_scheme_and_refuses_mixed():
    h = _helper()
    assert h.qmap_scheme({"a": {"weights": "qint8", "activations": "none"}, "b": {"weights": "qint8", "activations": "none"}}) == ("qint8", None)
    with pytest.raises(ValueError):
        h.qmap_scheme({"a": {"weights": "qint8", "activations": "none"}, "b": {"weights": "qint4", "activations": "none"}})


def test_fuse_then_quantize_matches_the_blend_on_a_tiny_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("optimum.quanto")
    st = pytest.importorskip("safetensors.torch")
    diffusers = pytest.importorskip("diffusers")
    from optimum.quanto import quantization_map
    Model = diffusers.Flux2Transformer2DModel
    try:
        model = Model(patch_size=1, in_channels=8, out_channels=8, num_layers=1, num_single_layers=1,
                      attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
                      timestep_guidance_channels=16, mlp_ratio=2.0, axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False)
    except TypeError as e:  # this diffusers build takes other arguments; the maths below is what matters
        pytest.skip(f"tiny Flux2 config not constructible here: {e}")
    model.eval()
    mdir = tmp_path / "transformer"
    model.save_pretrained(mdir)
    # a LoRA on the layers the training script targets, with the diffusers settings record (rank 4, alpha 4)
    targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and (n.endswith((".to_q", ".to_k", ".to_v", ".to_out.0", ".to_qkv_mlp_proj")) or ".attn.to_out" in n and n.endswith("to_out"))]
    assert targets
    g = torch.Generator().manual_seed(0)
    sd, expected = {}, {}
    for n in targets:
        lin = dict(model.named_modules())[n]
        A = torch.randn(4, lin.in_features, generator=g) * 0.1
        B = torch.randn(lin.out_features, 4, generator=g) * 0.1
        sd[f"transformer.{n}.lora_A.weight"], sd[f"transformer.{n}.lora_B.weight"] = A, B
        expected[n] = lin.weight.detach().clone() + 0.65 * (B @ A)
    lora = tmp_path / "pytorch_lora_weights.safetensors"
    st.save_file(sd, str(lora), metadata={"format": "pt", "lora_adapter_metadata": json.dumps(
        {"transformer.r": 4, "transformer.lora_alpha": 4, "transformer.target_modules": sorted({t.rsplit(".", 1)[-1] for t in targets})})})
    # a "pre-quantized model" scheme: every Linear in INT8 weights, activations untouched
    int8 = tmp_path / "int8"
    int8.mkdir()
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    (int8 / "quanto_qmap.json").write_text(json.dumps({n: {"weights": "qint8", "activations": "none"} for n in linears}))

    h = _helper()
    fused = h.fuse_lora_into_int8(Model, str(mdir), str(int8), str(lora), 0.65, "cpu", torch.float32)
    assert set(quantization_map(fused)) == set(linears)          # same layers quantized as the pre-quantized model
    for n in targets:
        q = dict(fused.named_modules())[n]
        w = q.weight.dequantize() if hasattr(q.weight, "dequantize") else q.weight
        err = (w.float() - expected[n]).abs().max().item()
        step = expected[n].abs().amax(dim=1).max().item() / 127      # one INT8 step of the largest channel
        assert err <= step * 1.01, (n, err, step)                    # fused weight = W + 0.65 B A, then INT8 rounding
    assert not any(".lora_" in k for k in fused.state_dict())       # no adapter layers left behind


def _tiny(torch, diffusers):
    try:
        return diffusers.Flux2Transformer2DModel(
            patch_size=1, in_channels=8, out_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=8,
            num_attention_heads=2, joint_attention_dim=16, timestep_guidance_channels=16, mlp_ratio=2.0,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
    except TypeError as e:
        pytest.skip(f"tiny Flux2 config not constructible here: {e}")


def _tiny_lora(torch, st, model, path, scale):
    """A rank-4 LoRA on the attention layers and what the fused weights must be."""
    targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
               and n.endswith((".to_q", ".to_k", ".to_v", ".to_out.0", ".to_qkv_mlp_proj"))]
    g = torch.Generator().manual_seed(1)
    sd, expected = {}, {}
    for n in targets:
        lin = dict(model.named_modules())[n]
        A = torch.randn(4, lin.in_features, generator=g) * 0.1
        B = torch.randn(lin.out_features, 4, generator=g) * 0.1
        sd[f"transformer.{n}.lora_A.weight"], sd[f"transformer.{n}.lora_B.weight"] = A, B
        expected[n] = lin.weight.detach().clone() + scale * (B @ A)
    st.save_file(sd, str(path), metadata={"format": "pt", "lora_adapter_metadata": json.dumps(
        {"transformer.r": 4, "transformer.lora_alpha": 4,
         "transformer.target_modules": sorted({t.rsplit(".", 1)[-1] for t in targets})})})
    return expected


def test_fused_full_precision_is_the_exact_blend(tmp_path):
    """The step INT8 and FP8 share: W + scale x B A, nothing else changed."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    st = pytest.importorskip("safetensors.torch")
    diffusers = pytest.importorskip("diffusers")
    model = _tiny(torch, diffusers)
    model.save_pretrained(tmp_path / "transformer")
    expected = _tiny_lora(torch, st, model, tmp_path / "lora.safetensors", 0.65)
    h = _helper()
    fused = h.load_fused_full_precision(diffusers.Flux2Transformer2DModel, str(tmp_path / "transformer"),
                                        str(tmp_path / "lora.safetensors"), 0.65, torch.float32)
    mods = dict(fused.named_modules())
    for n, w in expected.items():
        assert torch.allclose(mods[n].weight, w, atol=1e-6), n
    untouched = dict(model.named_modules())
    assert torch.equal(mods["proj_out"].weight, untouched["proj_out"].weight)
    assert not any(".lora_" in k for k in fused.state_dict())
    plain = h.load_fused_full_precision(diffusers.Flux2Transformer2DModel, str(tmp_path / "transformer"), None,
                                        0.65, torch.float32)
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), model.state_dict().values()))


def test_fp8_layer_choice_and_cpu_refusal(tmp_path):
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    h = _helper()
    model = _tiny(torch, diffusers)
    chosen = [n for n, m in model.named_modules() if h.fp8_layer(n, m)]
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    assert chosen and all(n.startswith(("transformer_blocks.", "single_transformer_blocks.")) for n in chosen)
    assert set(linears) - set(chosen) >= {"x_embedder", "context_embedder", "proj_out", "norm_out.linear"}
    assert not h.fp8_layer("transformer_blocks.0.attn.to_q", torch.nn.Linear(16, 24))     # 24: not a multiple of 16
    assert not h.fp8_layer("transformer_blocks.0.norm", torch.nn.LayerNorm(16))
    assert h.fp8_unavailable("cpu") == "no CUDA GPU"
    with pytest.raises(RuntimeError, match="FP8 unavailable: no CUDA GPU"):          # caller falls back to INT8
        h.build_fp8_transformer(diffusers.Flux2Transformer2DModel, str(tmp_path / "missing"), None, 1.0, "cpu",
                                torch.bfloat16)
    assert not h.is_fp8_weight(torch.nn.Linear(16, 16).weight)


def test_precision_marker_is_best_effort(tmp_path):
    h = _helper()
    h.write_precision_marker(tmp_path / "m.json", {"precision": "fp8", "fp8_layers": 3})
    assert json.loads((tmp_path / "m.json").read_text()) == {"precision": "fp8", "fp8_layers": 3}
    h.write_precision_marker(None, {"precision": "int8"})                            # no path: nothing to do
    h.write_precision_marker(tmp_path / "no" / "such" / "dir" / "m.json", {"precision": "int8"})   # never raises


def test_fp8_build_end_to_end_on_cpu(tmp_path, monkeypatch):
    """torchao's FP8 conversion of favourite #2's blocks, run on the CPU (FP8 maths emulated there): the layers
    picked are converted, the rest stays full precision, the 17-token probe passes and the output stays close
    to full precision. The GPU checks are bypassed for this test only."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    st = pytest.importorskip("safetensors.torch")
    diffusers = pytest.importorskip("diffusers")
    h = _helper()
    folder = os.environ.get("AZYAH_TORCHAO_PATH", "")       # torchao's own folder, as in the image
    if importlib.util.find_spec("torchao") is None and not os.path.isdir(os.path.join(folder, "torchao")):
        pytest.skip("torchao not installed (point AZYAH_TORCHAO_PATH at its folder)")
    monkeypatch.setattr(h, "TORCHAO_PATH", folder)
    monkeypatch.setattr(sys, "path", list(sys.path))          # import_torchao adds the folder; undone afterwards
    h.import_torchao()
    import torchao.float8.inference as ao_inference
    import torchao.quantization.quant_api as ao_api
    try:
        one = torch.tensor(1.0)
        a = torch.randn(32, 32).to(torch.float8_e4m3fn)
        torch._scaled_mm(a, a.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"this torch has no CPU FP8 matmul: {e}")
    monkeypatch.setattr(ao_api, "is_sm_at_least_89", lambda: True)
    monkeypatch.setattr(ao_inference, "is_sm_at_least_89", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=None: "cpu (test)")
    monkeypatch.setattr(h, "fp8_unavailable", lambda d: None)
    M = diffusers.Flux2Transformer2DModel
    torch.manual_seed(0)
    model = _tiny(torch, diffusers)
    model.save_pretrained(tmp_path / "transformer")
    _tiny_lora(torch, st, model, tmp_path / "lora.safetensors", 0.65)
    ref = h.load_fused_full_precision(M, str(tmp_path / "transformer"), str(tmp_path / "lora.safetensors"), 0.65,
                                      torch.bfloat16).eval()
    fp8 = h.build_fp8_transformer(M, str(tmp_path / "transformer"), str(tmp_path / "lora.safetensors"), 0.65,
                                  "cpu", torch.bfloat16)
    picked = [n for n, m in fp8.named_modules() if h.fp8_layer(n, m)]
    mods = dict(fp8.named_modules())
    assert fp8.azyah_fp8_layers == len(picked) and all(h.is_fp8_weight(mods[n].weight) for n in picked)
    assert not h.is_fp8_weight(mods["proj_out"].weight) and not h.is_fp8_weight(mods["x_embedder"].weight)
    n_img, n_txt = 23, 5
    x = torch.randn(1, n_img, 8, dtype=torch.bfloat16)
    c = torch.randn(1, n_txt, 16, dtype=torch.bfloat16)
    ids = torch.zeros(1, n_img, 4)
    ids[..., 1] = torch.arange(n_img)
    kw = dict(hidden_states=x, encoder_hidden_states=c, timestep=torch.tensor([0.5], dtype=torch.bfloat16),
              img_ids=ids, txt_ids=torch.zeros(1, n_txt, 4), return_dict=False)
    with torch.no_grad():
        want, got = ref(**kw)[0].float(), fp8(**kw)[0].float()
    assert torch.isfinite(got).all()
    assert ((got - want).norm() / want.norm()).item() < 0.05          # measured 0.4% on this tiny model


def test_peft_never_sees_torchao_before_the_lora_is_fused(tmp_path, monkeypatch):
    """PEFT raises once it can see a torchao older than it wants, so torchao stays out of sight until FP8 needs it,
    and PEFT's answer is recorded before the folder is added."""
    calls = []
    peft_utils = types.ModuleType("peft.import_utils")
    peft_utils.is_torchao_available = lambda: calls.append(list(sys.path)) or False
    monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))
    monkeypatch.setitem(sys.modules, "peft.import_utils", peft_utils)
    folder = tmp_path / "azyah-torchao"
    (folder / "torchao" / "quantization").mkdir(parents=True)
    (folder / "torchao" / "__init__.py").write_text("")
    (folder / "torchao" / "quantization" / "__init__.py").write_text(
        "Float8DynamicActivationFloat8WeightConfig = object\nquantize_ = print\nclass PerTensor: pass\n")
    for name in [m for m in sys.modules if m == "torchao" or m.startswith("torchao.")]:
        monkeypatch.delitem(sys.modules, name)
    h = _helper()
    monkeypatch.setattr(h, "TORCHAO_PATH", str(folder))
    monkeypatch.setattr(sys, "path", [p for p in sys.path])
    cfg, per_tensor, quantize = h.import_torchao()
    assert calls and str(folder) not in calls[0]              # PEFT asked first, while torchao was out of sight
    assert str(folder) in sys.path and cfg is object and quantize is print and per_tensor.__name__ == "PerTensor"

"""fluxrt-fuse-lora: the LoRA is folded into the weights before INT8, with the pre-quantized model's scheme."""

import importlib.util
import json
import sys
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
    # Dockerfile.livecam applies fluxrt/patches/*.patch in name order: anchor first, then this one.
    names = sorted(p.name for p in (ROOT / "fluxrt/patches").glob("*.patch"))
    assert names == ["fluxrt-anchor-reference.patch", "fluxrt-fuse-lora.patch"]
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

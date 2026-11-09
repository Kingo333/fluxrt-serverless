"""merge_lora: the merged file equals the blend exactly, and its settings record fits the new rank."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))


def _lora(torch, rank, seed, layers=("transformer.transformer_blocks.0.attn.to_q", "transformer.single_transformer_blocks.3.attn.to_out")):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name in layers:
        sd[f"{name}.lora_A.weight"] = torch.randn(rank, 16, generator=g)
        sd[f"{name}.lora_B.weight"] = torch.randn(24, rank, generator=g)
    return sd


def test_merge_equals_the_weighted_blend_exactly():
    torch = pytest.importorskip("torch")
    from fluxrt_lora.merge_lora import merge_state_dicts, rank_of
    a, b = _lora(torch, 32, 1), _lora(torch, 32, 2)
    m = merge_state_dicts([a, b], [0.5, 0.5], scale=0.65)
    assert rank_of(m) == 64
    for name in ("transformer.transformer_blocks.0.attn.to_q", "transformer.single_transformer_blocks.3.attn.to_out"):
        want = 0.325 * a[f"{name}.lora_B.weight"] @ a[f"{name}.lora_A.weight"] + 0.325 * b[f"{name}.lora_B.weight"] @ b[f"{name}.lora_A.weight"]
        got = m[f"{name}.lora_B.weight"] @ m[f"{name}.lora_A.weight"]
        assert torch.allclose(got, want, atol=1e-5)
        assert m[f"{name}.lora_A.weight"].shape == (64, 16) and m[f"{name}.lora_B.weight"].shape == (24, 64)
    # unequal weights and a bf16 file work too, and files with different layers are refused
    m2 = merge_state_dicts([{k: v.to(torch.bfloat16) for k, v in a.items()}, b], [0.3, 0.7], scale=1.0)
    assert m2[next(k for k in m2 if ".lora_A." in k)].dtype == torch.bfloat16
    with pytest.raises(ValueError):
        merge_state_dicts([a, _lora(torch, 32, 3, layers=("transformer.other.to_k",))], [0.5, 0.5])


def test_metadata_is_rewritten_for_the_new_rank():
    from fluxrt_lora.merge_lora import merged_metadata
    meta = {"transformer.r": 32, "transformer.lora_alpha": 32, "transformer.rank_pattern": {}, "transformer.alpha_pattern": {},
            "transformer.target_modules": ["to_q", "to_k"], "transformer.init_lora_weights": "gaussian"}
    out = merged_metadata([meta, meta], 64)
    assert out["transformer.r"] == 64 and out["transformer.lora_alpha"] == 64      # scaling alpha/r stays 1.0
    assert out["transformer.rank_pattern"] == {} and out["transformer.target_modules"] == ["to_q", "to_k"]
    assert merged_metadata([None, None], 64) is None


def test_cli_writes_file_selection_and_readme(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from fluxrt_lora import merge_lora
    meta = {"format": "pt", merge_lora.METADATA_KEY: json.dumps({"transformer.r": 32, "transformer.lora_alpha": 32})}
    d3, d4 = tmp_path / "fx3", tmp_path / "fx4"
    d3.mkdir(), d4.mkdir()
    st.save_file(_lora(torch, 32, 1), str(d3 / "pytorch_lora_weights.safetensors"), metadata=meta)
    st.save_file(_lora(torch, 32, 2), str(d4 / "pytorch_lora_weights.safetensors"), metadata=meta)
    (d3 / "selection.json").write_text(json.dumps({"commercial_use_cleared": False, "trained_with_vivid": True}))
    r = merge_lora.main(["--inputs", str(d3 / "pytorch_lora_weights.safetensors"), str(d4 / "pytorch_lora_weights.safetensors"),
                         "--weights", "0.5", "0.5", "--scale", "0.65", "--out", str(tmp_path / "out"), "--selection-from", str(d3)])
    assert r["rank"] == 64 and r["metadata_rewritten"]
    out = tmp_path / "out"
    assert merge_lora.read_metadata(out / "pytorch_lora_weights.safetensors") == {"transformer.r": 64, "transformer.lora_alpha": 64}
    sel = json.loads((out / "selection.json").read_text())
    assert sel["commercial_use_cleared"] is False and sel["strength_baked_in"] == 0.65 and len(sel["merged_from"]) == 2
    assert "sha256" in (out / "README.txt").read_text()

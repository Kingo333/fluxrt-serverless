"""fxreal8 on a real (tiny) FLUX.2 transformer: favourite #2 merged into the frozen weights, the add-on, the
reference switch and the export as one file. Needs torch, peft and diffusers (skipped without them, e.g. in CI):
run it where the trainer's packages are installed. The helpers are taken from the patch itself."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
diffusers = pytest.importorskip("diffusers")

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "training/patches/flux2-klein-fx8.patch"
TARGETS = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj",
           *[f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)]]   # the trainer's default layers


def fx8():
    """The helper functions exactly as the patch adds them."""
    lines = PATCH.read_text().splitlines()
    src = "\n".join(l[1:] for l in lines if l[:1] in (" ", "+") and not l.startswith("+++"))
    code = src[src.index("_AZ_FX8 = {"):src.index("def collate_fn(examples):")]
    ns = {"torch": torch, "os": os, "json": json, "math": math}
    exec(compile(code, "fx8", "exec"), ns)
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if k.startswith("_az")}, state=ns["_AZ_FX8"])


def tiny(seed=0):
    from diffusers import Flux2Transformer2DModel
    torch.manual_seed(seed)
    m = Flux2Transformer2DModel(patch_size=1, in_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=16,
                                num_attention_heads=2, joint_attention_dim=32, timestep_guidance_channels=16,
                                mlp_ratio=2.0, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False)
    return m.eval()


def inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    b, n, t = 2, 6, 3
    return dict(hidden_states=torch.randn(b, n, 8, generator=g), timestep=torch.tensor([0.9, 1.0]), guidance=None,
                encoder_hidden_states=torch.randn(b, t, 32, generator=g),
                txt_ids=torch.zeros(b, t, 4), img_ids=torch.arange(n).float().view(1, n, 1).expand(b, n, 4).contiguous(),
                return_dict=False)


def lora_config(r):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=r, lora_dropout=0.0, init_lora_weights="gaussian", target_modules=TARGETS)


def save_lora(model, out_dir, r):
    from diffusers import Flux2KleinPipeline
    from peft.utils import get_peft_model_state_dict
    meta = {"r": r, "lora_alpha": r, "target_modules": TARGETS}
    Flux2KleinPipeline.save_lora_weights(out_dir, transformer_lora_layers=get_peft_model_state_dict(model),
                                         transformer_lora_adapter_metadata=meta)
    return Path(out_dir) / "pytorch_lora_weights.safetensors"


def favourite_2(tmp_path, base_state):
    """A made-up favourite #2: a rank-4 LoRA with non-zero weights, saved the way the trainer saves."""
    m = tiny()
    m.load_state_dict(base_state)
    m.add_adapter(lora_config(4))
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    with torch.no_grad():
        y = m(**inputs())[0]
    return save_lora(m, tmp_path / "fav2", 4), y


def frozen_with_addon(f, fav_file, base_state):
    """What the patched trainer does: merge the file into the frozen weights, then add the rank-2 add-on."""
    from diffusers import Flux2KleinPipeline
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft.utils import get_peft_model_state_dict
    m = tiny()
    m.load_state_dict(base_state)
    m.requires_grad_(False)
    sd = Flux2KleinPipeline.lora_state_dict(str(fav_file.parent))
    sd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
    layers = f._azyah_lora_layers(convert_unet_state_dict_to_peft(sd))
    n, rel = f._azyah_merge_into(m, layers, f._azyah_lora_file_scale(str(fav_file.parent)))
    m.add_adapter(lora_config(2))
    new = {k[: -len(".lora_A.weight")] for k in get_peft_model_state_dict(m) if k.endswith(".lora_A.weight")}
    return m, layers, n, rel, new


def test_favourite_2_merged_into_the_frozen_weights_gives_its_own_output(tmp_path):
    f = fx8()
    base = tiny().state_dict()
    fav_file, y_fav = favourite_2(tmp_path, base)
    m, layers, n, rel, new = frozen_with_addon(f, fav_file, base)
    assert n == len(layers) > 0 and rel < 1e-5                     # float32 here: the change is kept exactly
    assert new == set(layers)                                      # the add-on sits on the same layers
    with torch.no_grad():
        y_start = m(**inputs())[0]
        m.disable_adapters()
        y_frozen = m(**inputs())[0]
        m.enable_adapters()
    assert torch.allclose(y_frozen, y_fav, atol=1e-5)              # frozen model = favourite #2
    assert torch.equal(y_start, y_frozen)                          # a new add-on starts as an exact zero


def test_the_reference_switch_and_the_gradients(tmp_path):
    f = fx8()
    base = tiny().state_dict()
    fav_file, y_fav = favourite_2(tmp_path, base)
    m, *_ = frozen_with_addon(f, fav_file, base)
    with torch.no_grad():                                          # as if the add-on had learned something
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.3)
    m.train()
    with torch.no_grad():
        m.disable_adapters()
        try:
            ref = m(**inputs())[0]
        finally:
            m.enable_adapters()
    out = m(**inputs())[0]
    assert torch.allclose(ref, y_fav, atol=1e-5) and not torch.allclose(out, ref, atol=1e-4)
    loss = ((out - ref) ** 2).mean()
    loss.backward()
    trained = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert trained and all("lora_" in n for n in trained)          # only the add-on learns
    assert all(not p.requires_grad for n, p in m.named_parameters() if "lora_" not in n)
    assert all(p.requires_grad for n, p in m.named_parameters() if "lora_" in n)   # switched back on


def test_a_save_merged_with_favourite_2_is_one_file_that_loads_like_any_lora(tmp_path):
    f = fx8()
    base = tiny().state_dict()
    fav_file, _ = favourite_2(tmp_path, base)
    m, *_ = frozen_with_addon(f, fav_file, base)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.3)
        y_trained = m(**inputs())[0]
    addon_file = save_lora(m, tmp_path / "addon", 2)
    out = tmp_path / "merged"
    r = subprocess.run([sys.executable, "-m", "fluxrt_lora.merge_lora", "--inputs", str(fav_file), str(addon_file),
                        "--weights", "1.0", "1.0", "--scale", "1.0", "--out", str(out)],
                       cwd=ROOT / "training", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    x = tiny()
    x.load_state_dict(base)
    x.load_lora_adapter(str(out / "pytorch_lora_weights.safetensors"), prefix="transformer", adapter_name="merged")
    with torch.no_grad():
        y = x(**inputs())[0]
    assert torch.allclose(y, y_trained, atol=1e-4)                 # favourite #2 + the add-on, as the trainer saw it


def test_rounding_to_bf16_is_measured():
    f = fx8()
    lin = torch.nn.Linear(64, 64).to(torch.bfloat16)
    model = torch.nn.Module()
    model.add_module("layer", lin)
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(4, 64, generator=g) * 0.05, torch.randn(64, 4, generator=g) * 0.05
    n, rel = f._azyah_merge_into(model, {"layer": (a, b)})
    assert n == 1 and 0 < rel < 0.5                                # some of the change is lost to bf16, and it says so


def test_the_steadiness_term():
    f = fx8()
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(2, 8, 6, 6, generator=g)
    x0[1] = x0[0]
    x0[1, :, :3, :] += 1.0                          # the person moved in the top half; the bottom half is still
    pairs = [(0, 1)]
    assert f._azyah_steady_term(x0.clone(), x0, pairs).item() == pytest.approx(0.0, abs=1e-7)
    still_flicker = x0.clone(); still_flicker[1, :, 3:, :] += 0.2       # the picture changes where nothing moved
    moving_change = x0.clone(); moving_change[1, :, :3, :] += 0.2       # the same change where the person moved
    a = f._azyah_steady_term(still_flicker, x0, pairs).item()
    b = f._azyah_steady_term(moving_change, x0, pairs).item()
    assert a > 0 and a > 5 * b                      # flicker where nothing moved costs far more
    # it is not turned down at high noise: the term does not even see the noise level
    x0h = x0.clone().requires_grad_(True)
    f._azyah_steady_term(x0h, x0.detach(), pairs).backward()
    assert x0h.grad is not None


def test_the_safety_check(tmp_path, capsys):
    f = fx8()
    args = types.SimpleNamespace(safety_steps=3, ref_weight=1.0, steady_ratio=0.25, steady_weight=0.0)
    wf = tmp_path / "w.json"
    # a non-finite term stops it at once
    f.state.update(seen=1, steady_on=0, data=0.0, steady=0.0, first_ref=0.0, first_data=1.0, weight=0.0, done=False)
    with pytest.raises(SystemExit) as e:
        f._azyah_fx8_check(args, {"steady": float("nan")}, torch.tensor(1.0), True, str(wf))
    assert e.value.code == 3 and "FX8_SAFETY_FAILED: not finite at step 1: steady" in capsys.readouterr().out
    # an add-on that does not start at favourite #2 stops it after the safety steps
    f.state.update(seen=3, steady_on=3, data=3.0, steady=0.3, first_ref=0.5, first_data=1.0, weight=0.0, done=False)
    with pytest.raises(SystemExit):
        f._azyah_fx8_check(args, {"ref": 0.1}, torch.tensor(1.0), True, str(wf))
    assert "does not start at the frozen model" in capsys.readouterr().out
    # no frame pairs: the steadiness term never ran
    f.state.update(seen=3, steady_on=0, data=0.0, steady=0.0, first_ref=0.0, first_data=1.0, weight=0.0, done=False)
    with pytest.raises(SystemExit):
        f._azyah_fx8_check(args, {}, torch.tensor(1.0), True, str(wf))
    assert "no frame pairs" in capsys.readouterr().out
    # all well: the steadiness weight is set to a quarter of the data term and written down for a restart
    f.state.update(seen=3, steady_on=3, data=3.0, steady=0.3, first_ref=0.0, first_data=1.0, weight=0.0, done=False)
    f._azyah_fx8_check(args, {"ref": 0.0, "steady": 0.1}, torch.tensor(1.0), True, str(wf))
    out = capsys.readouterr().out
    assert f.state["done"] and f.state["weight"] == pytest.approx(0.25 * 1.0 / 0.1)
    assert "FX8_SAFETY_OK: start = the frozen model" in out and "STEADY_WEIGHT 2.5" in out
    assert json.loads(wf.read_text())["steady_weight"] == pytest.approx(2.5)
    # a restart (not a fresh start) does not ask for the zero start again
    f.state.update(seen=3, steady_on=3, data=3.0, steady=0.3, first_ref=0.4, first_data=1.0, weight=2.5, done=False)
    f._azyah_fx8_check(args, {"ref": 0.4}, torch.tensor(1.0), False, str(wf))
    assert "FX8_SAFETY_OK" in capsys.readouterr().out

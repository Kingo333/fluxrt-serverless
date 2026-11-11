"""fxreal5: dress frames where the dress moves, dress-first data, init from favourite #2, dress eval set."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
TRAINER = "train_dreambooth_lora_flux2_klein_img2img.py"
PATCHES = ("flux2-klein-garment.patch", "flux2-klein-timeshift.patch", "flux2-klein-maskloss.patch",
           "flux2-klein-fx3.patch", "flux2-klein-initlora.patch")


def _mask(x0, x1, y0=10, y1=40, w=40, h=60):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_change_is_one_minus_overlap():
    from data_tools.vivid_motion import change
    a = _mask(0, 10)
    assert change(a, a) == 0.0
    assert change(a, _mask(20, 30)) == 1.0
    assert abs(change(a, _mask(5, 15)) - (1 - 5 / 15)) < 1e-9
    assert change(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 0.0


def test_pair_is_the_biggest_change_within_the_gap_and_singles_are_far_apart():
    from data_tools.vivid_motion import pick_pair, pick_singles
    # the dress shifts a little each frame, except a big swing between frames 10 and 14
    masks = {t: _mask(t // 2, t // 2 + 10) for t in range(0, 30, 2)}
    masks[14] = _mask(25, 35)
    t, t2, c = pick_pair(masks, 3, 8)
    assert 3 <= t2 - t <= 8 and 14 in (t, t2) and c > 0.9
    assert pick_pair(masks, 40, 50) is None                       # no frames that far apart
    singles = pick_singles(masks, [t, t2], 2)
    assert len(singles) == 2 and not set(singles) & {t, t2} and len(set(singles)) == 2
    assert pick_singles({0: _mask(0, 5)}, [0], 3) == []          # nothing left to pick


def test_motion_items_reads_pairs_singles_and_description(tmp_path):
    from data_tools.vivid_motion import motion_items
    d = tmp_path / "vivid_train_000001"
    d.mkdir()
    for f in ("a_target.png", "a_mask.png", "b_target.png", "b_mask.png", "ref.jpg", "s1_target.png", "s1_mask.png"):
        (d / f).write_bytes(b"x")
    (d / "meta.json").write_text(json.dumps({"category": "dress", "split": "train", "garment_description": "satin abaya"}))
    bad = tmp_path / "vivid_test_000002"
    bad.mkdir()
    (bad / "meta.json").write_text(json.dumps({"split": "test"}))
    items = list(motion_items(tmp_path))
    assert len(items) == 1
    clip, cat, ref, desc, pair, singles = items[0]
    assert (clip, cat, desc, len(pair), len(singles)) == ("vivid_train_000001", "dress", "satin abaya", 2, 1)


def test_outfit_area_reaches_the_bottom_and_captions_use_the_description(tmp_path):
    from PIL import Image
    from fluxrt_lora.build_real import caption_for, meta_description, outfit_area
    m = np.zeros((100, 60), dtype=np.uint8)
    m[20:60, 20:40] = 255                                          # a knee-length dress
    whole, band = outfit_area(Image.fromarray(m, "L"))
    w, b = np.asarray(whole) > 127, np.asarray(band) > 127
    assert b[99, 30] and b[60, 30] and not b[59, 30] and w[30, 30]   # legs below the hem to the bottom
    assert "satin abaya with a cape" in caption_for("dress", "satin abaya with a cape")
    assert caption_for("dress", None) == caption_for("dress", "")
    (tmp_path / "meta.json").write_text(json.dumps({"garment_description": "velvet kaftan"}))
    assert meta_description(tmp_path) == "velvet kaftan" and meta_description(tmp_path / "nope") is None


def _patched_trainer(tmp_path) -> str:
    local = Path("/home/user/repos/diffusers")
    if not (local / "examples" / "dreambooth" / TRAINER).exists():
        pytest.skip("diffusers checkout not present")
    r = subprocess.run(["git", "-C", str(local), "show", f"HEAD:examples/dreambooth/{TRAINER}"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("pinned diffusers file not available")
    repo = tmp_path / "d"
    (repo / "examples" / "dreambooth").mkdir(parents=True)
    (repo / "examples" / "dreambooth" / TRAINER).write_text(r.stdout)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name in PATCHES:
        r = subprocess.run(["git", "-C", str(repo), "apply", str(ROOT / "training" / "patches" / name)], capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr}"
    return (repo / "examples" / "dreambooth" / TRAINER).read_text()


def test_init_lora_patch_applies_last_and_loads_weights_exactly(tmp_path):
    src = _patched_trainer(tmp_path)
    compile(src, TRAINER, "exec")
    assert '"--init_lora_weights"' in src and "set_peft_model_state_dict(transformer, _init_sd" in src
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    diffusers = pytest.importorskip("diffusers")
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import save_file
    block = src[src.index("    if args.init_lora_weights:"): src.index('        logger.info(f"LoRA initialised from')]
    block = "\n".join(line[4:] for line in block.splitlines())
    kw = dict(patch_size=1, in_channels=8, out_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=8,
              num_attention_heads=2, joint_attention_dim=16, timestep_guidance_channels=16, mlp_ratio=2.0,
              axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False)
    targets = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj", "single_transformer_blocks.0.attn.to_out"]

    def fresh(rank):
        torch.manual_seed(0)
        try:
            m = diffusers.Flux2Transformer2DModel(**kw)
        except TypeError as e:
            pytest.skip(f"tiny Flux2 config not constructible here: {e}")
        m.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=targets))
        return m

    src_model = fresh(4)
    with torch.no_grad():
        for n, p in src_model.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn_like(p) * 0.1)
    folder = tmp_path / "fav2"
    folder.mkdir()
    save_file({f"transformer.{k}": v.contiguous() for k, v in get_peft_model_state_dict(src_model).items()},
              str(folder / "pytorch_lora_weights.safetensors"), metadata={"format": "pt"})

    class Args:
        init_lora_weights = str(folder)

    class Log:
        def info(self, *_): pass

    def run(transformer):
        ns = {"args": Args, "transformer": transformer, "Flux2KleinPipeline": diffusers.Flux2KleinPipeline,
              "convert_unet_state_dict_to_peft": convert_unet_state_dict_to_peft,
              "set_peft_model_state_dict": set_peft_model_state_dict, "logger": Log()}
        exec(compile(block + "\n", "init_block", "exec"), ns)

    model = fresh(4)
    run(model)
    a, b = get_peft_model_state_dict(src_model), get_peft_model_state_dict(model)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)   # starts exactly at favourite #2
    with pytest.raises(Exception):                                              # wrong rank: fails loudly
        run(fresh(8))


def test_dress_eval_set_is_dress_only_and_auto_described(tmp_path, capsys):
    from eval import step_a
    vivid = tmp_path / "data/staging/vivid"
    for i, cat in enumerate(["dress"] * 8 + ["upper"] * 4):
        d = vivid / f"vivid_test_{i:06d}"
        d.mkdir(parents=True)
        (d / "target.mp4").write_bytes(b"v")
        (d / "ref.jpg").write_bytes(f"g{i}".encode())
        meta = {"split": "test", "category": cat}
        if i % 2 == 0:
            meta["garment_description"] = f"long satin dress number {i}"
        (d / "meta.json").write_text(json.dumps(meta))
    items = step_a.pick_set(tmp_path, "dress10", n=4, categories=["dress"], auto_describe=True)
    assert len(items) == 4 and all(it["category"] == "dresses" for it in items)
    assert not any(it.get("needs_description") for it in items)
    for it in items:
        if it["description"]:
            assert it["description"] in it["prompt"] and it["description_by"] == "vivid"
        else:
            assert it["description_by"] == "none"
    assert not (tmp_path / "eval/fxa/dress10_descriptions.json").exists()
    assert "PROMPT:" in capsys.readouterr().out


def test_fx5_script_continues_favourite2_with_dresses_first():
    sh = (ROOT / "training/fluxrt_lora/fx5.sh").read_text()
    assert subprocess.run(["bash", "-n", str(ROOT / "training/fluxrt_lora/fx5.sh")]).returncode == 0
    for s in ("AZ_INIT_LORA=$INIT", "AZ_LORA_RANK=64", "AZ_LR=${AZ_LR:-5e-5}", "AZ_REF_SIZE=$REF_SIZE",
              "data_tools.vivid_motion", "--motion \"$S/vivid_dress_motion\"", "--dress-outfit-frac",
              "--other-live-max", "AZ_MASK_WEIGHT=${AZ_MASK_WEIGHT:-3.0}", "AZ_DETAIL_WEIGHT=${AZ_DETAIL_WEIGHT:-0.05}",
              "--categories dress --auto-describe", "--ref-size $REF_SIZE", "fav2_$TAG=favourite #2 (today)",
              "eval.preserve_check", "outputs/fluxrt_lora/blend05s065"):
        assert s in sh, s
    rt = (ROOT / "training/fluxrt_lora/run_train.sh").read_text()
    assert "flux2-klein-initlora.patch" in rt and '--init_lora_weights="$AZ_INIT_LORA"' in rt
    assert rt.index("flux2-klein-fx3.patch") < rt.index("flux2-klein-initlora.patch")

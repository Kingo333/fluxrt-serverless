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


# ---------------------------------------------------------------- owner additions (2026-11-11)
PATCHES_ALL = PATCHES + ("flux2-klein-fx5.patch",)


def test_colour_and_texture_terms_patch_applies_last_and_behaves(tmp_path):
    import importlib
    local = Path("/home/user/repos/diffusers")
    if not (local / "examples" / "dreambooth" / TRAINER).exists():
        pytest.skip("diffusers checkout not present")
    r = subprocess.run(["git", "-C", str(local), "show", f"HEAD:examples/dreambooth/{TRAINER}"], capture_output=True, text=True)
    repo = tmp_path / "d"
    (repo / "examples" / "dreambooth").mkdir(parents=True)
    (repo / "examples" / "dreambooth" / TRAINER).write_text(r.stdout)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name in PATCHES_ALL:
        rr = subprocess.run(["git", "-C", str(repo), "apply", str(ROOT / "training" / "patches" / name)], capture_output=True, text=True)
        assert rr.returncode == 0, f"{name}: {rr.stderr}"
    src = (repo / "examples" / "dreambooth" / TRAINER).read_text()
    compile(src, TRAINER, "exec")
    for s in ('"--colour_weight"', '"--texture_weight"', '_az_logs["colour"]', '_az_logs["texture"]'):
        assert s in src, s
    torch = pytest.importorskip("torch")
    start = src.index("                if (args.colour_weight > 0 or args.texture_weight > 0) and loss_mask is not None:")
    end = src.index('                        _az_logs["texture"] = _texture.detach().item()') + len('                        _az_logs["texture"] = _texture.detach().item()')
    block = "\n".join(line[16:] for line in src[start:end].splitlines())

    class Args:
        colour_weight, texture_weight = 1.0, 1.0

    def run(pred_shift=0.0, pred_scale=1.0):
        g = torch.Generator().manual_seed(0)
        x0 = torch.randn(2, 8, 6, 6, generator=g)
        noise = torch.randn(2, 8, 6, 6, generator=g)
        sig = torch.tensor([0.3, 0.6]).reshape(2, 1, 1, 1)
        noisy = (1 - sig) * x0 + sig * noise
        m = torch.zeros(2, 1, 6, 6)
        m[..., 1:5, 1:5] = 1.0
        mu = (x0 * m).sum((-2, -1), keepdim=True) / m.sum((-2, -1), keepdim=True)   # the garment's own average
        x0_hat = (x0 - mu) * pred_scale + mu + pred_shift
        pred = (noisy - x0_hat) / sig                       # so that noisy - sig * pred == x0_hat
        ns = {"args": Args, "sigmas": sig, "noisy_model_input": noisy, "model_pred": pred, "model_input": x0,
              "m": m, "loss_mask": m, "loss": torch.tensor(0.0), "_az_logs": {}, "torch": torch}
        exec(compile(block + "\n", "fx5_block", "exec"), ns)
        return ns["_az_logs"]

    perfect = run()
    assert perfect["colour"] < 1e-5 and perfect["texture"] < 1e-5          # right answer: no penalty
    off_colour = run(pred_shift=0.5)
    assert off_colour["colour"] > 0.4 and off_colour["texture"] < 1e-5      # colour off, contrast right
    flat = run(pred_scale=0.5)
    assert flat["texture"] > 0.1 and flat["colour"] < 1e-5                  # dull fabric: contrast term fires


def test_dress_share_builds_dresses_first_then_fills_tops_to_the_share(tmp_path):
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from fluxrt_lora import build_real

    def jb(img):
        b = io.BytesIO(); img.save(b, format="JPEG"); return {"bytes": b.getvalue(), "path": "x.jpg"}
    t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    rows = {"cloth": [], "image": [], "agnostic": []}
    for i in range(40):
        person = Image.new("RGB", (768, 1024), (200, 30 + i, 40))
        agn = person.copy(); agn.paste((128, 128, 128), (200, 300, 560, 700))
        rows["cloth"].append(jb(Image.new("RGB", (300, 400), (i * 5, 90, 200))))
        rows["image"].append(jb(person)); rows["agnostic"].append(jb(agn))
    (tmp_path / "viton").mkdir()
    pq.write_table(pa.table({k: pa.array(v, type=t) for k, v in rows.items()}), tmp_path / "viton" / "a.parquet")
    motion = tmp_path / "motion"
    for c in range(5):                                   # 5 dress clips: a pair + 2 singles each = 20 dress rows
        d = motion / f"vivid_train_{c:06d}"
        d.mkdir(parents=True)
        for tag in ("a", "b", "s1", "s2"):
            Image.new("RGB", (432, 768), (90 + c, 60, 70)).save(d / f"{tag}_target.png")
            mk = Image.new("L", (432, 768), 0); mk.paste(255, (120, 200, 320, 700)); mk.save(d / f"{tag}_mask.png")
        Image.new("RGB", (100, 140), (10 * c, 20, 30)).save(d / "ref.jpg")
        (d / "meta.json").write_text(json.dumps({"category": "dress", "split": "train", "garment_description": f"dress {c}"}))

    class FakeRedress:
        def __init__(self, *a, **k): pass
        def __call__(self, person, garment, seed, category="tops", prompt=None):
            return Image.new("RGB", (432, 768), (0, 255, 0))

    class FakeDino:
        def __call__(self, a, b): return 0.5

    old = build_real.Redresser, build_real.DinoSim
    build_real.Redresser, build_real.DinoSim = FakeRedress, FakeDino
    try:
        s = build_real.main(["--out", str(tmp_path / "data"), "--viton-dir", str(tmp_path / "viton"), "--klein", "x",
                             "--test-out", str(tmp_path / "pairs"), "--viton-n", "35", "--holdout", "2", "--test-items", "0",
                             "--motion", str(motion), "--dress-outfit-frac", "0.5", "--dress-share", "0.57",
                             "--nochange-frac", "0.2"])
    finally:
        build_real.Redresser, build_real.DinoSim = old
    assert s["dress_rows"] == 20 and s["motion_pairs_kept"] == 5 and s["motion_singles_kept"] == 10
    assert s["other_quota"] == round(20 * 0.43 / 0.57) == s["other_rows"]            # tops filled exactly to 43%
    assert abs(s["dress_share"] - 0.57) < 0.02 and s["samples"] == 20 + s["other_quota"]
    assert s["described"] == 20                                                       # captions carry the dress text
    keys = []
    for f in sorted((tmp_path / "data" / "data").glob("*.parquet")):
        keys += pq.read_table(f, columns=["key"]).column("key").to_pylist()
    assert keys[0].startswith("vivid_motion_") and keys[-1].startswith("viton_hd_")   # dresses first, then tops


def test_garment_report_scores_the_five_priorities():
    from eval import garment_report as gr
    assert np.allclose(gr.srgb_to_lab(np.array([255, 255, 255])), [100, 0, 0], atol=0.05)
    assert abs(gr.srgb_to_lab(np.array([255, 0, 0]))[0] - 53.24) < 0.1
    H, W, T = 64, 40, 8
    ref = np.full((80, 60, 3), 255, np.uint8)
    ref[10:70, 15:45] = (120, 20, 120)                                               # a purple garment on white
    ref[30:40, 15:45] = (240, 240, 240)                                              # a light trim band
    ref_mask = gr.ref_garment_mask(ref, None)
    assert ref_mask[50, 30] and not ref_mask[2, 2]
    assert ref_mask[35, 30]                                                          # the light trim is garment, not background
    big = np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(ref).resize((600, 800), 0))
    assert gr.ref_garment_mask(big, None)[350, 300] and not gr.ref_garment_mask(big, None)[10, 10]
    # real product photos: an off-white backdrop with a black letterbox bar, a grey backdrop behind off-white
    # trousers, and a soft gradient backdrop: the backdrop goes, the garment stays
    yy, xx = np.mgrid[0:600, 0:360]
    dress = np.zeros((600, 360, 3), np.uint8)
    dress[:] = (242, 241, 238)
    dress[560:] = 0
    dress[(((yy - 300) / 220.0) ** 2 + ((xx - 180) / 90.0) ** 2) < 1] = (205, 25, 45)
    m = gr.ref_garment_mask(dress, None)
    assert m[300, 180] and not m[20, 20] and not m[590, 180] and 0.15 < m.mean() < 0.35
    assert gr.delta_e(gr.median_lab(dress, m), gr.srgb_to_lab(np.array([205, 25, 45]))) < 1.0
    trousers = np.full((600, 360, 3), 200, np.uint8)
    trousers[100:550, 120:240] = (240, 238, 230)
    m = gr.ref_garment_mask(trousers, None)
    assert m[300, 180] and not m[50, 50] and abs(m.mean() - 450 * 120 / (600 * 360)) < 0.02
    grad = (215 + 30 * (1 - np.hypot((yy - 300) / 300.0, (xx - 180) / 180.0) / np.sqrt(2)))[..., None].repeat(3, -1)
    grad = grad.astype(np.uint8)
    grad[150:450, 110:250] = (30, 40, 90)
    m = gr.ref_garment_mask(grad, None)
    assert m[300, 180] and not m[300, 40] and not m[100, 180] and abs(m.mean() - 300 * 140 / (600 * 360)) < 0.02
    cams, cats_cam, good, cats_good, bad, cats_bad = [], [], [], [], [], []
    rng = np.random.default_rng(0)
    for t in range(T):
        x = 8 + t                                                                    # the person walks right
        cam = np.full((H, W, 3), 200, np.uint8)
        cc = np.zeros((H, W), np.uint8)
        cc[5:60, x:x + 16] = gr.CLOTHES
        cc[2:6, x + 4:x + 12] = gr.FACE
        cams.append(cam); cats_cam.append(cc)
        g = cam.copy(); g[5:60, x:x + 16] = (120, 20, 120); g[25:32, x:x + 16] = (240, 240, 240)
        good.append(g); cats_good.append(cc.copy())
        jx = x + (5 if t % 2 else -5)                                                 # jumps around, wrong colour
        b = cam.copy(); b[5:60, jx:jx + 16] = (60 + 20 * (t % 2), 60, 160)
        cb = np.zeros((H, W), np.uint8); cb[5:60, max(0, jx):jx + 16] = gr.CLOTHES; cb[2:6, x + 4:x + 12] = gr.FACE
        bad.append(b); cats_bad.append(cb)
    rg = gr.clip_metrics(cams, good, cats_cam, cats_good, ref, ref_mask, "dresses")
    rb = gr.clip_metrics(cams, bad, cats_cam, cats_bad, ref, ref_mask, "dresses")
    assert rg["colour_de"] < rb["colour_de"] and rg["follow_px"] < rb["follow_px"]
    assert rg["jitter"] < rb["jitter"] and rg["flicker_de"] < rb["flicker_de"] and rg["fabric"] < rb["fabric"]
    card = gr.scorecard({"clip1": {"fav2": rb, "fx5": rg}}, ["fav2", "fx5"], "fav2")
    got = {(r["priority"], r["metric"]): (r["better"], r["worse"]) for r in card}
    assert got[("COLOUR", "colour_de")] == (1, 0) and got[("MOVEMENT", "follow_px")] == (1, 0)
    assert got[("CONSISTENCY", "jitter")] == (1, 0) and got[("CONSISTENCY", "flicker_de")] == (1, 0)
    top = gr.band_mask(np.ones((10, 4), bool), np.ones((10, 4), bool), "tops")
    assert top[:5].all() and not top[6:].any()                                     # a top: trousers do not count


def test_fx5_script_runs_the_five_priority_report_and_57_percent_dresses():
    sh = (ROOT / "training/fluxrt_lora/fx5.sh").read_text()
    for s in ('--dress-share "${AZ_DRESS_SHARE:-0.57}"', "AZ_COLOUR_WEIGHT=${AZ_COLOUR_WEIGHT:-0.5}",
              "AZ_TEXTURE_WEIGHT=${AZ_TEXTURE_WEIGHT:-0.3}", "eval.garment_report seg", "eval.garment_report score",
              '--baseline "fav2_$TAG"', "eval.stability_report", "AZ_PAIR_COLUMN=pair"):
        assert s in sh, s
    rt = (ROOT / "training/fluxrt_lora/run_train.sh").read_text()
    assert rt.index("flux2-klein-initlora.patch") < rt.index("flux2-klein-fx5.patch")
    assert '--colour_weight="$AZ_COLOUR_WEIGHT"' in rt and '--texture_weight="$AZ_TEXTURE_WEIGHT"' in rt


def _person_cat(h, w, hijab=True):
    """A standing person: face, a hijab around and above it (or hair), clothes below; background elsewhere."""
    cat = np.zeros((h, w), np.uint8)
    cat[int(.27 * h):int(.95 * h), int(.3 * w):int(.7 * w)] = 4                      # clothes
    cat[int(.08 * h):int(.27 * h), int(.33 * w):int(.67 * w)] = 4 if hijab else 1   # hijab, or hair
    cat[int(.15 * h):int(.25 * h), int(.42 * w):int(.58 * w)] = 3                    # face
    return cat


def test_preserve_check_sees_a_hijab_turned_into_hair():
    from eval import preserve_check as pc
    cat_in = _person_cat(512, 288)
    head, cover = pc.head_zone(cat_in)
    assert head is not None and cover is not None and cover.sum() > 1000
    hair_in = _person_cat(512, 288, hijab=False)
    assert pc.head_zone(hair_in)[1] is None                                        # hair is not a covering
    inp = np.full((512, 288, 3), 128, np.uint8)
    kept = pc.head_stats(inp, inp.copy(), cat_in, cat_in)
    assert kept["head_lost"] == 0.0 and kept["head"] == 0.0 and kept["head_covered"] == 1.0
    out = inp.copy()
    out[cover] = (40, 30, 20)                                                       # the hijab became dark hair
    lost = pc.head_stats(inp, out, cat_in, hair_in)
    assert lost["head_lost"] > 0.5 and lost["head"] > 50
    assert pc.worse("head_lost", lost["head_lost"], kept["head_lost"]) and not pc.worse("head_lost", 0.02, 0.0)
    assert pc.head_stats(inp, inp, hair_in, hair_in)["head_lost"] is None          # no covering: not judged
    hand = np.zeros((512, 288), np.uint8)
    hand[0:500, 90:200] = 4                                                         # a coat from the top of the frame
    hand[250:280, 130:160] = 3                                                      # a hand taken for a face
    assert pc.head_zone(hand) == (None, None)
    neck = np.zeros((512, 288), np.uint8)
    neck[0:500, 90:200] = 4                                                         # a coat, head out of the frame
    neck[8:70, 130:160] = 3                                                         # chest skin taken for a face
    assert pc.head_zone(neck) == (None, None)                                       # no head in view
    tight = _person_cat(512, 288)
    tight[: int(.11 * 512)] = 0                                                     # a hijab tight on the forehead
    assert pc.head_zone(tight)[1] is not None


def test_hijab_set_checks_frames_and_cleans_the_dress_photo():
    from eval import hijab_set as hs
    assert hs.best_window([False, True, True, True, False], 3) == (1, 1.0)
    assert hs.frame_ok(_person_cat(512, 288)) and not hs.frame_ok(_person_cat(512, 288, hijab=False))
    photo = np.full((400, 300, 3), 20, np.uint8)
    cleaned = hs.clean_dress_photo(photo, _person_cat(400, 300))
    assert (cleaned[10, 150] == 255).all() and cleaned[300, 150][0] == 20           # above the chin cut, dress kept
    assert hs.clean_dress_photo(photo, np.full((400, 300), 4, np.uint8)) is None    # no person: used as sent


def test_hijab_set_keeps_only_clips_with_a_head_covering(tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    from PIL import Image
    from eval import hijab_set as hs

    class FakeSeg:
        def __init__(self, path):
            pass

        def __call__(self, img):
            arr = np.asarray(img)
            hijab = arr[..., 2].mean() > arr[..., 0].mean()                          # blue-ish clips "wear" a hijab
            return _person_cat(img.height, img.width, hijab=hijab)[..., None]

    monkeypatch.setattr("data_tools.clothes_mask.Segmenter", FakeSeg)
    own = tmp_path / "own"
    own.mkdir()
    for name, rgb in (("blue", (40, 60, 160)), ("red", (160, 60, 40))):
        with av.open(str(own / f"{name}.mp4"), "w") as c:
            s = c.add_stream("libx264", rate=25)
            s.width, s.height, s.pix_fmt = 360, 640, "yuv420p"
            for _ in range(80):
                for pkt in s.encode(av.VideoFrame.from_image(Image.new("RGB", (360, 640), rgb))):
                    c.mux(pkt)
            for pkt in s.encode():
                c.mux(pkt)
    photo = tmp_path / "abaya.jpg"
    Image.new("RGB", (300, 400), (20, 20, 20)).save(photo)
    set2 = tmp_path / "set2.json"
    set2.write_text(json.dumps([{"key": "a", "ref": str(photo), "category": "dresses", "name": "Black maxi dress",
                                 "description": "long black maxi dress"},
                                {"key": "b", "ref": str(photo), "category": "tops"}]))
    out = tmp_path / "hijab.json"
    s = hs.main([str(tmp_path), "--out", str(out), "--own", str(own), "--clips-dir", str(tmp_path / "clips"),
                 "--refs-from", str(set2), "--no-stock", "--model", "x"])
    items = json.loads(out.read_text())
    assert [it["key"] for it in items] == ["hijab1_own_blue"] and "own_red" in s["rejected"]
    it = items[0]
    assert it["category"] == "dresses" and it["name"] == "Black maxi dress" and it["case"] == "hijab"
    with av.open(it["video"]) as c:
        assert sum(1 for _ in c.decode(video=0)) == hs.WINDOW
    ref = np.asarray(Image.open(it["ref"]))
    assert s["refs"][0]["cleaned"] and (ref[10, 150] == 255).all()                 # above the chin: cut to white
    assert ref[300, 150][0] < 60                                                   # the dress is kept


def test_fx5_script_runs_the_hijab_set_at_the_live_photo_size():
    sh = (ROOT / "training/fluxrt_lora/fx5.sh").read_text()
    assert 'REF_SIZE="${AZ_REF_SIZE:-288x512}"' in sh
    for s in ("eval.hijab_set", '--own "$AZ_ROOT/eval/own_clips"', "dress10 set2 cases5 hijab", "HIJAB "):
        assert s in sh, s

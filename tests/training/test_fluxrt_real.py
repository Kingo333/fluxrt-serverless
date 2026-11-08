"""fxreal1 (2026-11-08): real-answer dataset helpers, the compact prompt twin, the noise-shift patch
and the run script settings (Base model, shift, 288x512 garment)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from data_tools.fluxrt_prompt import compact_prompt  # noqa: E402
from fluxrt_lora import build_real  # noqa: E402


def test_compact_prompt_matches_the_app():
    assert compact_prompt("upper") == (
        "Virtual try-on. The person in the camera frame wears the item from the reference image. "
        "It replaces the current top; the pants or skirt stay as they are. "
        "Keep its exact colour, print, pattern, logo, fabric and fit. "
        "Keep the person's face, hair, body, pose and the background unchanged.")
    p = compact_prompt("dresses", "satin maxi dress in zebra print", "Long sleeve satin.")
    assert "wears the satin maxi dress in zebra print from the reference image: Long sleeve satin." in p
    assert "at its own length" in p and ", with full-length sleeves." in p
    assert "They replace the current shoes, on the feet." in compact_prompt(None, "Nike Air Max sneakers")


def test_portrait_box_is_9_16_and_centred():
    x0, y0, x1, y1 = build_real.portrait_box(768, 1024)
    assert (x1 - x0, y1 - y0) == (576, 1024) and x0 == 96
    x0, y0, x1, y1 = build_real.portrait_box(400, 1000)
    assert x1 - x0 == 400 and abs((x1 - x0) / (y1 - y0) - 9 / 16) < 0.01


def test_agnostic_mask_paste_back_and_change_check():
    real = Image.new("RGB", (64, 64), (200, 40, 40))
    agn = real.copy()
    agn.paste((128, 128, 128), (16, 16, 48, 48))           # VITON-HD's grey garment area
    mask = build_real.agnostic_mask(real, agn)
    assert mask.getpixel((32, 32)) == 255 and mask.getpixel((2, 2)) == 0
    gen = Image.new("RGB", (64, 64), (10, 200, 10))       # the model changed EVERYTHING
    out = build_real.paste_back(gen, real, build_real.soft_mask(mask, grow=2, feather=1))
    assert out.getpixel((2, 2)) == (200, 40, 40)           # outside: the real photo, untouched
    assert out.getpixel((32, 32)) == (10, 200, 10)         # inside: the new garment
    assert build_real.changed_enough(out, real, mask)
    assert not build_real.changed_enough(real, real, mask)
    assert build_real.bbox_of(mask, pad=0) == (16, 16, 48, 48)
    assert build_real.bbox_of(Image.new("L", (8, 8), 0)) is None


def test_pick_other_never_returns_the_same_row():
    import random
    rng = random.Random(0)
    assert all(build_real.pick_other(3, [1, 2, 3, 4], rng) != 3 for _ in range(50))


def test_vivid_live_items_only_train_items_with_a_live_input(tmp_path):
    root = tmp_path / "vivid_dress_stills"
    for name, split, live in (("a", "train", True), ("b", "test", True), ("c", "train", False)):
        d = root / name
        d.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(d / "target.png")
        Image.new("RGB", (8, 8)).save(d / "ref.jpg")
        meta = {"split": split, "category": "dress"}
        if live:
            Image.new("RGB", (8, 8)).save(d / "live_source.png")
            meta["live_source"] = {"model": "live0811"}
        (d / "meta.json").write_text(json.dumps(meta))
    items = list(build_real.vivid_live_items([root, tmp_path / "missing"]))
    assert [i[0] for i in items] == ["vivid_vivid_dress_stills_a"] and items[0][1] == "dress"


def test_timeshift_math_puts_half_the_updates_in_fluxrts_band():
    e = math.exp(2.01)
    shifted = lambda s: e / (e + (1 / s - 1))  # noqa: E731
    assert abs(shifted(0.5) - 0.882) < 0.002   # FluxRT's 2nd step at 288x512
    n = 20000
    share = sum(shifted((k + 0.5) / n) >= 0.88 for k in range(n)) / n
    assert 0.45 < share < 0.55


def test_timeshift_patch_applies_after_the_garment_patch(tmp_path):
    local = Path("/home/user/repos/diffusers")
    trainer = local / "examples" / "dreambooth" / "train_dreambooth_lora_flux2_klein_img2img.py"
    if not trainer.exists():
        pytest.skip("diffusers checkout not present")
    # The pinned, unpatched file (the local checkout may already carry the garment patch).
    r = subprocess.run(["git", "-C", str(local), "show", f"HEAD:examples/dreambooth/{trainer.name}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("pinned diffusers file not available")
    repo = tmp_path / "d"
    (repo / "examples" / "dreambooth").mkdir(parents=True)
    (repo / "examples" / "dreambooth" / trainer.name).write_text(r.stdout)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name in ("flux2-klein-garment.patch", "flux2-klein-timeshift.patch"):
        r = subprocess.run(["git", "-C", str(repo), "apply", str(ROOT / "training" / "patches" / name)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    src = (repo / "examples" / "dreambooth" / trainer.name).read_text()
    assert "--time_shift_mu" in src and "shift / (shift + (1.0 / base_sigma - 1.0))" in src
    compile(src, trainer.name, "exec")


def test_run_train_uses_one_folder_per_model_and_passes_the_shift():
    sh = (ROOT / "training" / "fluxrt_lora" / "run_train.sh").read_text()
    assert 'KLEIN="$AZ_CKPT/$(basename "$KLEIN_REPO")"' in sh   # Base never lands in the distilled folder
    assert "flux2-klein-timeshift.patch" in sh and "--time_shift_mu" in sh
    assert '--checkpoints_total_limit="${AZ_CKPT_LIMIT:-2}"' in sh


def test_fx1_trains_on_base_with_shift_and_live_garment_size():
    sh = (ROOT / "training" / "fluxrt_lora" / "fx1.sh").read_text()
    for s in ("KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B", "AZ_TIME_SHIFT_MU=2.01", "AZ_REF_SIZE=288x512",
              "--model-dir $C/FLUX.2-klein-4B", "--steps 2", "--prompt-style compact", "--ref-size 288x512",
              "runpodctl stop pod", "fx1_started"):
        assert s in sh, s
    r = subprocess.run(["bash", "-n", str(ROOT / "training" / "fluxrt_lora" / "fx1.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_test_garments_cover_print_logo_and_shoes():
    g = json.loads((ROOT / "training" / "eval" / "test_garments.json").read_text())["items"]
    cats = {x["category"] for x in g}
    assert {"dresses", "tops", "bottoms", "shoes"} <= cats
    assert any(x["test_for"] == "print" for x in g) and any(x["test_for"] == "logo" for x in g)
    assert all(x["image_url"].startswith("https://") for x in g)


def test_select_test_clips_choose_and_noface(tmp_path):
    from eval import select_test_clips as st
    scored = [{"clip": f"c{i}", "motion": i / 10, "brightness": 1 - i / 10} for i in range(6)]
    used: set = set()
    still = st.choose(scored, "motion", 2, False, used)
    moving = st.choose(scored, "motion", 2, True, used)
    assert [s["clip"] for s in still] == ["c0", "c1"] and [s["clip"] for s in moving] == ["c5", "c4"]
    d = tmp_path / "shoe_set" / "shoe_01"
    d.mkdir(parents=True)
    Image.new("RGB", (90, 160), (9, 9, 9)).save(d / "person.png")
    Image.new("RGB", (20, 20)).save(d / "ref.jpg")
    (d / "meta.json").write_text(json.dumps({"category": "shoes"}))
    assert st.make_noface(tmp_path / "shoe_set", tmp_path / "noface") == 1
    out = Image.open(tmp_path / "noface" / "noface_shoe_01" / "person.png")
    assert out.size == (90, 80)                            # lower half only: no face in view

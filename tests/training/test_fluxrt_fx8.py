"""fxreal8 (owner, 2026-11-18): favourite #2 frozen with a small add-on on top, a steadiness term, a reference term
to favourite #2, a 20-step safety check, and a run-time steadiness filter tested on favourite #2 itself.
No GPU and no torch here: the filter, the picker and the wiring. The trainer helpers are tested on a real tiny
FLUX.2 transformer in test_fluxrt_fx8_torch.py, and the whole Pod script in test_fluxrt_fx7_dryrun.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
SCRIPT = ROOT / "training/fluxrt_lora/fx6b_continue.sh"


# ---------------------------------------------------------------- the run-time steadiness filter
def test_still_weight_is_one_where_nothing_moved_and_zero_where_it_did():
    from eval.steady_filter import still_weight
    a = np.full((40, 30, 3), 100, np.uint8)
    b = a.copy()
    b[:20] = 200                                                    # the top half moved a lot
    s = still_weight(a, b, motion=0.04, blur=3)
    assert s.shape == (40, 30) and s[30:].min() == pytest.approx(1.0) and s[:15].max() == pytest.approx(0.0)
    assert np.all((s >= 0) & (s <= 1))
    assert still_weight(a, a).min() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        still_weight(a, a[:10])


def test_the_filter_calms_flicker_where_the_camera_is_still_and_leaves_movement_alone():
    from eval.steady_filter import SteadyFilter
    rng = np.random.default_rng(0)
    cam = np.full((32, 32, 3), 120, np.uint8)
    f = SteadyFilter(0.5, motion=0.04, blur=3)
    base = np.full((32, 32, 3), 128.0)
    outs_raw, outs = [], []
    for t in range(12):
        o = (base + rng.normal(0, 10, base.shape)).clip(0, 255).astype(np.uint8)   # the model shimmers
        outs_raw.append(o.astype(float)); outs.append(f(cam, o).astype(float))
    raw_flicker = np.mean([np.abs(outs_raw[t] - outs_raw[t - 1]).mean() for t in range(1, 12)])
    new_flicker = np.mean([np.abs(outs[t] - outs[t - 1]).mean() for t in range(1, 12)])
    assert new_flicker < 0.7 * raw_flicker
    # where the camera moved, the new output is passed through untouched
    g = SteadyFilter(0.5, motion=0.04, blur=1)
    first = np.zeros((8, 8, 3), np.uint8)
    g(first, np.full((8, 8, 3), 10, np.uint8))
    moved = np.full((8, 8, 3), 255, np.uint8)
    out = g(moved, np.full((8, 8, 3), 200, np.uint8))
    assert np.all(out == 200)
    # the first frame of a clip, off, and a reset are untouched too
    h = SteadyFilter(0.0)
    o = np.full((4, 4, 3), 7, np.uint8)
    assert np.all(h(cam[:4, :4], o) == 7)
    g.reset()
    assert np.all(g(moved, o) == 7)
    with pytest.raises(ValueError):
        SteadyFilter(1.0)


def test_the_renderer_uses_the_filter_only_when_asked():
    src = (ROOT / "training/eval/generate_fluxrt.py").read_text()
    assert '"--steady-blend", type=float, default=0.0' in src and "SteadyFilter(a.steady_blend, a.steady_motion)" in src
    assert "if steady is not None:" in src


# ---------------------------------------------------------------- the picker: the run-time test is never the pick
def _entry(ev: Path, d: str, step: int, name: str, kind: str, colour: float, label: str):
    e = ev / d
    e.mkdir(parents=True)
    (e / "meta.json").write_text(json.dumps({"step": step, "name": name, "weights": f"/w/{name}", "kind": kind,
                                             "label": label}))
    for s in ("fx6test", "dress10", "cases5"):
        (e / s).mkdir()
        rows = [{"metric": "colour_de", "model": name, "baseline": "fav2", "mean": colour, "baseline_mean": 12.0}]
        (e / s / "garment.json").write_text(json.dumps({"scorecard": rows}))
        (e / s / "preserve_check.json").write_text(json.dumps({"verdict": {name: {"pass": True}}}))
    (e / ".scored").touch()


def test_runtime_entries_are_reported_but_never_picked(tmp_path, capsys):
    from fluxrt_lora.pick_best_save import main
    ev = tmp_path / "ev"
    _entry(ev, "aa53", -300, "fav2seed53", "aa", 12.1, "favourite #2, another random seed")
    _entry(ev, "rt50", -450, "fav2rt50", "runtime", 10.0, "favourite #2 + run-time steadiness 0.50")
    _entry(ev, "fx8s500", 500, "fx8500", "save", 11.5, "fxreal8 step 500 at 0.65")
    r = main([str(ev), "--baseline", "fav2", "--weights", "fx6test=1,dress10=1,cases5=0.5"])
    out = capsys.readouterr().out
    assert r["best"]["label"] == "fxreal8 step 500 at 0.65"         # the run-time test scores higher, but is not a file
    assert "RUNTIME favourite #2 + run-time steadiness 0.50: score +" in out and "worth a live test" in out
    assert r["runtime"][0]["beyond_chance"] is True
    assert json.loads((ev / "best.json").read_text())["runtime"][0]["label"] == "favourite #2 + run-time steadiness 0.50"


# ---------------------------------------------------------------- wiring
def test_the_trainer_wrapper_applies_the_fx8_patch_last_and_passes_its_options():
    rt = (ROOT / "training/fluxrt_lora/run_train.sh").read_text()
    assert rt.index("flux2-klein-diskcache.patch") < rt.index("flux2-klein-fx8.patch")
    for x in ('${AZ_FROZEN_LORA:+--frozen_init_lora="$AZ_FROZEN_LORA"}', '${AZ_STEADY_RATIO:+--steady_ratio="$AZ_STEADY_RATIO"}',
              '${AZ_REF_WEIGHT:+--ref_weight="$AZ_REF_WEIGHT"}', '${AZ_SAFETY_STEPS:+--safety_steps="$AZ_SAFETY_STEPS"}'):
        assert x in rt, x
    patch = (ROOT / "training/patches/flux2-klein-fx8.patch").read_text()
    assert patch.count("diff --git") == 1 and "train_dreambooth_lora_flux2_klein_img2img.py" in patch
    for x in ('"--frozen_init_lora"', '"--steady_ratio"', '"--ref_weight"', '"--safety_steps"', "FX8_SAFETY_FAILED",
              "FX8_SAFETY_OK", "STEADY_WEIGHT", "FROZEN_LORA merged"):
        assert x in patch, x
    # the reference prediction switches the add-on off and always back on
    ref = patch[patch.index("_az_ref_pred = None"):patch.index("# Predict the noise residual")]
    assert "torch.no_grad()" in ref and ref.index("disable_adapters()") < ref.index("finally:") < ref.index("enable_adapters()")
    # every option is off unless set: older runs train exactly as before
    for x in ('"--steady_ratio",\n+        type=float,\n+        default=0.0', '"--ref_weight",\n+        type=float,\n+        default=0.0',
              '"--safety_steps",\n+        type=int,\n+        default=0'):
        assert x in patch, x


def test_the_fxreal8_mode_settings():
    sh = SCRIPT.read_text()
    line = next(l for l in sh.splitlines() if l.strip().startswith("fav2_addon) D_LR="))
    for x in ("D_LR=1e-4", "D_SCALE=65", "D_ANCHOR=0", "D_AA=1", "D_STEPS=1000", "D_EVERY=250", "D_RANK=16", 'D_RUNTIME="30 50"'):
        assert x in line, x
    for x in ('fav2_addon) EXP="${AZ_EXP:-fxreal8}"', 'STEPS="${AZ_MAX_STEPS:-$D_STEPS}"', 'EVERY="${AZ_SAVE_EVERY:-$D_EVERY}"',
              "AZ_FROZEN_LORA=$UNSCALED_DIR AZ_STEADY_RATIO=${AZ_STEADY_RATIO:-0.25} AZ_REF_WEIGHT=${AZ_REF_WEIGHT:-1.0} "
              "AZ_SAFETY_STEPS=${AZ_SAFETY_STEPS:-20}",
              'min=("$UNSCALED_DIR/pytorch_lora_weights.safetensors" "$w"); mw=(1.0 1.0)', '"--steady-blend 0.$rt"',
              "AZ_INIT_LORA=$TRAIN_INIT $FX8_ENV bash"):
        assert x in sh, x

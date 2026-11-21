"""The better yardstick (owner, 2026-11-21): more clips, several seeds, a 95% range on every figure. Made-up records
where the right answer is known: a real gain must be called, noise must not, a seed that only looked good on the
clips it was picked on must not hold on fresh ones."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
BASE = "fav2_r528"
OLD = {"fx6test": 24, "dress10": 10, "cases5": 5}
FRESH = {"fx12test": 48}
TYPES = ("tops", "dresses", "bottoms")


def _lists(fxa: Path):
    fxa.mkdir(parents=True, exist_ok=True)
    for s, n in {**OLD, **FRESH}.items():
        items = [{"key": f"{s}_{i}", "category": "dresses" if s == "dress10" else TYPES[i % 3]} for i in range(n)]
        (fxa / f"{s}.json").write_text(json.dumps(items))


def _entry(yard: Path, name: str, model: str, seed: int, sets: dict, effect, noise: float = 0.0, rng_seed: int = 0):
    """effect(set, clip index, garment type) -> the model's colour as a multiple of favourite #2's 12.0 (1 = the
    same); noise: per-clip spread on top (favourite #2's own numbers stay fixed)."""
    rng = np.random.default_rng(rng_seed)
    d = yard / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"name": name, "label": name, "model": model, "seed": seed}))
    for s, n in sets.items():
        sd = d / s
        sd.mkdir()
        items, per, st = {}, {}, {}
        for i in range(n):
            c = f"{s}_{i}"
            t = "dresses" if s == "dress10" else TYPES[i % 3]
            mult = effect(s, i, t) * (1 + noise * rng.standard_normal())
            items[c] = {name: {"colour_de": 12.0 * mult, "fabric": 0.5, "jitter": 0.02 * (1 + noise * rng.standard_normal())},
                        BASE: {"colour_de": 12.0, "fabric": 0.5, "jitter": 0.02}}
            face = {"face_match": 0.93, "skin_de": 3.0, "face_drift": 1.0, "texture": 1.0}
            per[c] = {name: face, BASE: face}
            st[c] = {name: {"wobble_garment": 0.02, "sharpness": 1.0, "colour_wander": 1.0},
                     BASE: {"wobble_garment": 0.02, "sharpness": 1.0, "colour_wander": 1.0}}
        (sd / "garment.json").write_text(json.dumps({"items": items}))
        (sd / "face_skin.json").write_text(json.dumps({"per_clip": per}))
        (sd / "stability.json").write_text(json.dumps(st))


def _run(yard: Path, capsys, **kw):
    from eval.yardstick import main
    res = main([str(yard), "--baseline", BASE, "--old", *OLD, "--fresh", *FRESH,
                "--weights", "fx6test=1,dress10=1,cases5=0.5,fx12test=2", "--boot", str(kw.get("boot", 400))])
    return res, capsys.readouterr().out


def test_a_real_gain_is_called_and_pure_noise_is_not(tmp_path, capsys):
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    allsets = {**OLD, **FRESH}
    # favourite #2 at two other seeds: seed 54 is truly better on every clip, seed 55 is the same model with noise
    _entry(yard, "fav2seed54_r528", "fav2", 54, allsets, lambda s, i, t: 0.80, noise=0.05, rng_seed=1)
    _entry(yard, "fav2seed55_r528", "fav2", 55, allsets, lambda s, i, t: 1.00, noise=0.25, rng_seed=2)
    res, out = _run(yard, capsys)
    by = {x["seed"]: x for x in res["seeds"]}
    # colour is 1 of the 10 measures these records hold that differ or not: +20% colour = +2.0% score
    assert by[54]["fresh"]["score"] == pytest.approx(0.2 / 10, abs=0.004) and by[54]["fresh"]["low"] > 0
    assert by[55]["fresh"]["low"] < 0 < by[55]["fresh"]["high"]
    assert "YARD_SEED favourite #2 at seed 54 against seed 52: old clips +" in out and "holds on fresh clips" in out
    assert "YARD_SEED favourite #2 at seed 55 against seed 52:" in out and "not shown on fresh clips" in out
    assert "YARD_SETS old clips: 39 in fx6test, dress10, cases5 (20 dresses, 10 tops, 9 bottoms)" in out
    assert "YARD_SETS fresh clips: 48 in fx12test (16 dresses, 16 tops, 16 bottoms)" in out
    assert "YARD_RESOLUTION this yardstick shows differences of about" in out


def test_a_seed_that_only_looked_good_where_it_was_picked_does_not_hold(tmp_path, capsys):
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    # better on the old clips only (luck of the draw there), the same on the fresh ones
    _entry(yard, "fav2seed61_r528", "fav2", 61, {**OLD, **FRESH},
           lambda s, i, t: 0.7 if s in OLD else 1.0, noise=0.1, rng_seed=3)
    res, out = _run(yard, capsys)
    r = res["seeds"][0]
    assert r["old"]["low"] > 0 and r["fresh"]["low"] < 0 < r["fresh"]["high"]
    assert "not shown on fresh clips" in out and "holds on fresh clips" not in out


def test_a_model_is_compared_with_favourite_2_at_the_same_seeds_and_per_garment_type(tmp_path, capsys):
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    allsets = {**OLD, **FRESH}
    # favourite #2 itself differs by seed: seed 54 is 10% better everywhere
    _entry(yard, "fav2seed54_r528", "fav2", 54, allsets, lambda s, i, t: 0.90)
    # the model: better colour on dresses only (x0.8 of favourite #2 AT THE SAME SEED), the same elsewhere
    def model(seed_mult):
        return lambda s, i, t: seed_mult * (0.8 if t == "dresses" else 1.0)
    _entry(yard, "fx8750s65_r528", "fxreal8 step 750", 52, allsets, model(1.0), noise=0.02, rng_seed=4)
    _entry(yard, "fx8750s65seed54_r528", "fxreal8 step 750", 54, allsets, model(0.90), noise=0.02, rng_seed=5)
    res, out = _run(yard, capsys)
    m = res["models"][0]
    assert m["model"] == "fxreal8 step 750" and m["seeds"] == [52, 54] and m["all"]["clips"] == 87
    # against favourite #2 at the same seed the seed's own 10% cancels: only the dresses' 20% is left
    assert m["garments"]["dresses"]["score"] == pytest.approx(0.2 / 10, abs=0.003) and m["garments"]["dresses"]["low"] > 0
    assert abs(m["garments"]["tops"]["score"]) < 0.003 and m["garments"]["tops"]["low"] < 0 < m["garments"]["tops"]["high"]
    assert len(m["all"]["per_seed"]) == 2 and all(x > 0 for x in m["all"]["per_seed"])
    assert m["all"]["areas"]["colour"]["low"] > 0
    assert "YARD_MODEL fxreal8 step 750 against favourite #2 at the same seeds (52, 54), 87 clips: +" in out
    assert ": better; each seed alone +" in out
    line = next(l for l in out.splitlines() if l.startswith("YARD_MODEL_GARMENT fxreal8 step 750:"))
    assert " better, 36 clips" in line.split("|")[0] and "no difference shown" in line.split("|")[1]
    assert "YARD_MODEL_AREAS fxreal8 step 750: colour +" in out


def test_the_range_gets_narrower_with_more_clips_and_more_seeds(tmp_path):
    from eval.yardstick import Pairs, load
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    allsets = {**OLD, **FRESH}
    for k, seed in enumerate((52, 54, 55, 56)):
        if seed != 52:
            _entry(yard, f"fav2seed{seed}_r528", "fav2", seed, allsets, lambda s, i, t: 1.0)
        _entry(yard, f"m_seed{seed}", "model", seed, allsets, lambda s, i, t: 1.0, noise=0.3, rng_seed=10 + k)
    table = load(yard, BASE, list(allsets))
    w = {"fx6test": 1, "dress10": 1, "cases5": 0.5, "fx12test": 2}
    width = lambda r: r["high"] - r["low"]                                   # noqa: E731
    one_old = Pairs(table, "model", "fav2", [(52, 52)], list(OLD), w).result(400)
    one_all = Pairs(table, "model", "fav2", [(52, 52)], list(allsets), w).result(400)
    four_all = Pairs(table, "model", "fav2", [(s, s) for s in (52, 54, 55, 56)], list(allsets), w).result(400)
    assert width(four_all) < width(one_all) < width(one_old)
    assert four_all["low"] < 0 < four_all["high"]                            # a model that is the same is not called


def test_clips_without_a_number_and_entries_without_a_seed_are_left_out(tmp_path, capsys):
    from eval.yardstick import load
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    _entry(yard, "fav2seed54_r528", "fav2", 54, FRESH, lambda s, i, t: 0.9)
    g = yard / "fav2seed54_r528/fx12test/garment.json"
    d = json.loads(g.read_text())
    d["items"]["fx12test_0"]["fav2seed54_r528"]["colour_de"] = None
    d["items"]["fx12test_1"]["fav2seed54_r528"]["colour_de"] = float("nan")
    g.write_text(json.dumps(d))
    (yard / "junk").mkdir()
    (yard / "junk/meta.json").write_text(json.dumps({"name": "junk", "label": "junk"}))      # no model, no seed
    table = load(yard, BASE, list(FRESH))
    assert set(table) == {("fav2", 54), ("fav2", 52)}
    res, out = _run(yard, capsys)
    assert res["seeds"][0]["fresh"]["score"] == pytest.approx(0.1 / 10, abs=1e-6)            # the other 46 clips

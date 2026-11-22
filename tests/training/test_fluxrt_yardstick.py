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


def _entry(yard: Path, name: str, model: str, seed: int, sets: dict, effect, noise: float = 0.0, rng_seed: int = 0,
           jitter=None):
    """effect(set, clip index, garment type) -> the model's colour as a multiple of favourite #2's 12.0 (1 = the
    same); noise: per-clip spread on top (favourite #2's own numbers stay fixed); jitter: the same for the shape
    jitter (favourite #2's 0.02)."""
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
            jit = jitter(s, i, t) if jitter else 1.0
            items[c] = {name: {"colour_de": 12.0 * mult, "fabric": 0.5, "jitter": 0.02 * jit * (1 + noise * rng.standard_normal())},
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
                "--weights", "fx6test=1,dress10=1,cases5=0.5,fx12test=2", "--boot", str(kw.get("boot", 400))]
               + (["--worn", str(kw["worn"])] if kw.get("worn") else []))
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
    assert ": better; each seed alone +" in out and "; on the 48 fresh clips alone +" in out
    assert m["fresh"]["clips"] == 48 and m["fresh"]["score"] == pytest.approx(0.2 / 10 / 3, abs=0.003)   # a third are dresses
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


def test_the_direction_of_the_fabric_gap_is_reported_next_to_favourite_2(tmp_path, capsys):
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    _entry(yard, "fav2seed54_r528", "fav2", 54, FRESH, lambda s, i, t: 1.0)
    _entry(yard, "m52", "fxreal8 step 750", 52, FRESH, lambda s, i, t: 1.0)
    _entry(yard, "m54", "fxreal8 step 750", 54, FRESH, lambda s, i, t: 1.0)
    # the scorers' newer records: favourite #2 x0.92 of the photo's spread at seed 52 and x0.96 at seed 54; the
    # model x0.80 at both seeds
    for name, mine in (("fav2seed54_r528", 0.96), ("m52", 0.80), ("m54", 0.80)):
        g = yard / name / "fx12test/garment.json"
        d = json.loads(g.read_text())
        for clip, v in d["items"].items():
            v[name]["fabric_ratio"] = mine
            v[BASE]["fabric_ratio"] = 0.92
        g.write_text(json.dumps(d))
    res, out = _run(yard, capsys)
    assert "YARD_FABRIC favourite #2: the garment's light-dark spread is x0.92 at seed 52, x0.96 at seed 54 of the photo's" in out
    # against favourite #2 at the SAME seeds: (0.92 x 0.96) ** 0.5 = 0.94
    assert ("YARD_FABRIC fxreal8 step 750: x0.80 of the photo's, favourite #2 at the same seeds and clips x0.94 (96 renders): "
            "flatter than favourite #2") in out
    assert res["fabric"]["models"]["fxreal8 step 750"]["pairs"] == 96
    # records without the direction (older scorings) give no line and nothing breaks
    for name in ("fav2seed54_r528", "m52", "m54"):
        g = yard / name / "fx12test/garment.json"
        d = json.loads(g.read_text())
        for v in d["items"].values():
            v[name].pop("fabric_ratio")
            v[BASE].pop("fabric_ratio")
        g.write_text(json.dumps(d))
    res, out = _run(yard, capsys)
    assert "YARD_FABRIC" not in out and "fabric" not in res


# ---------------------------------------------------------------- after the seed sweep (2026-11-22)
SWEEP = {53: 0.72, 54: 0.78, 55: 0.84, 56: 0.90, 57: 0.96, 58: 1.00}      # colour as a multiple of seed 52's


def test_every_other_seed_better_on_old_and_fresh_clips_means_the_live_seed_is_a_poor_one(tmp_path, capsys):
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    for k, (seed, mult) in enumerate(SWEEP.items()):
        _entry(yard, f"fav2seed{seed}_r528", "fav2", seed, {**OLD, **FRESH}, lambda s, i, t, m=mult: m, noise=0.03, rng_seed=20 + k)
    res, out = _run(yard, capsys)
    avg = res["seed_average"]
    assert avg["old"]["seeds"] == avg["fresh"]["seeds"] == list(SWEEP) and avg["fresh"]["low"] > 0 and avg["old"]["low"] > 0
    # the six seeds' colours average x0.867 of seed 52's: +13.3% colour, 1 of 10 measures
    assert avg["fresh"]["score"] == pytest.approx(0.1333 / 10, abs=0.002)
    assert "YARD_SEED_AVERAGE the 6 other seeds together against seed 52: old clips +" in out
    assert ": seed 52 is a poor seed on the fresh clips too" in out
    line = next(l for l in out.splitlines() if l.startswith("YARD_SEED_AVERAGE_GARMENT the other seeds together, fresh clips: "))
    assert [x.strip().split()[0] for x in line.split(": ", 1)[1].split("|")] == ["dresses", "tops", "bottoms"]
    assert line.count(" better, 16 clips") == 3
    # the same order on old and fresh clips: a seed's worth is its own, not the clips'
    assert res["seed_agreement"] == {"rank_correlation": pytest.approx(1.0), "seeds": 6}
    assert "rank correlation +1.00 (1 = the same order, 0 = no relation): a seed that is good on some clips is good on others" in out
    # the seeds that are really better are picked, best first; seed 58 is the same as seed 52 and is not
    assert res["seed_pick"][:4] == [53, 54, 55, 56] and 58 not in res["seed_pick"] and res["seed_show"] == [53, 54, 55]
    assert ("YARD_SEED_PICK better than seed 52 on the fresh clips at the level that allows for 6 seeds being tried, and no area "
            "clearly worse on both the old and the fresh clips: 53 (fresh +") in out
    assert all("low_strict" in x["fresh"] and x["fresh"]["low_strict"] <= x["fresh"]["low"] for x in res["seeds"])
    assert out.count("YARD_SEED_GARMENT seed ") == 3 and "YARD_SEED_GARMENT seed 53, fresh clips: dresses +" in out


def test_seeds_that_were_lucky_on_the_old_clips_only_are_not_picked_and_their_order_does_not_carry_over(tmp_path, capsys):
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    # on the old clips every seed looks better (as in the sweep); on the fresh clips they are seed 52 with noise,
    # and what little order there is runs the other way
    for k, (seed, mult) in enumerate(SWEEP.items()):
        _entry(yard, f"fav2seed{seed}_r528", "fav2", seed, {**OLD, **FRESH},
               lambda s, i, t, m=mult: m - 0.1 if s in OLD else 1.0 + (1.0 - m) * 0.02, noise=0.15, rng_seed=40 + k)
    res, out = _run(yard, capsys)
    avg = res["seed_average"]
    assert avg["old"]["low"] > 0 and avg["fresh"]["low"] < 0 < avg["fresh"]["high"]
    assert "on the fresh clips seed 52 is not shown to differ from the others" in out
    # one of the six does pass the ordinary 95% test, by luck: that is what trying six seeds does, and why a pick
    # must stand at the stricter level. It is named, as luck.
    assert res["seed_pick"] == [] and res["seed_pick_maybe"] == [54] and "YARD_SEED_GARMENT" not in out
    assert "clips: none; better at the ordinary 95% level only, which one of 6 seeds can reach by luck: 54 (fresh +" in out
    assert len(res["seed_show"]) == 3                                         # the videos still show the best three
    assert res["seed_agreement"]["rank_correlation"] < 0.6


def test_a_seed_that_is_better_overall_but_worse_in_one_area_is_named_apart(tmp_path, capsys):
    import re
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    allsets = {**OLD, **FRESH}
    _entry(yard, "fav2seed57_r528", "fav2", 57, allsets, lambda s, i, t: 0.75, noise=0.01, rng_seed=1)
    # seed 60: much better colour everywhere, but the garment's shape jitters a fifth more on ALL clips
    # (the flicker area is the jitter and two steadiness measures: -20% / 3)
    _entry(yard, "fav2seed60_r528", "fav2", 60, allsets, lambda s, i, t: 0.60, noise=0.01, rng_seed=2,
           jitter=lambda s, i, t: 1.2)
    # seed 61: the same, but the jitter is on the old clips only: with so many areas and seeds that can be luck, so
    # it is a remark beside the seed, not a reason against it
    _entry(yard, "fav2seed61_r528", "fav2", 61, allsets, lambda s, i, t: 0.70, noise=0.01, rng_seed=3,
           jitter=lambda s, i, t: 1.2 if s in OLD else 1.0)
    res, out = _run(yard, capsys)
    assert res["seed_pick"] == [61, 57] and res["seed_pick_flawed"] == [60] and res["seed_show"] == [61, 57, 60]
    line = next(l for l in out.splitlines() if l.startswith("YARD_SEED_PICK"))
    assert re.search(r"clips: 61 \(fresh \+3\.0%, old \+1\.0%; but flicker -6\.[5-9]% on the old clips only\), 57 \(fresh \+2\.5%, old \+2\.5%\); ", line), line
    assert re.search(r"better but clearly worse in an area: 60 \(flicker -6\.[5-9]% on the old and -6\.[5-9]% on the fresh clips\)", line), line
    assert list(res["seed_remarks"]) == ["61"] and res["seed_remarks"]["61"][0].endswith("on the old clips only")


def test_the_control_is_favourite_2_again_and_says_whether_the_stored_videos_can_be_trusted(tmp_path, capsys):
    from eval.yardstick import CONTROL
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    allsets = {**OLD, **FRESH}
    for seed in (53, 54):
        _entry(yard, f"fav2seed{seed}_r528", "fav2", seed, allsets, lambda s, i, t: 0.9 if s in OLD else 1.0)
    _entry(yard, "fav2again_r528", CONTROL, 52, allsets, lambda s, i, t: 1.0)
    res, out = _run(yard, capsys)
    assert res["control"]["fresh"]["ok"] and res["control"]["old"]["ok"] and res["models"] == []
    assert ("YARD_CONTROL favourite #2 rendered again at seed 52 against its videos of this run, 48 fresh clips: +0.0% "
            "(95% +0.0% to +0.0%), largest area 0.0%: the same: a render can be repeated") in out
    assert "against its stored videos, 39 old clips: +0.0% (95% +0.0% to +0.0%), largest area 0.0%: the same: the stored videos are what today's renderer makes" in out
    assert "YARD_CONTROL_SEEDS the 2 other seeds together against favourite #2 rendered again today at seed 52, old clips: +1.0%" in out
    assert "YARD_MODEL" not in out and "YARD_FABRIC favourite #2 rendered again" not in out
    # the stored seed-52 videos of the OLD clips are stale: today's renderer makes what the other seeds make there.
    # Then "every seed beats seed 52 on the old clips" was the stored videos, not the seed.
    import shutil
    shutil.rmtree(yard / "fav2again_r528")
    _entry(yard, "fav2again_r528", CONTROL, 52, allsets, lambda s, i, t: 0.9 if s in OLD else 1.0)
    res, out = _run(yard, capsys)
    assert res["control"]["fresh"]["ok"] and not res["control"]["old"]["ok"]
    assert ("39 old clips: +1.0% (95% +1.0% to +1.0%), largest area 10.0%: NOT the same: the stored seed 52 videos are not "
            "what today's renderer makes; on the old clips read the seeds against today's render (YARD_CONTROL_SEEDS)") in out
    assert "at seed 52, old clips: +0.0% (95% +0.0% to +0.0%): no difference shown" in out
    assert res["control"]["seeds_old"]["score"] == pytest.approx(0.0, abs=1e-9)
    # a renderer that cannot repeat itself even within one run: said as such, and the old clips are read with it
    shutil.rmtree(yard / "fav2again_r528")
    _entry(yard, "fav2again_r528", CONTROL, 52, allsets, lambda s, i, t: 0.9)
    res, out = _run(yard, capsys)
    assert "48 fresh clips: +1.0% (95% +1.0% to +1.0%), largest area 10.0%: NOT the same: two renders of the same thing differ by this much" in out
    assert "NOT the same, as on the fresh clips: the renderer itself is not repeatable to that extent" in out


def test_too_few_clips_are_never_called_better_or_worse():
    from eval.yardstick import word
    assert word({"low": 0.01, "high": 0.02, "clips": 4}) == "too few clips to say"
    assert word({"low": 0.01, "high": 0.02, "clips": 5}) == "better" and word({"low": -0.02, "high": -0.01, "clips": 16}) == "worse"
    assert word({"low": -0.01, "high": 0.02, "clips": 48}) == "no difference shown"


def _worn(worn: Path, sets: dict, ratio: float = 2.0):
    """The real videos' scores as the scorers write them for one model alone (eval.worn_reference names it worn)."""
    for s, n in sets.items():
        d = worn / s
        d.mkdir(parents=True)
        items, st = {}, {}
        for i in range(n):
            t = "dresses" if s == "dress10" else TYPES[i % 3]
            rec = {"colour_de": 9.0, "likeness": 0.7, "fabric": abs(float(np.log(ratio))), "fabric_ratio": ratio, "jitter": 0.01,
                   "frames": 49}
            if t == "dresses":
                rec["length_ratio"] = 0.98
            items[f"{s}_{i}"] = {"worn": rec}
            same = {"wobble_garment": 0.01, "sharpness": 2.0, "colour_wander": 0.5}
            st[f"{s}_{i}"] = {"frames": 49, "walk_in_frames": 0, "camera": same, "worn": same}
        (d / "garment.json").write_text(json.dumps({"baseline": "worn", "models": ["worn"], "scorecard": [], "items": items}))
        (d / "stability.json").write_text(json.dumps(st))


def test_the_real_videos_of_the_same_garments_are_shown_next_to_favourite_2_measure_by_measure(tmp_path, capsys):
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    allsets = {**OLD, **FRESH}
    _entry(yard, "fav2seed54_r528", "fav2", 54, allsets, lambda s, i, t: 1.0)
    _entry(yard, "near52", "model near", 52, allsets, lambda s, i, t: 1.0)
    _entry(yard, "far52", "model far", 52, allsets, lambda s, i, t: 1.0)
    for name, mine in (("fav2seed54_r528", 2.2), ("near52", 2.05), ("far52", 2.6)):
        for s in allsets:
            g = yard / name / s / "garment.json"
            d = json.loads(g.read_text())
            for v in d["items"].values():
                v[name]["fabric_ratio"] = mine
                v[BASE]["fabric_ratio"] = 2.2
                v[BASE]["length_ratio"] = 0.9
            g.write_text(json.dumps(d))
    real = {k: v for k, v in allsets.items() if k != "cases5"}               # the owner's own photos have no video
    _worn(yard / "worn", real)
    res, out = _run(yard, capsys, worn=yard / "worn")
    assert ("YARD_REAL the same garments as really worn (82 real videos) next to favourite #2 at seed 52 on the same garments: "
            "colour distance from the photo: real 9.0, favourite #2 12.0 | light-dark spread (1 = the photo's): real x2.00, "
            "favourite #2 x2.20 | dress length (1 = the photo's): real x0.98, favourite #2 x0.90 | shape jitter: real 0.010, "
            "favourite #2 0.020 | wobble: real 0.01, favourite #2 0.02 | colour wander: real 0.50, favourite #2 1.00 | "
            "sharpness: real 2.0, favourite #2 1.0") in out
    assert res["real"]["clips"] == 82 and res["real"]["all"]["length_ratio"]["clips"] == 34   # the dresses only
    assert "YARD_REAL_GARMENT dresses, 34 real videos: colour distance from the photo: real 9.0" in out
    assert "YARD_REAL_GARMENT tops, 24 real videos:" in out and "YARD_REAL_GARMENT bottoms, 24 real videos:" in out
    tops = next(l for l in out.splitlines() if l.startswith("YARD_REAL_GARMENT tops"))
    assert "dress length" not in tops
    # the fabric score calls the photo's own spread ideal; the real videos say where a worn garment really stands
    assert "of the photo's (1.00 = the same, under 1 = flatter); the same garments as really worn: x2.00" in out
    assert ("YARD_FABRIC model far: x2.60 of the photo's, favourite #2 at the same seeds and clips x2.20 (87 renders): "
            "more contrast than favourite #2; further from the real videos than favourite #2") in out
    assert ("YARD_FABRIC model near: x2.05 of the photo's, favourite #2 at the same seeds and clips x2.20 (87 renders): "
            "flatter than favourite #2; nearer to the real videos than favourite #2") in out
    assert res["fabric"]["real"] == pytest.approx(2.0)
    # the folder of the real videos is not taken for a render of its own; without it nothing about them is said
    assert {m["model"] for m in res["models"]} == {"model near", "model far"}
    res, out = _run(yard, capsys)
    assert "YARD_REAL" not in out and "really worn" not in out and "real" not in res


def test_a_score_file_cut_off_in_the_middle_is_named_and_never_taken_for_scores(tmp_path, capsys):
    fxa = tmp_path / "fxa"
    yard = fxa / "yard"
    _lists(fxa)
    _entry(yard, "fav2seed54_r528", "fav2", 54, {**OLD, **FRESH}, lambda s, i, t: 0.9)
    _entry(yard, "fav2seed55_r528", "fav2", 55, {**OLD, **FRESH}, lambda s, i, t: 0.9)
    g = yard / "fav2seed55_r528/fx12test/garment.json"
    g.write_text(g.read_text()[:300])                                        # the Pod stopped while it was written
    res, out = _run(yard, capsys)
    assert res["unreadable"] == ["fav2seed55_r528/fx12test/garment.json"]
    assert "YARD_NOTE 1 score file(s) are there but cannot be read and are left out" in out and "fav2seed55_r528/fx12test/garment.json" in out
    by = {x["seed"]: x for x in res["seeds"]}
    # seed 55 on the fresh clips has no garment numbers: its colour gain is not there, and nothing is made up for it
    assert by[54]["fresh"]["score"] == pytest.approx(0.01, abs=1e-6) and "colour" not in by[55]["fresh"]["areas"]
    # a whole run without such files says nothing
    g.unlink()
    res, out = _run(yard, capsys)
    assert "unreadable" not in res and "YARD_NOTE" not in out

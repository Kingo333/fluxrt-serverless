"""fxreal7 (owner, 2026-11-17): train favourite #2 itself, at the strength it was made at.

Favourite #2 = 0.5 x fxreal3 + 0.5 x fxreal4 turned down to strength 0.65 on the live model. fxreal5, fxreal6,
fxreal6b and fxreal6c all started from that file and were judged at full strength; all landed about 7% behind
from the first save on. fxreal7 un-scales favourite #2 before training, trains without the earlier-frame input,
and judges every save turned back down to 0.65. No GPU here."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
SCRIPT = ROOT / "training/fluxrt_lora/fx6b_continue.sh"


def _lora(rng, rank, layers=3, din=40, dout=30, scale=0.1):
    sd = {}
    for i in range(layers):
        sd[f"transformer.blocks.{i}.attn.to_q.lora_A.weight"] = rng.normal(0, scale, (rank, din)).astype(np.float32)
        sd[f"transformer.blocks.{i}.attn.to_q.lora_B.weight"] = rng.normal(0, scale, (dout, rank)).astype(np.float32)
    return sd


def _merge(parts, weights):
    """What merge_lora writes: A stacked along rank, B scaled and stacked along rank."""
    out = {}
    for k in parts[0]:
        if ".lora_A." in k:
            out[k] = np.concatenate([p[k] for p in parts], 0)
        else:
            out[k] = np.concatenate([p[k] * w for p, w in zip(parts, weights)], 1)
    return out


def _dw(sd):
    from fluxrt_lora.lora_geometry import pair_layers
    return np.concatenate([(b @ a).ravel() for a, b in pair_layers(sd).values()])


# ---------------------------------------------------------------- how strong a file is against favourite #2
def test_geometry_reads_strength_new_part_and_direction_exactly():
    from fluxrt_lora.lora_geometry import geometry, inner, pair_layers
    rng = np.random.default_rng(0)
    f3, f4 = _lora(rng, 8), _lora(rng, 8)
    fav = _merge([f3, f4], [0.5 * 0.65, 0.5 * 0.65])                      # favourite #2: the mix at strength 0.65
    ref = pair_layers(fav)
    # the per-layer trace formula is the plain inner product of the two weight changes
    want = float(_dw(f3) @ _dw(fav))
    assert abs(inner(pair_layers(f3), ref) - want) < 1e-9 * abs(want)
    # favourite #2 before it was turned down reads 1 / 0.65, with nothing new and the same direction
    un = {k: (v / 0.65 if ".lora_B." in k else v) for k, v in fav.items()}
    g = geometry(ref, pair_layers(un))
    assert abs(g["strength"] - 1 / 0.65) < 1e-6 and g["new_part"] < 1e-4 and abs(g["cosine"] - 1) < 1e-9
    # favourite #2 plus something unrelated: strength stays about 1, the new part shows, files of other ranks work
    plus = _merge([fav, _lora(rng, 16, scale=0.05)], [1.0, 1.0])
    g = geometry(ref, pair_layers(plus))
    assert abs(g["strength"] - 1) < 0.2 and g["new_part"] > 0.1 and g["cosine"] < 0.99
    # turning a file down by 0.65 turns its strength down by 0.65
    low = {k: (v * 0.65 if ".lora_B." in k else v) for k, v in plus.items()}
    assert abs(geometry(ref, pair_layers(low))["strength"] - 0.65 * g["strength"]) < 1e-6
    # alpha / rank of either file scales its change
    assert abs(geometry(ref, pair_layers(un), 1.0, 0.65)["strength"] - 1.0) < 1e-6


def test_unscale_then_rescale_gives_favourite_2_back():
    """fxreal7's start (favourite #2 / 0.65) turned down to 0.65 is favourite #2 itself: step 0 is the live model."""
    from fluxrt_lora.lora_geometry import geometry, pair_layers
    rng = np.random.default_rng(1)
    fav = _merge([_lora(rng, 8), _lora(rng, 8)], [0.325, 0.325])
    up = round(1 / 0.65, 6)                                                # what the script passes to merge_lora
    start = {k: (v * np.float32(up) if ".lora_B." in k else v) for k, v in fav.items()}
    back = {k: (v * np.float32(0.65) if ".lora_B." in k else v) for k, v in start.items()}
    assert np.abs(_dw(back) - _dw(fav)).max() < 1e-6 * np.abs(_dw(fav)).max()
    assert abs(geometry(pair_layers(fav), pair_layers(back))["strength"] - 1) < 1e-5


# ---------------------------------------------------------------- kiosk-weighted pick, per-area lines, chance
def _entry(root: Path, sid: str, step: int, name: str, kind: str, label: str, sets: dict):
    """sets: {set: {"colour": x, "fabric": x, "likeness": x, "drift": x, "gate": bool}} (favourite #2: 12, 0.6, 0.5, 1.0)."""
    d = root / sid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"step": step, "name": name, "weights": f"/w/{sid}", "sha256": "x",
                                             "kind": kind, "label": label}))
    for s, v in sets.items():
        sd = d / s
        sd.mkdir()
        rows = [{"metric": "colour_de", "model": name, "baseline": "fav2", "mean": v.get("colour", 12.0), "baseline_mean": 12.0},
                {"metric": "fabric", "model": name, "baseline": "fav2", "mean": v.get("fabric", 0.6), "baseline_mean": 0.6},
                {"metric": "likeness", "model": name, "baseline": "fav2", "mean": v.get("likeness", 0.5), "baseline_mean": 0.5}]
        (sd / "garment.json").write_text(json.dumps({"scorecard": rows}))
        face = {"face_match": 0.93, "skin_de": 3.0, "face_drift": v.get("drift", 1.0), "texture": 1.0}
        (sd / "face_skin.json").write_text(json.dumps({"average": {name: face, "fav2": dict(face, face_drift=1.0)},
                                                       "verdict": {name: {"pass": v.get("gate", False)}}}))
        (sd / "preserve_check.json").write_text(json.dumps({"verdict": {name: {"pass": v.get("gate", False)}}}))
        (sd / "stability.json").write_text(json.dumps({"c": {name: {"wobble_garment": 0.02, "sharpness": 1.0, "colour_wander": 1.0},
                                                             "fav2": {"wobble_garment": 0.02, "sharpness": 1.0, "colour_wander": 1.0}}}))
    (d / ".scored").touch()


KIOSK = "fx6test=1,dress10=1,cases5=0.5"
SETS3 = ("fx6test", "dress10", "cases5")


def test_areas_line_shows_where_a_model_gains_and_loses(tmp_path, capsys):
    from fluxrt_lora.pick_best_save import main
    ev = tmp_path / "ev"
    # better colour everywhere, worse face steadiness on the dresses only
    _entry(ev, "a", 500, "m500", "save", "fxreal7 step 500 at 0.65",
           {"fx6test": {"colour": 9.0, "gate": True}, "dress10": {"colour": 9.0, "drift": 2.0, "gate": True},
            "cases5": {"colour": 9.0, "gate": True}})
    r = main([str(ev), "--baseline", "fav2", "--weights", KIOSK])
    areas = r["saves"][0]["areas"]
    assert abs(areas["colour"] - 0.25) < 1e-6 and areas["fabric"] == 0 and areas["face and skin"] < 0
    # face steadiness lost 50% (capped) on one of 2.5 set weights, as one of four face measures
    assert abs(areas["face and skin"] - (-0.5 / 4) * (1 / 2.5)) < 1e-4
    out = capsys.readouterr().out
    assert "AREAS fxreal7 step 500 at 0.65: colour +25.0%" in out and "face and skin -5.0%" in out
    assert areas["length"] is None and "length" not in out.split("AREAS", 1)[1].splitlines()[0]


def test_kiosk_weights_put_the_studio_sets_first(tmp_path):
    from fluxrt_lora.pick_best_save import main
    ev = tmp_path / "ev"
    studio = {"fx6test": {"colour": 10.0}, "dress10": {"colour": 10.0}, "cases5": {"colour": 16.0}}
    phone = {"fx6test": {"colour": 13.0}, "dress10": {"colour": 13.0}, "cases5": {"colour": 8.0}}
    _entry(ev, "a", 1, "studio", "save", "good in the studio", studio)
    _entry(ev, "b", 2, "phone", "save", "good on phone clips", phone)
    assert main([str(ev), "--baseline", "fav2", "--weights", KIOSK])["best"]["name"] == "studio"
    assert main([str(ev), "--baseline", "fav2", "--weights", "cases5=2,fx6test=1,dress10=1"])["best"]["name"] == "phone"


def test_chance_entry_sets_the_bar_and_is_never_the_pick(tmp_path, capsys):
    from fluxrt_lora.pick_best_save import main
    ev = tmp_path / "ev"
    ok = {s: {"gate": True} for s in SETS3}
    # favourite #2 with another seed: 0.7% off by chance (one set +1.7%), and it fails its own GATE on one set
    _entry(ev, "aa", -300, "fav2seed53", "aa", "favourite #2, another random seed",
           {"fx6test": {"colour": 10.0, "gate": False}, "dress10": {"gate": True}, "cases5": {"gate": True}})
    # a model 0.2% better: inside chance -> not a candidate, even with every GATE passed
    _entry(ev, "s1", 500, "small", "save", "small gain", {s: dict(ok[s], colour=11.8) for s in SETS3})
    r = main([str(ev), "--baseline", "fav2", "--weights", KIOSK])
    out = capsys.readouterr().out
    assert r["best"]["name"] == "small" and r["chance"]["gate_sets"] == 2 and abs(r["chance"]["size"] - 0.0067) < 2e-4
    assert "CHANCE favourite #2 with another random seed" in out
    assert "VERDICT nothing beats favourite #2 beyond chance: keep favourite #2 live" in out
    # a model clearly better that passes 2 of 3 GATEs (as many as favourite #2 against itself): a candidate
    _entry(ev, "s2", 1000, "big", "save", "big gain",
           {"fx6test": {"colour": 8.0, "gate": False}, "dress10": {"colour": 8.0, "gate": True}, "cases5": {"colour": 8.0, "gate": True}})
    r = main([str(ev), "--baseline", "fav2", "--weights", KIOSK])
    out = capsys.readouterr().out
    assert r["best"]["name"] == "big" and "VERDICT big gain beats favourite #2 beyond chance" in out
    assert "a candidate for a kiosk test" in out
    # passing more GATEs than favourite #2 does against itself earns no extra rank: the score decides
    assert all(x["kind"] != "aa" for x in [r["best"]])
    # without a chance entry the old rule holds: every GATE and a better score
    (ev / "aa" / ".scored").unlink()
    r = main([str(ev), "--baseline", "fav2", "--weights", KIOSK])
    assert r["chance"] is None and r["best"]["name"] == "small"
    assert "VERDICT small gain beats favourite #2 (GATE on every set and a better score)" in capsys.readouterr().out


def test_chance_size_does_not_cancel_between_sets():
    from fluxrt_lora.pick_best_save import chance_size
    aa = {"score": 0.0, "per_set": {"fx6test": {"score": 0.02}, "dress10": {"score": -0.02}, "cases5": {"score": None}}}
    w = {"fx6test": 1.0, "dress10": 1.0, "cases5": 0.5}
    assert abs(chance_size(aa, w) - (2 * 0.02 ** 2) ** 0.5 / 2) < 1e-9        # 1.4%, not 0
    assert chance_size({"score": -0.03, "per_set": {"fx6test": {"score": -0.03}}}, w) == 0.03

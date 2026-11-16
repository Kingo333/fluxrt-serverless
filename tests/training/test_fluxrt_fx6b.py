"""fxreal6b (owner, 2026-11-16): carry on from step 5800 at half the learning rate, check every save, keep the best;
dress length in the garment scores; never past the cap or the RunPod balance. No GPU."""

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


def _dress(h, w, top, length, waist, skirt):
    m = np.zeros((h, w), bool)
    c = w // 2
    for y in range(top, top + length):
        f = (y - top) / length
        half = waist // 2 if 0.15 <= f <= 0.55 else skirt // 2
        m[y, c - half: c + half] = True
    return m


# ---------------------------------------------------------------- dress length
def test_length_proportion_reads_length_over_waist_and_skips_cut_off_dresses():
    from eval.garment_report import length_proportion
    photo = _dress(400, 300, 20, 300, 60, 120)
    same = _dress(512, 288, 40, 200, 40, 80)            # same proportions, smaller and elsewhere
    short = _dress(512, 288, 40, 150, 40, 80)           # a quarter shorter
    assert abs(length_proportion(photo) - length_proportion(same)) < 1e-9
    assert length_proportion(short) < 0.8 * length_proportion(photo)
    cut = np.zeros((512, 288), bool); cut[40:510, 100:180] = True
    assert length_proportion(cut) is None                # hem at the frame's bottom edge: length unknown
    head_cut = np.zeros((512, 288), bool); head_cut[0:300, 100:180] = True
    assert length_proportion(head_cut) is None
    shoe = same.copy(); shoe[480:500, 100:120] = True    # a shoe below the dress does not make it longer
    assert length_proportion(shoe) == length_proportion(same)


def test_clip_metrics_gives_dresses_a_length_score_and_tops_none():
    import eval.garment_report as gr
    H, W, T = 512, 288, 6
    ref = np.full((400, 300, 3), 240, np.uint8)
    ref_mask = _dress(400, 300, 20, 300, 60, 120)
    ref[ref_mask] = (30, 30, 30)
    cams, cats_cam, outs, cats = [], [], {}, {}
    for t in range(T):
        cam = np.full((H, W, 3), 200, np.uint8)
        cc = np.zeros((H, W), np.uint8); cc[30:470, 100:190] = gr.CLOTHES; cc[10:30, 130:160] = gr.FACE
        cams.append(cam); cats_cam.append(cc)
        for name, length in (("right", 200), ("short", 150)):
            m = _dress(H, W, 40, length, 40, 80)
            o = cam.copy(); o[m] = (30, 30, 30)
            c = np.zeros((H, W), np.uint8); c[m] = gr.CLOTHES; c[10:30, 130:160] = gr.FACE
            outs.setdefault(name, []).append(o); cats.setdefault(name, []).append(c)
    right = gr.clip_metrics(cams, outs["right"], cats_cam, cats["right"], ref, ref_mask, "dresses")
    short = gr.clip_metrics(cams, outs["short"], cats_cam, cats["short"], ref, ref_mask, "dresses")
    assert right["length_err"] < 0.01 and abs(right["length_ratio"] - 1) < 0.01
    assert short["length_ratio"] < 0.8 and short["length_err"] > 0.2
    top = gr.clip_metrics(cams, outs["short"], cats_cam, cats["short"], ref, ref_mask, "tops")
    assert "length_err" not in top
    card = gr.scorecard({"c": {"fav2": short, "new": right}}, ["fav2", "new"], "fav2")
    assert {(r["priority"], r["metric"]): r["better"] for r in card}[("LENGTH", "length_err")] == 1


# ---------------------------------------------------------------- picking the best save
def _save(root: Path, sid: str, step: int, name: str, sets: dict):
    d = root / sid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"step": step, "name": name, "weights": f"/w/{sid}", "sha256": "x"}))
    for s, (colour, gate) in sets.items():
        sd = d / s
        sd.mkdir()
        (sd / "garment.json").write_text(json.dumps({"scorecard": [
            {"metric": "colour_de", "model": name, "baseline": "fav2", "mean": colour, "baseline_mean": 12.0},
            {"metric": "fabric", "model": name, "baseline": "fav2", "mean": 0.6, "baseline_mean": 0.6}]}))
        (sd / "face_skin.json").write_text(json.dumps({"average": {name: {"face_match": 0.93, "skin_de": 3.0, "face_drift": 0.4,
                                                                          "texture": 1.0},
                                                                   "fav2": {"face_match": 0.93, "skin_de": 3.0, "face_drift": 0.4,
                                                                            "texture": 1.0}},
                                                       "verdict": {name: {"pass": gate}}}))
        (sd / "preserve_check.json").write_text(json.dumps({"verdict": {name: {"pass": gate}}}))
        (sd / "stability.json").write_text(json.dumps({"clip1": {name: {"wobble_garment": 0.02, "sharpness": 1.0,
                                                                        "colour_wander": 1.0},
                                                                 "fav2": {"wobble_garment": 0.02, "sharpness": 1.0,
                                                                          "colour_wander": 1.0}}}))
    (d / ".scored").touch()


def test_pick_best_save_ranks_gate_first_then_score_with_real_cases_counting_double(tmp_path, capsys):
    from fluxrt_lora.pick_best_save import main
    ev = tmp_path / "fx6b"
    # step 5800 itself: worse colour everywhere
    _save(ev, "s0", 0, "fx6s5800", {"cases5": (16.0, False), "fx6test": (13.0, False), "dress10": (10.0, False)})
    # 500: much better on the studio sets, worse on the real cases (which count double)
    _save(ev, "s500xa", 500, "fx6b500", {"cases5": (14.0, False), "fx6test": (9.0, False), "dress10": (8.0, False)})
    # 1000: better on the real cases, a little worse on the studio sets
    _save(ev, "s1000xb", 1000, "fx6b1000", {"cases5": (10.0, False), "fx6test": (12.5, False), "dress10": (12.5, False)})
    r = main([str(ev), "--baseline", "fav2"])
    scores = {x["step"]: x["score"] for x in r["saves"]}
    assert scores[0] < scores[500] and r["best"]["step"] == 1000
    out = capsys.readouterr().out
    assert "SAVE step 0 " in out and "BEST step 1000" in out and json.loads((ev / "best.json").read_text())["best"]["step"] == 1000
    # a save that passes the GATE on a set beats a better-scoring one that passes nowhere
    _save(ev, "s1500xc", 1500, "fx6b1500", {"cases5": (11.9, True), "fx6test": (12.0, False), "dress10": (12.0, False)})
    assert main([str(ev), "--baseline", "fav2"])["best"]["step"] == 1500
    # an unfinished save (no .scored marker) is not judged
    (ev / "s2000xd").mkdir(); (ev / "s2000xd" / "meta.json").write_text(json.dumps({"step": 2000, "name": "n"}))
    assert 2000 not in {x["step"] for x in main([str(ev), "--baseline", "fav2"])["saves"]}


def test_gain_is_relative_capped_and_direction_aware():
    from fluxrt_lora.pick_best_save import gain
    assert abs(gain(9.0, 12.0, True, 1.0) - 0.25) < 1e-9          # colour 12 -> 9: 25% better
    assert gain(0.0, 12.0, True, 1.0) == 0.5                      # capped
    assert gain(0.90, 0.93, False, 0.05) < 0 and gain(None, 1.0, True, 1.0) is None


# ---------------------------------------------------------------- the Pod script
def test_fx6b_script_continues_from_step_5800_at_half_rate_into_a_new_run_without_hijab():
    sh = SCRIPT.read_text()
    for s in ('LR="${AZ_LR:-2.5e-5}"', 'CAP="${AZ_CAP:-55}"', 'STEPS="${AZ_MAX_STEPS:-2000}"', 'EVERY="${AZ_SAVE_EVERY:-500}"',
              'START_SHA="${AZ_START_SHA:-0ab25649}"', "EXP=fxreal6b", "AZ_INIT_LORA=$START_DIR", 'ln -s "$DATA_SRC" "$DATA"',
              'SETS="${AZ_EVAL_SETS:-fx6test cases5 dress10}"', "fluxrt_lora.pick_best_save", "AZ_CKPT_INTERVAL=$EVERY",
              "AZ_COLOUR_WEIGHT=${AZ_COLOUR_WEIGHT:-0.5}", "AZ_TEXTURE_WEIGHT=${AZ_TEXTURE_WEIGHT:-0.3}",
              'mv "$OUT/checkpoint-$n" "$ASIDE/"', 'RESUME_FROM="${AZ_RESUME_FROM:-best}"', "WRONG_START_WEIGHTS"):
        assert s in sh, s
    code = "\n".join(line for line in sh.splitlines() if not line.lstrip().startswith("#"))
    assert "hijab" not in code.lower()                                     # no hijab set, no hijab weighting
    assert "rm -rf" not in sh and 'rm -f "$EV/.training_done"' in sh   # saves are moved aside, never deleted
    assert 'SRC_OUT="$AZ_ROOT/outputs/fluxrt_lora/fxreal6"' in sh and 'OUT="$AZ_ROOT/outputs/fluxrt_lora/$EXP"' in sh
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_fx6b_money_limit_is_the_smaller_of_cap_and_balance():
    text = SCRIPT.read_text()
    start = text.index("allowed_minutes() {")
    end = text.index("ALLOWED=$(allowed_minutes")
    fn = text[start:end]

    def run(*args):
        r = subprocess.run(["bash", "-c", fn + 'allowed_minutes "$@"', "x", *args], capture_output=True, text=True)
        return int(r.stdout)
    assert run("55", "37.08", "2.79", "11.35", "1.50") == int((11.35 - 1.5) / 2.79 * 60)     # the balance binds
    assert run("55", "37.08", "2.79", "100", "1.50") == int((55 - 37.08) / 2.79 * 60)       # the cap binds
    assert run("55", "37.08", "2.79", "", "1.50") == int((55 - 37.08) / 2.79 * 60)          # no balance given
    assert run("45", "47", "2.79", "", "1.50") == 0


def test_fx6b_refuses_to_start_without_enough_money(tmp_path):
    env = {"PATH": os.environ["PATH"], "AZ_REF": "abc", "AZ_SPENT": "37.08", "AZ_BALANCE": "3", "AZ_NO_STOP": "1",
           "HOME": str(tmp_path)}
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode != 0 and "top up" in r.stdout and "STOPPED" in r.stdout

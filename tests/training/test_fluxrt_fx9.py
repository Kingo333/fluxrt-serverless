"""fxreal9 (owner, 2026-11-21): two short runs from fxreal8's step 1000, compared with fxreal8's own continuation at
equal steps. No GPU and no torch here: the wiring and the picker's new lines. The new terms on real tensors:
test_fluxrt_fx9_torch.py. The whole Pod script with both arms: test_fluxrt_fx7_dryrun.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
SCRIPT = ROOT / "training/fluxrt_lora/fx6b_continue.sh"
ARMS = ROOT / "training/fluxrt_lora/fx9_arms.sh"
PATCH = ROOT / "training/patches/flux2-klein-fx9.patch"
KIOSK = "fx6test=1,dress10=1,cases5=0.5"


def test_the_trainer_wrapper_applies_the_fx9_patch_last_and_passes_its_options():
    s = (ROOT / "training/fluxrt_lora/run_train.sh").read_text()
    applies = [l.split("flux2-klein-")[1].split(".patch")[0] for l in s.splitlines() if l.startswith('git -C "$SRC" apply')]
    assert applies[-3:] == ["fx8", "fx9", "resume"]
    assert '$( [ "${AZ_RESUME_SKIP:-0}" = "1" ] && echo --resume_skip_seen )' in s
    for opt in ('${AZ_PAIR_FLOOR:+--pair_floor="$AZ_PAIR_FLOOR"}', '${AZ_DETAIL_FLOOR:+--detail_floor="$AZ_DETAIL_FLOOR"}',
                '${AZ_TEXTURE_FLOOR:+--texture_floor="$AZ_TEXTURE_FLOOR"}',
                '${AZ_KEEP_SPREAD:+--keep_spread_weight="$AZ_KEEP_SPREAD"}'):
        assert opt in s, opt


def test_the_patch_is_off_unless_asked_and_reuses_the_frozen_models_prediction():
    p = PATCH.read_text()
    for arg in ("--pair_floor", "--detail_floor", "--texture_floor", "--keep_spread_weight"):
        i = p.index(f'"{arg}"')
        assert "default=0.0" in p[i:i + 120], arg
    # one more forward pass is never added: the keep-spread term reads the prediction the reference term already made
    assert "+                if args.ref_weight > 0 or args.keep_spread_weight > 0:" in p
    assert p.count("transformer(") == 0
    assert "+                    if _az_ref_pred is not None and args.ref_weight > 0:" in p
    assert 'print(f"FX9_TERMS same-clip colour floor' in p and '_az_logs["keep"]' in p
    assert "--ref_weight and --keep_spread_weight need --frozen_init_lora" in p


def test_the_two_arms():
    s = ARMS.read_text()
    assert 'a) NAME="protect"; TERMS=(AZ_PAIR_FLOOR=1.0 AZ_KEEP_SPREAD=1.0);;' in s
    assert 'b) NAME="reach";   TERMS=(AZ_PAIR_FLOOR=1.0 AZ_KEEP_SPREAD=1.0 AZ_TEXTURE_FLOOR=0.5 AZ_DETAIL_FLOOR=0.5);;' in s
    assert 'BRANCH="${AZ_BRANCH_FROM:-fxreal8:1000}"' in s and 'STEPS="${AZ_MAX_STEPS:-2000}"' in s and 'EVERY="${AZ_SAVE_EVERY:-500}"' in s
    assert 'AZ_TRAIN_FROM=fav2_addon AZ_EXP="fxreal9$arm" AZ_BRANCH_FROM="$BRANCH"' in s
    # only the last arm stops the Pod; a script that cannot be downloaded stops it at once
    assert "the Pod keeps running: the next arm starts now" in s and 'echo "FX9 STOPPED: the run script could not be downloaded"; stop_pod; exit 1' in s


def test_the_pod_script_can_start_a_run_from_another_runs_save():
    s = SCRIPT.read_text()
    assert 'BRANCH_FROM="${AZ_BRANCH_FROM:-}"' in s and "NO_BRANCH_POINT" in s and "BRANCH_MISMATCH" in s
    assert '[ "$n" -le "$BRANCH_STEP" ] && continue' in s            # the other run's save is not checked again
    assert 'cp "$BSRC/fx8_steady_weight.json" "$OUT/fx8_steady_weight.json"' in s
    assert "for v in AZ_PAIR_FLOOR AZ_DETAIL_FLOOR AZ_TEXTURE_FLOOR AZ_KEEP_SPREAD; do" in s
    assert 'grep -E "^(FABRIC|COMPARE)" "$EV/pick.log"' in s and "^FX9_" in s


# ---------------------------------------------------------------- the picker: equal steps, fabric direction
def _entry(ev: Path, sid: str, step: int, name: str, label: str, colour: float, ratio: float | None):
    d = ev / sid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"step": step, "name": name, "weights": f"/w/{sid}", "sha256": "x",
                                             "kind": "save", "label": label}))
    for s in ("fx6test", "dress10", "cases5"):
        sd = d / s
        sd.mkdir()
        a, b = {"colour_de": colour}, {"colour_de": 12.0}
        if ratio is not None:
            a["fabric_ratio"], b["fabric_ratio"] = ratio, 0.9
        rows = [{"metric": "colour_de", "model": name, "baseline": "fav2", "mean": colour, "baseline_mean": 12.0}]
        (sd / "garment.json").write_text(json.dumps({"scorecard": rows, "items": {"c1": {name: a, "fav2": b},
                                                                                  "c2": {name: a, "fav2": b}}}))
        face = {"face_match": 0.93, "skin_de": 3.0, "face_drift": 1.0, "texture": 1.0}
        (sd / "face_skin.json").write_text(json.dumps({"average": {name: face, "fav2": face}, "verdict": {name: {"pass": True}}}))
        (sd / "preserve_check.json").write_text(json.dumps({"verdict": {name: {"pass": True}}}))
        (sd / "stability.json").write_text("{}")
    (d / ".scored").touch()


def test_runs_with_saves_at_the_same_step_are_put_side_by_side(tmp_path, capsys):
    from fluxrt_lora.pick_best_save import main
    fxa = tmp_path / "fxa"
    ev = fxa / "fx6b"
    fxa.mkdir()
    for s in ("fx6test", "dress10", "cases5"):
        (fxa / f"{s}.json").write_text(json.dumps([{"key": "c1", "category": "dresses"}, {"key": "c2", "category": "tops"}]))
    _entry(ev, "fx8s1000", 1000, "fx81000", "fxreal8 step 1000 at 0.65", 11.4, None)       # an older scoring: no direction
    _entry(ev, "fx8s1500", 1500, "fx81500", "fxreal8 step 1500 at 0.65", 11.1, 0.80)
    _entry(ev, "fx9as1500", 1500, "fx9a1500", "fxreal9a step 1500 at 0.65", 10.8, 0.88)
    _entry(ev, "fx9bs1500", 1500, "fx9b1500", "fxreal9b step 1500 at 0.65", 10.5, 1.10)
    res = main([str(ev), "--baseline", "fav2", "--weights", KIOSK])
    out = capsys.readouterr().out
    compare = [l for l in out.splitlines() if l.startswith("COMPARE")]
    assert [l.split(":")[0] for l in compare] == ["COMPARE step 1500, fxreal8", "COMPARE step 1500, fxreal9a",
                                                   "COMPARE step 1500, fxreal9b"]            # step 1000 has one run only
    # (these made-up records hold the face numbers per set only, so the per-garment figures are the colour alone)
    assert compare[2].startswith("COMPARE step 1500, fxreal9b: score +2.5%, dresses +12.5%, tops +12.5%, colour +12.5%")
    # the direction of the fabric score: flatter or more contrasty than the photo, next to favourite #2
    assert "FABRIC fxreal8 step 1500 at 0.65: the garment's light-dark spread is x0.80 of the photo's (favourite #2 x0.90" in out
    assert "FABRIC fxreal9b step 1500 at 0.65: the garment's light-dark spread is x1.10 of the photo's" in out
    assert "FABRIC fxreal8 step 1000" not in out
    rows = {r["label"]: r for r in res["saves"]}
    assert rows["fxreal8 step 1000 at 0.65"]["fabric_ratio"] is None
    assert tuple(rows["fxreal9a step 1500 at 0.65"]["fabric_ratio"]) == (0.88, 0.9)


def test_the_garment_scorer_keeps_the_direction_of_the_fabric_score():
    s = (ROOT / "training/eval/garment_report.py").read_text()
    assert 'res["fabric_ratio"] = round(float(s_out / s_ref), 4)' in s
    assert 'res["fabric"] = round(abs(float(np.log(s_out / s_ref))), 4)' in s        # the score itself is unchanged


def test_a_restarted_run_can_carry_on_in_the_data_instead_of_replaying_it():
    p = (ROOT / "training/patches/flux2-klein-resume.patch").read_text()
    assert '"--resume_skip_seen"' in p and 'action="store_true"' in p                      # off unless asked
    assert "+        _az_resume_skip = (initial_global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps" in p
    assert "+            if epoch == first_epoch and step < _az_resume_skip:" in p and "+                continue" in p
    assert 'print(f"RESUME_SKIP carrying on after the' in p
    s = SCRIPT.read_text()
    assert 'if [ "${AZ_RESUME_SKIP:-0}" = "1" ]; then FX8_ENV="$FX8_ENV AZ_RESUME_SKIP=1"' in s and "^RESUME_SKIP" in s


def test_the_yardstick_script_never_trains_or_deletes_and_keeps_to_the_money():
    s = (ROOT / "training/fluxrt_lora/fxb_yard.sh").read_text()
    assert "run_train.sh" not in s and "rm -rf" not in s and "endpoint" in s.split("set -uo pipefail")[0]
    assert 'out_of_money() { [ "$(left_min)" -le "$END_MIN" ]; }' in s and s.count("out_of_money &&") >= 4
    assert 'case "$FAV_SUM" in "$FAV2_SHA"*) ;; *) fail "WRONG_FAVOURITE2' in s
    assert "trap stop_pod EXIT" in s and 'runpodctl stop pod "${RUNPOD_POD_ID:-}"' in s

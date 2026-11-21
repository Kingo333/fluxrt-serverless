"""By garment type (owner, 2026-11-21): the result is judged on dresses first, and on shirts and trousers too.
The picker works the same score out again for the clips of one type, measures chance per type from favourite #2
with other random seeds, and eval.garment_mix builds one review set per type from videos that are already there.
No GPU: the real picker and the real set builder on made-up numbers. The whole Pod script: test_fluxrt_fx7_dryrun.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
SCRIPT = ROOT / "training/fluxrt_lora/fx6b_continue.sh"
KIOSK = "fx6test=1,dress10=1,cases5=0.5"
# fx6test: two tops, two dresses, two bottoms; dress10: three dresses; cases5: one of each (the real lists are longer)
LISTS = {"fx6test": [("t1", "tops"), ("t2", "tops"), ("d1", "dresses"), ("d2", "dresses"), ("b1", "bottoms"), ("b2", "bottoms")],
         "dress10": [("e1", "dresses"), ("e2", "dresses"), ("e3", "dresses")],
         "cases5": [("1_x", "tops"), ("2_x", "dresses"), ("3_x", "bottoms")]}


def _lists(fxa: Path, lists=LISTS):
    fxa.mkdir(parents=True, exist_ok=True)
    for s, clips in lists.items():
        (fxa / f"{s}.json").write_text(json.dumps([{"key": k, "category": c, "ref": f"/x/{k}/ref.jpg"} for k, c in clips]))


def _entry(ev: Path, sid: str, step: int, name: str, kind: str, label: str, colour: dict, wobble: dict | None = None,
           drift: dict | None = None, gate: bool = True):
    """colour / wobble / drift: {garment type: the model's number} (favourite #2: 12.0, 0.02, 1.0 on every clip)."""
    d = ev / sid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"step": step, "name": name, "weights": f"/w/{sid}", "sha256": "x",
                                             "kind": kind, "label": label}))
    for s, clips in LISTS.items():
        sd = d / s
        sd.mkdir()
        items = {k: {name: {"colour_de": colour.get(c, 12.0)}, "fav2": {"colour_de": 12.0}} for k, c in clips}
        mean = sum(v[name]["colour_de"] for v in items.values()) / len(items)
        rows = [{"metric": "colour_de", "model": name, "baseline": "fav2", "mean": mean, "baseline_mean": 12.0}]
        (sd / "garment.json").write_text(json.dumps({"scorecard": rows, "items": items}))
        face = {"face_match": 0.93, "skin_de": 3.0, "face_drift": 1.0, "texture": 1.0}
        per = {k: {name: dict(face, face_drift=(drift or {}).get(c, 1.0)), "fav2": face} for k, c in clips}
        avg = dict(face, face_drift=sum(v[name]["face_drift"] for v in per.values()) / len(per))
        (sd / "face_skin.json").write_text(json.dumps({"per_clip": per, "average": {name: avg, "fav2": face},
                                                       "verdict": {name: {"pass": gate}}}))
        (sd / "preserve_check.json").write_text(json.dumps({"verdict": {name: {"pass": gate}}}))
        st = {k: {name: {"wobble_garment": (wobble or {}).get(c, 0.02), "sharpness": 1.0, "colour_wander": 1.0},
                  "fav2": {"wobble_garment": 0.02, "sharpness": 1.0, "colour_wander": 1.0}} for k, c in clips}
        (sd / "stability.json").write_text(json.dumps(st))
    (d / ".scored").touch()


def test_garment_types_from_the_words_the_lists_use():
    from fluxrt_lora.pick_best_save import garment_type
    assert [garment_type(x) for x in ("tops", "upper", "Outerwear", "shirt")] == ["tops"] * 4
    assert [garment_type(x) for x in ("dresses", "dress", "abaya", "jumpsuit")] == ["dresses"] * 4
    assert [garment_type(x) for x in ("bottoms", "lower", "trousers", "skirt")] == ["bottoms"] * 4
    assert garment_type(None) is None and garment_type("") is None and garment_type("shoes") is None


def test_the_score_is_split_by_garment_type_and_matches_the_whole_when_put_together(tmp_path, capsys):
    from fluxrt_lora.pick_best_save import main
    fxa = tmp_path / "fxa"
    ev = fxa / "fx6b"
    _lists(fxa)
    # better colour on dresses, the same on tops, worse on bottoms; shakier on bottoms only
    _entry(ev, "fx8s750", 750, "fx8", "save", "fxreal8 step 750 at 0.65",
           colour={"dresses": 10.8, "tops": 12.0, "bottoms": 13.2}, wobble={"bottoms": 0.03})
    res = main([str(ev), "--baseline", "fav2", "--weights", KIOSK])
    out = capsys.readouterr().out
    g = res["saves"][0]["garments"]
    assert set(g) == {"dresses", "tops", "bottoms"}
    assert (g["dresses"]["clips"], g["tops"]["clips"], g["bottoms"]["clips"]) == (6, 3, 3)
    # these made-up records hold 8 measures per set (colour; 4 of face and skin; 3 of steadiness); only colour
    # (and the garment wobble on bottoms) differ from favourite #2
    assert g["dresses"]["score"] == pytest.approx((1.2 / 12) / 8, abs=1e-4)
    assert g["tops"]["score"] == pytest.approx(0.0, abs=1e-9)
    assert g["bottoms"]["score"] == pytest.approx((-1.2 / 12 - 0.01 / 0.02) / 8, abs=1e-4)
    assert g["dresses"]["areas"]["colour"] == pytest.approx(0.1) and g["bottoms"]["areas"]["colour"] == pytest.approx(-0.1)
    # the flicker area here: garment wobble (-50%) and colour wander (the same) -> -25%
    assert g["bottoms"]["areas"]["flicker"] == pytest.approx(-0.25) and g["dresses"]["areas"]["flicker"] == pytest.approx(0.0)
    assert "GARMENT fxreal8 step 750 at 0.65: dresses +1.2% (6 clips) | tops +0.0% (3 clips) | bottoms -7.5% (3 clips)" in out
    assert "GARMENT_AREAS fxreal8 step 750 at 0.65, bottoms: colour -10.0%" in out
    # the pick itself is unchanged: the same weighted score over the sets as before
    assert res["best"]["label"] == "fxreal8 step 750 at 0.65" and res["saves"][0]["score"] is not None


def test_no_lists_no_split_and_nothing_else_changes(tmp_path, capsys):
    from fluxrt_lora.pick_best_save import main
    ev = tmp_path / "fxa/fx6b"
    _entry(ev, "fx8s750", 750, "fx8", "save", "fxreal8 step 750 at 0.65", colour={"dresses": 10.8})
    res = main([str(ev), "--baseline", "fav2", "--weights", KIOSK])
    out = capsys.readouterr().out
    assert res["saves"][0]["garments"] == {} and "GARMENT" not in out and "SAVE fxreal8 step 750 at 0.65" in out
    # an explicit folder works too
    _lists(tmp_path / "elsewhere")
    res = main([str(ev), "--baseline", "fav2", "--weights", KIOSK, "--clips", str(tmp_path / "elsewhere")])
    assert set(res["saves"][0]["garments"]) == {"dresses", "tops", "bottoms"}


def test_several_chance_seeds_give_the_largest_size_and_the_lowest_gate_bar(tmp_path, capsys):
    from fluxrt_lora.pick_best_save import main
    fxa = tmp_path / "fxa"
    ev = fxa / "fx6b"
    _lists(fxa)
    _entry(ev, "aa53", -300, "seed53", "aa", "favourite #2, another random seed", colour={"dresses": 11.9}, gate=True)
    _entry(ev, "aa54", -301, "seed54", "aa", "favourite #2, random seed 54", colour={"tops": 13.2, "bottoms": 11.4}, gate=False)
    _entry(ev, "fx8s750", 750, "fx8", "save", "fxreal8 step 750 at 0.65", colour={"dresses": 10.0, "tops": 12.6}, gate=False)
    res = main([str(ev), "--baseline", "fav2", "--weights", KIOSK])
    out = capsys.readouterr().out
    ch = res["chance"]
    assert ch["seeds"] == 2 and ch["gate_sets"] == 0                      # seed 54 passes the GATE on no set
    one = max(abs(s["score"]) for s in res["saves"] if s["kind"] == "aa")
    assert ch["size"] >= one - 1e-9
    assert ch["garments"]["tops"] == pytest.approx((1.2 / 12) / 8, abs=1e-4)       # seed 54's tops, the larger one
    assert ch["garments"]["dresses"] == pytest.approx((0.1 / 12) / 8, abs=1e-4)
    assert ch["areas"]["colour"] > 0
    assert "CHANCE favourite #2 with 2 other random seeds: scores" in out and "GATE 3/3, 0/3 sets" in out
    assert "CHANCE_AREAS differences up to these sizes mean nothing: colour" in out
    assert "GARMENT_CHANCE differences up to these sizes mean nothing: dresses 0.1% | tops 1.2% | bottoms 0.6%" in out
    # the candidate: clearly better on dresses, worse on tops but inside chance, the same on bottoms
    assert ("GARMENT_VERDICT fxreal8 step 750 at 0.65 vs favourite #2: dresses +2.1% (better, chance 0.1%) | "
            "tops -0.6% (level, chance 1.2%) | bottoms +0.0% (level, chance 0.6%)") in out
    # it is not asked to pass the GATE on more sets than favourite #2 does against itself
    assert "VERDICT fxreal8 step 750 at 0.65 beats favourite #2 beyond chance" in out


def test_one_chance_seed_keeps_the_old_wording(tmp_path, capsys):
    from fluxrt_lora.pick_best_save import main
    fxa = tmp_path / "fxa"
    _lists(fxa)
    _entry(fxa / "fx6b", "aa53", -300, "seed53", "aa", "favourite #2, another random seed", colour={"dresses": 11.9})
    main([str(fxa / "fx6b"), "--baseline", "fav2", "--weights", KIOSK])
    assert "CHANCE favourite #2 with another random seed: score" in capsys.readouterr().out


# ---------------------------------------------------------------- the review sets per garment type
def _results(root: Path):
    fxa = root / "eval/fxa"
    _lists(fxa)
    for s, clips in LISTS.items():
        for k, _ in clips:
            d = fxa / "results" / s / k
            d.mkdir(parents=True)
            for f in ("input.mp4", "ref.jpg", "out_fav2.mp4", "out_new.mp4", "seg_fav2.npz"):
                (d / f).write_text(f"{s}/{k}/{f}")
    return fxa


def test_review_sets_take_the_clips_in_turn_and_point_at_the_videos_that_are_there(tmp_path, capsys):
    from eval.garment_mix import build, in_turn
    assert in_turn([[1, 2, 3], [], [7]], 3) == [1, 7, 2] and in_turn([[1], [2]], 5) == [1, 2] and in_turn([], 3) == []
    fxa = _results(tmp_path)
    rep = build(tmp_path, ["dress10", "fx6test", "cases5"], per_type=4)
    assert {t: r["clips"] for t, r in rep.items()} == {"dresses": 4, "tops": 3, "bottoms": 3}
    dresses = json.loads((fxa / "g_dresses.json").read_text())
    assert [(d["from_set"], d["key"]) for d in dresses] == [("dress10", "e1"), ("fx6test", "d1"), ("cases5", "2_x"), ("dress10", "e2")]
    assert dresses[0]["case"] == "dress 1 of 4 (test set dress10)" and dresses[0]["category"] == "dresses"
    for t in ("dresses", "tops", "bottoms"):
        for it in json.loads((fxa / f"g_{t}.json").read_text()):
            d = fxa / "results" / f"g_{t}" / it["key"]
            assert (d / "input.mp4").read_text() == f"{it['from_set']}/{it['key']}/input.mp4"
            assert (d / "out_new.mp4").exists() and next(p for p in d.iterdir() if p.stem == "ref").name == "ref.jpg"
    assert "GARMENT_SETS dresses: 4 clips from cases5, dress10, fx6test | tops: 3 clips" in capsys.readouterr().out
    # run again with fewer: the lists and folders are replaced, the clips' own folders are untouched
    build(tmp_path, ["dress10"], per_type=2)
    assert len(json.loads((fxa / "g_dresses.json").read_text())) == 2 and not (fxa / "g_tops.json").exists()
    assert (fxa / "results/fx6test/t1/out_new.mp4").read_text() == "fx6test/t1/out_new.mp4"


def test_two_clips_with_the_same_key_get_their_own_folders(tmp_path):
    from eval.garment_mix import build
    fxa = tmp_path / "eval/fxa"
    _lists(fxa, {"dress10": [("c1", "dresses")], "cases5": [("c1", "dress")]})
    for s in ("dress10", "cases5"):
        d = fxa / "results" / s / "c1"
        d.mkdir(parents=True)
        (d / "input.mp4").write_text(s)
    build(tmp_path, ["dress10", "cases5"])
    items = json.loads((fxa / "g_dresses.json").read_text())
    assert [it["key"] for it in items] == ["c1", "cases5_c1"]
    assert [(fxa / "results/g_dresses" / it["key"] / "input.mp4").read_text() for it in items] == ["dress10", "cases5"]


def test_where_links_are_not_possible_the_videos_are_copied(tmp_path, monkeypatch):
    from eval import garment_mix
    fxa = _results(tmp_path)

    def no_links(self, *a, **k):
        raise OSError("links are not supported here")
    monkeypatch.setattr(Path, "symlink_to", no_links)
    rep = garment_mix.build(tmp_path, ["fx6test"], per_type=1)
    assert rep["tops"]["as"] == ["copy"]
    d = fxa / "results/g_tops/t1"
    assert not d.is_symlink() and {p.name for p in d.iterdir()} == {"input.mp4", "ref.jpg", "out_fav2.mp4", "out_new.mp4"}


# ---------------------------------------------------------------- the wiring in the Pod script
def test_the_pod_script_reports_and_films_each_garment_type():
    s = SCRIPT.read_text()
    assert 'AA_SEEDS="${AZ_AA_SEEDS:-53}"' in s and 'register "aa$sd"' in s and '"--seed $sd"' in s
    assert "1b. by garment type, from the checks already on the volume" in s
    assert 'grep -E "^GARMENT( |_CHANCE)" "$EV/pick.log"' in s and 'grep -E "^GARMENT_VERDICT" "$EV/pick.log"' in s
    assert '-m eval.garment_mix "$AZ_ROOT" --sets $GSETS' in s
    assert 'for t in dresses tops bottoms; do' in s and '"$EV/${PFX}_${t}_best.mp4"' in s
    # the review file takes the new videos with it
    assert '"${PFX}"_*_best.mp4' in s


def test_a_clip_without_a_number_is_left_out_of_that_measure_only(tmp_path):
    from fluxrt_lora.pick_best_save import subset_numbers
    d = tmp_path / "s"
    d.mkdir()
    items = {"a": {"m": {"colour_de": 10.0, "fabric": float("nan")}, "fav2": {"colour_de": 12.0, "fabric": 0.5}},
             "b": {"m": {"colour_de": None, "fabric": 0.4}, "fav2": {"colour_de": 12.0, "fabric": 0.6}},
             "c": {"m": {"colour_de": 11.0}, "fav2": {"colour_de": 13.0}}}
    (d / "garment.json").write_text(json.dumps({"items": items}))
    (d / "stability.json").write_text(json.dumps({"a": {"m": {"sharpness": 2.0}, "fav2": {"sharpness": 1.0}}, "b": "broken"}))
    nums, n = subset_numbers(d, "m", "fav2", {"a", "b"})
    assert n == 2 and nums["colour_de"] == (10.0, 12.0) and nums["fabric"] == (0.4, 0.6)      # clip c is not in the group
    assert nums["sharpness"] == (2.0, 1.0) and nums["face_match"] == (None, None)

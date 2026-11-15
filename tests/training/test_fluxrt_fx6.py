"""fxreal6 tools (owner, 2026-11-15): captions = the live app, frame windows, logo stamping, skin-tone
groups, face and skin measures. Pure functions; no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))


# ---------------------------------------------------------------- captions = the app's live prompt
def test_plain_caption_is_the_apps_plain_wording():
    from data_tools.fluxrt_prompt import fx6_prompt
    # The exact string azyah-shopping's plainWording.test.ts expects (PR #39).
    hint = "A plain black crew neck short sleeve t-shirt with a soft, smooth, matte, cotton-like jersey fabric"
    assert fx6_prompt("top", "top", hint, plain=True) == (
        "Virtual try-on. The person in the camera frame wears the top from the reference image: A plain black crew "
        "neck short sleeve t-shirt with a soft, smooth, matte, cotton-like jersey fabric. "
        "It replaces the current top completely, so nothing of the current top stays visible; the pants or skirt stay as they are. "
        "It is plain: one solid colour all over, with a clean, blank surface. Keep its exact colour, fabric and fit, with short sleeves. "
        "Keep the person's face, hair, body, pose and the background unchanged.")


def test_plain_caption_names_no_print_or_logo_and_only_clothes_get_it():
    from data_tools.fluxrt_prompt import fx6_prompt
    for cat in ("upper", "lower", "dress", "outerwear"):
        p = fx6_prompt(cat, cat, plain=True).lower()
        assert "it is plain: one solid colour all over" in p
        assert not any(w in p for w in ("print", "pattern", "logo")), cat
    for cat in ("shoes", "bags", "accessories"):
        assert fx6_prompt(cat, cat, plain=True) == fx6_prompt(cat, cat)


def test_dress_caption_says_head_covering_and_never_hijab():
    from data_tools.fluxrt_prompt import compact_prompt, fx6_prompt
    d = fx6_prompt("dress", "abaya")
    assert d.endswith("Keep the person's face, hair, head covering, body, pose and the background unchanged.")
    assert "hijab" not in d.lower()
    assert "hijab" in compact_prompt("dress", "abaya").lower()  # what fxreal5 trained with
    # everything else is the same sentence as before
    assert fx6_prompt("upper", "White tee") == compact_prompt("upper", "White tee")


# ---------------------------------------------------------------- frame windows
def _mask(x0, w=40, h=60):
    m = np.zeros((h, w), dtype=bool)
    m[10:40, x0:x0 + 10] = True
    return m


def test_stride_matches_the_live_frame_gap():
    from data_tools.sequences import stride_for
    assert stride_for(30) == 5 and stride_for(24) == 4 and stride_for(60) == 10
    assert stride_for(0) == 5 and stride_for(2) == 1


def test_windows_take_the_moments_with_most_movement_and_never_overlap():
    from data_tools.sequences import pick_windows
    # still for frames 0-39, then the garment moves 2 px a frame from 40 to 79, then still again
    masks = {t: _mask(0 if t < 40 else min(28, (t - 40) // 2 * 2) if t < 80 else 28) for t in range(0, 120)}
    w = pick_windows(masks, k=4, stride=5, n=2)
    assert len(w) == 2
    assert all(40 <= f <= 85 for win in w for f in win["frames"])
    assert w[0]["frames"][-1] < w[1]["frames"][0]
    assert all(b - a == 5 for win in w for a, b in zip(win["frames"], win["frames"][1:]))
    assert w[0]["motion"] > 0


def test_windows_skip_missing_frames_and_respect_min_motion():
    from data_tools.sequences import pick_windows, window_motion
    masks = {t: _mask(t // 3) for t in range(0, 30)}
    del masks[10]
    assert window_motion(masks, [5, 10, 15, 20]) is None
    assert all(10 not in win["frames"] for win in pick_windows(masks, n=3))
    still = {t: _mask(5) for t in range(0, 60)}
    assert pick_windows(still, n=2, min_motion=0.05) == []
    assert pick_windows(masks, k=1) == [] and pick_windows({}, n=2) == []


# ---------------------------------------------------------------- logo stamping
def _shirt(colour, h=120, w=90):
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[30:110, 15:75] = True
    rgb[mask] = colour
    return rgb, mask


def test_stamp_changes_only_the_garment_and_is_clearly_visible():
    from data_tools.logo_stamp import KINDS, stamp
    for colour in ((250, 250, 250), (15, 15, 15), (40, 70, 160)):
        for kind in KINDS:
            rgb, mask = _shirt(colour)
            out, k = stamp(rgb, mask, np.random.default_rng(3), kind=kind)
            assert k == kind
            assert np.array_equal(out[~mask], rgb[~mask]), (colour, kind)
            changed = np.abs(out.astype(int) - rgb.astype(int)).sum(axis=2) > 0
            assert 0.01 < changed[mask].mean() < 0.75, (colour, kind, changed[mask].mean())
            assert np.abs(out.astype(int) - rgb.astype(int))[changed].mean() > 40  # a sharp mark, not a tint


def test_stamp_is_repeatable_and_safe_on_an_empty_area():
    from data_tools.logo_stamp import stamp
    rgb, mask = _shirt((250, 250, 250))
    a, ka = stamp(rgb, mask, np.random.default_rng(11))
    b, kb = stamp(rgb, mask, np.random.default_rng(11))
    assert ka == kb and np.array_equal(a, b)
    out, _ = stamp(rgb, np.zeros_like(mask), np.random.default_rng(1))
    assert np.array_equal(out, rgb)


# ---------------------------------------------------------------- skin-tone groups
def test_ita_groups_from_light_to_dark():
    from data_tools.skin_tone import group, ita
    assert group(ita((75, 8, 14))) == "very_light"
    assert group(ita((62, 12, 17))) in ("light", "intermediate")
    assert group(ita((45, 14, 20))) == "brown"
    assert group(ita((28, 10, 14))) == "dark"
    assert ita((80, 0, 10)) > ita((60, 0, 10)) > ita((40, 0, 10))


def test_balance_lifts_rare_groups_without_memorising_a_few_people():
    from data_tools.skin_tone import balance_weights, missing, shares
    groups = ["very_light"] * 700 + ["light"] * 200 + ["brown"] * 60 + ["dark"] * 40
    w = balance_weights(groups, min_share=0.12)
    s = shares(groups, w)
    assert s["brown"] >= 0.115 and s["dark"] >= 0.115
    assert w["very_light"] == 1.0 and 1.0 < w["dark"] <= 6.0
    assert shares(groups)["dark"] == 0.04
    assert missing(groups, 50) == ["intermediate", "tan", "dark"]
    tiny = ["very_light"] * 1000 + ["dark"] * 5
    assert balance_weights(tiny)["dark"] == 6.0  # capped: 5 people are not shown 24 times each
    assert balance_weights([]) == {}


# ---------------------------------------------------------------- face and skin measures
def test_ciede2000_matches_published_pairs():
    from eval.face_skin import ciede2000
    # Sharma, Wu, Dalal (2005), test data rows 1, 4 and 17
    assert abs(ciede2000((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485)) - 2.0425) < 1e-3
    assert abs(ciede2000((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485)) - 1.0000) < 1e-3
    assert abs(ciede2000((50.0, 2.5, 0.0), (73.0, 25.0, -18.0)) - 27.1492) < 1e-3
    assert ciede2000((60, 10, 15), (60, 10, 15)) == 0.0


def _skin(h=60, w=60, tone=(190, 140, 110), grain=6.0, seed=0):
    """A patch of skin: one tone with fine brightness grain (the same grain on the three channels)."""
    noise = np.random.default_rng(seed).normal(0, grain, (h, w, 1))
    return np.clip(np.array(tone, dtype=np.float64) + noise, 0, 255).astype(np.uint8)


def test_skin_frame_sees_lightening_darkening_and_smoothing():
    from eval.face_skin import skin_frame
    cam = _skin()
    mask = np.ones(cam.shape[:2], dtype=bool)
    same = skin_frame(cam, cam, mask)
    assert same["de"] == 0.0 and same["dl"] == 0.0 and abs(same["texture"] - 1) < 1e-9
    lighter = skin_frame(cam, np.clip(cam.astype(int) + 25, 0, 255).astype(np.uint8), mask)
    darker = skin_frame(cam, np.clip(cam.astype(int) - 25, 0, 255).astype(np.uint8), mask)
    assert lighter["dl"] > 2 and darker["dl"] < -2 and lighter["de"] > 3 and darker["de"] > 3
    smooth = skin_frame(cam, _skin(grain=1.0), mask)
    assert smooth["texture"] < 0.5 and smooth["de"] < 1.5
    assert skin_frame(cam, cam, np.zeros_like(mask)) is None


def test_face_numbers():
    from eval.face_skin import cosine, face_match, face_pixel_error, face_width
    cam = _skin()
    m = np.zeros(cam.shape[:2], dtype=bool)
    m[10:50, 20:45] = True
    assert face_pixel_error(cam, cam, m) == 0.0
    assert abs(face_pixel_error(cam, np.clip(cam.astype(int) + 10, 0, 255).astype(np.uint8), m) - 10) < 0.5
    assert face_width(m) == 25 and face_width(np.zeros_like(m)) == 0
    assert abs(face_match(0.72, 0.80) - 0.9) < 1e-9 and face_match(0.9, 0.8) == 1.0
    assert face_match(None, 0.8) is None and face_match(0.5, 0) is None
    assert abs(cosine([1, 0], [1, 0]) - 1) < 1e-9 and cosine([1, 0], [0, 1]) == 0.0 and cosine([0, 0], [1, 1]) == 0.0


def test_judge_is_the_bar_against_favourite_2():
    from eval.face_skin import by_group, clip_summary, judge
    base = {"skin_de": 6.0, "face_px": 12.0, "face_drift": 3.0, "face_match": 0.70, "texture": 0.70, "closeup": False}
    better = {"skin_de": 2.5, "face_px": 8.0, "face_drift": 2.5, "face_match": 0.85, "texture": 0.95, "closeup": False}
    assert judge(better, base) == []
    assert judge(base, base) == []  # equal to favourite #2 passes (texture no further from 1 than it was)
    worse = {**better, "skin_de": 6.5, "face_match": 0.6}
    assert len(judge(worse, base)) == 2
    close = {**better, "closeup": True, "face_match": 0.86}
    assert any("close-up" in f for f in judge(close, base))
    assert judge({**close, "face_match": 0.93}, base) == []
    assert any("texture" in f for f in judge({**better, "texture": 0.55}, base))
    s = clip_summary([{"skin_de": 2, "skin_dl": -3, "closeup": True}, {"skin_de": 4, "skin_dl": 1, "closeup": True}, {"closeup": False}])
    assert s["skin_de"] == 3 and s["skin_dl"] == -1 and s["skin_dl_abs"] == 2 and s["closeup"] is True and s["face_px"] is None
    g = by_group({"a": {"skin_de": 2.0}, "b": {"skin_de": 2.0}, "c": {"skin_de": 8.0}}, {"a": "light", "b": "light", "c": "dark"})
    assert g["unfair"] == ["dark"] and g["groups"]["light"] == 2.0

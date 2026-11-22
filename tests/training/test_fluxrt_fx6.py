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


# ---------------------------------------------------------------- baseline step: logo set, report, script
def test_face_ncc_is_one_for_the_same_face_whatever_the_brightness():
    from eval.face_skin import face_ncc
    rng = np.random.default_rng(1)
    cam = rng.integers(40, 200, (60, 60, 3)).astype(np.uint8)
    m = np.zeros((60, 60), dtype=bool)
    m[10:50, 10:50] = True
    assert abs(face_ncc(cam, cam, m) - 1) < 1e-9
    brighter = np.clip(cam.astype(int) + 30, 0, 255).astype(np.uint8)
    assert face_ncc(cam, brighter, m) > 0.99
    other = rng.integers(40, 200, (60, 60, 3)).astype(np.uint8)
    assert abs(face_ncc(cam, other, m)) < 0.2
    assert face_ncc(cam, cam, np.zeros_like(m)) is None
    flat = np.full((60, 60, 3), 120, dtype=np.uint8)
    assert face_ncc(flat, flat, m) is None


def test_leak_ratio_is_zero_when_the_logo_is_gone_and_one_when_it_is_kept():
    from data_tools.logo_stamp import stamp
    from eval.face_skin import leak_ratio
    clean, garment = _shirt((250, 250, 250))
    marked, _ = stamp(clean, garment, np.random.default_rng(5), kind="logo")
    out = np.full_like(clean, 200)
    out[garment] = (128, 128, 128)                    # the try-on: a plain grey tee
    assert leak_ratio(clean, marked, out, out, garment) == 0.0
    diff = marked.astype(int) - clean.astype(int)     # the logo copied straight into the try-on
    scale = 0.5                                       # at half its contrast, so nothing clips at black or white
    kept = np.clip(out.astype(int) + (diff * scale).astype(int), 0, 255).astype(np.uint8)
    full = leak_ratio(clean, marked, out, kept, garment)
    assert abs(full - scale) < 0.02
    faint = np.clip(out.astype(int) + (diff * 0.1).astype(int), 0, 255).astype(np.uint8)
    assert 0 < leak_ratio(clean, marked, out, faint, garment) < full
    assert leak_ratio(clean, clean, out, out, garment) is None  # no mark to measure


def test_plain_score_tells_a_plain_garment_photo_from_a_printed_one():
    from eval.logo_set import choose, plain_score
    def photo(fill):
        img = np.full((120, 100, 3), 255, dtype=np.uint8)
        img[20:100, 20:80] = fill(np.zeros((80, 60, 3), dtype=np.uint8))
        return img
    def plain(a):
        a[:] = (30, 60, 150)
        a[:, ::7] = (26, 54, 140)      # folds: a little darker, the same colour
        return a
    def striped(a):
        a[:] = (30, 60, 150)
        a[::6] = (240, 240, 240)
        a[1::6] = (240, 240, 240)
        return a
    p, s = plain_score(photo(plain)), plain_score(photo(striped))
    assert p["plain"] is True and s["plain"] is False and p["rank"] < s["rank"]
    assert plain_score(np.full((50, 50, 3), 255, dtype=np.uint8)) is None  # nothing but backdrop
    items = [{"key": "a", "category": "tops"}, {"key": "b", "category": "dresses"}, {"key": "c", "category": "bottoms"},
             {"key": "d", "category": "tops"}]
    got = choose(items, {"a": s, "b": p, "c": p, "d": None}, 5)
    assert [i["key"] for i in got] == ["b", "a"]  # plain first; bottoms and unreadable photos left out


def test_logo_follows_the_garment_and_stays_on_the_top_for_tops():
    from eval.logo_set import garment_area, stamp_frames
    cat = np.zeros((120, 90), dtype=np.uint8)
    cat[20:110, 20:70] = 4
    top = garment_area(cat, "tops")
    assert top[25, 40] and not top[100, 40] and garment_area(cat, "dresses")[100, 40]
    frames, cats = [], []
    for shift in (0, 8):
        c = np.zeros((120, 90), dtype=np.uint8)
        c[20:110, 15 + shift:65 + shift] = 4
        f = np.full((120, 90, 3), 200, dtype=np.uint8)
        f[c == 4] = (245, 245, 245)
        frames.append(f)
        cats.append(c)
    out, kind = stamp_frames(frames, cats, "tops", seed=101)
    marks = [np.abs(o.astype(int) - f.astype(int)).sum(axis=2) > 0 for o, f in zip(out, frames)]
    assert all(m.any() for m in marks) and kind in ("logo", "text", "stripes", "graphic")
    cx = [np.nonzero(m.any(axis=0))[0].mean() for m in marks]
    assert 6 <= cx[1] - cx[0] <= 10                    # the mark moved with the garment
    assert all(not m[70:, :].any() for m in marks)      # never on the trousers


def test_report_numbers_for_a_clip_with_a_fake_segmenter():
    from PIL import Image
    from eval.face_skin_report import averages, clip_numbers, table
    rng = np.random.default_rng(2)
    cat = np.zeros((128, 72), dtype=np.uint8)
    cat[10:50, 20:52] = 3      # face
    cat[50:120, 10:62] = 4     # clothes
    cat[60:100, 2:10] = 2      # arm
    base = rng.integers(60, 200, (128, 72, 3)).astype(np.uint8)
    cams = [Image.fromarray(base), Image.fromarray(np.roll(base, 1, axis=1))]
    seg = lambda im: cat  # noqa: E731
    same = clip_numbers(cams, cams, seg)
    assert abs(same["face_match"] - 1) < 1e-9 and same["face_px"] == 0 and same["skin_de"] == 0 and same["face_drift"] == 0
    lighter = [Image.fromarray(np.clip(np.asarray(c).astype(int) + 30, 0, 255).astype(np.uint8)) for c in cams]
    lit = clip_numbers(cams, lighter, seg)
    assert lit["skin_dl"] > 2 and lit["face_px"] > 20 and lit["face_match"] > 0.95
    per = {"clip1": {"fav2": same, "new": lit}}
    av = averages(per, "new")
    assert av["clips"] == 1 and av["skin_dl"] == lit["skin_dl"]
    assert any("clip1 | new" in line for line in table(per, ["fav2", "new"]))


def test_live_prompt_style_and_baseline_script():
    src = (ROOT / "training/eval/generate_fluxrt.py").read_text()
    assert '"live": fx6_prompt' in src and 'choices=("full", "compact", "live")' in src
    sh = (ROOT / "training/fluxrt_lora/fx6_baseline.sh").read_text()
    for must in ("--prompt-style live", "528x704", "blend05s065", "eval.logo_set", "--leak-pairs", "FX6_BASELINE_DONE",
                 "GUARD", "AZ_NO_STOP"):
        assert must in sh, must
    assert "run_train.sh" not in sh and "train_dreambooth" not in sh  # it trains nothing
    import subprocess
    assert subprocess.run(["bash", "-n", str(ROOT / "training/fluxrt_lora/fx6_baseline.sh")]).returncode == 0


# ---------------------------------------------------------------- data build: windows, captions, regions
def test_fx6_caption_turns_a_compact_caption_into_the_live_wording():
    from data_tools.fluxrt_prompt import compact_prompt, fx6_caption, fx6_prompt
    for cat in ("upper", "lower", "dress", "outerwear", "shoes", "bags"):
        for desc in (None, "red satin, long sleeves"):
            c = compact_prompt(cat, None, desc)
            assert fx6_caption(c) == fx6_prompt(cat, None, desc), (cat, desc)
            assert fx6_caption(c, plain=True) == fx6_prompt(cat, None, desc, plain=True), (cat, desc)
    assert "hijab" not in fx6_caption(compact_prompt("dress", None, "black abaya"))


def test_a_window_becomes_two_pairs_the_pair_code_can_read():
    from data_tools.sequences import pair_folders
    assert pair_folders([40, 44, 48, 52]) == [("a", (40, 44)), ("b", (48, 52))]


def test_sleeve_band_is_the_bare_arms_beside_the_top():
    from PIL import Image
    sys.path.insert(0, str(ROOT / "training" / "fluxrt_lora"))
    from fluxrt_lora.build_real import closeup_box, sleeve_band
    m = np.zeros((200, 120), dtype=np.uint8)
    m[60:140, 35:85] = 255                      # a top
    skin = np.zeros_like(m)
    skin[70:150, 20:35] = 255                   # left arm, beside and below the top
    skin[70:150, 85:100] = 255                  # right arm
    skin[20:55, 45:75] = 255                    # face and neck, above the top
    band = np.asarray(sleeve_band(Image.fromarray(m, "L"), Image.fromarray(skin, "L"))) > 127
    assert band[100, 25] and band[100, 92]      # arms level with the top
    assert not band[30, 60]                     # never the face or neck
    assert not band[148, 25]                    # nor the hands below the hem
    assert not (band & (m > 127)).any()         # nor the garment itself
    assert sleeve_band(Image.fromarray(m, "L"), Image.fromarray(np.zeros_like(m), "L")) is None
    assert sleeve_band(Image.fromarray(np.zeros_like(m), "L"), Image.fromarray(skin, "L")) is None
    box = closeup_box(Image.fromarray(m, "L"), 120, 200)
    assert box == (24, 0, 96, 120)              # top 60%, 60% wide, centred on the garment, same 9:16 shape
    assert closeup_box(Image.fromarray(np.zeros_like(m), "L"), 120, 200) == (24, 0, 96, 120)
    m2 = np.zeros_like(m)
    m2[60:140, 0:30] = 255                      # garment at the left edge: the box stays inside the picture
    assert closeup_box(Image.fromarray(m2, "L"), 120, 200)[0] == 0


def test_region_mask_gives_face_full_weight_and_skin_and_hair_half():
    from data_tools.region_weight import region_mask, repeats, tone_group
    cat = np.zeros((60, 40), dtype=np.uint8)
    cat[5:12, 15:25] = 1       # hair
    cat[12:22, 15:25] = 3      # face
    cat[30:50, 2:8] = 2        # arm
    garment = np.zeros((60, 40), dtype=np.uint8)
    garment[24:55, 10:30] = 255
    m = region_mask(garment, cat)
    assert m[40, 20] == 255 and m[15, 20] == 255 and m[40, 5] == 128 and m[8, 20] == 128 and m[58, 38] == 0
    rgb = np.zeros((60, 40, 3), dtype=np.uint8)
    rgb[cat == 3] = (235, 200, 180)
    rgb[cat == 2] = (235, 200, 180)
    assert tone_group(rgb, cat) in ("very_light", "light")
    assert tone_group(rgb, np.zeros_like(cat)) is None
    units = [(f"pair:{i}", "very_light") for i in range(90)] + [(f"row:{i}", "dark") for i in range(6)] + [("row:x", None)]
    r = repeats(units, 0.12, 3)
    assert r["pair:0"] == 1 and r["row:0"] == 2 and r["row:x"] == 1   # 6 x 2 of 102 shown = 12%
    assert repeats([("a", "very_light")] * 200 + [("b", "dark")], 0.12, 3)["b"] == 3  # never more than max_repeat


def test_build_real_has_the_fx6_options_and_runs_the_window_section():
    src = (ROOT / "training/fluxrt_lora/build_real.py").read_text()
    for must in ('"--windows"', '"--caption-style"', '"--logo-frac"', '"--closeup-frac"', '"--exposure-frac"',
                 "def windows_section()", "windows_section()\n    flush(final=True)", 'pair=f"w_{clip}"'):
        assert must in src, must
    # the older runs are untouched: without the new options nothing new happens
    assert 'default="compact"' in src and src.count("default=0.0, help=\"fxreal6") == 3


def test_window_rows_are_built_as_pairs_with_live_captions_closeups_marks_and_sleeves(tmp_path):
    """The real builder, with a fake re-dresser: every window pair becomes two rows that share a pair id."""
    import io
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from fluxrt_lora import build_real

    def jb(img):
        b = io.BytesIO(); img.save(b, format="JPEG"); return {"bytes": b.getvalue(), "path": "x.jpg"}
    t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    rows = {"cloth": [], "image": [], "agnostic": []}
    for i in range(12):
        person = Image.new("RGB", (768, 1024), (200, 30 + i, 40))
        agn = person.copy(); agn.paste((128, 128, 128), (200, 300, 560, 700))
        rows["cloth"].append(jb(Image.new("RGB", (300, 400), (i * 5, 90, 200))))
        rows["image"].append(jb(person)); rows["agnostic"].append(jb(agn))
    (tmp_path / "viton").mkdir()
    pq.write_table(pa.table({k: pa.array(v, type=t) for k, v in rows.items()}), tmp_path / "viton" / "a.parquet")
    win = tmp_path / "windows"
    cats = ["upper", "upper", "upper", "dress", "dress", "lower", "lower"]
    for c, cat in enumerate(cats):
        for suffix in ("a", "b"):
            d = win / f"vivid_train_{c:06d}_w1{suffix}"
            d.mkdir(parents=True)
            for tag in ("a", "b"):
                Image.new("RGB", (432, 768), (90 + c, 60, 70)).save(d / f"{tag}_target.png")
                mk = Image.new("L", (432, 768), 0); mk.paste(255, (130, 220, 300, 520)); mk.save(d / f"{tag}_mask.png")
                sk = Image.new("L", (432, 768), 0); sk.paste(255, (90, 240, 130, 500)); sk.paste(255, (300, 240, 340, 500))
                sk.save(d / f"{tag}_skin.png")
            # a plain garment photo for the tops, a two-colour one for the rest
            ref = Image.new("RGB", (200, 260), (255, 255, 255)); ref.paste((20, 40, 120), (40, 40, 160, 220))
            if cat != "upper":
                ref.paste((230, 200, 40), (40, 40, 160, 130))
            ref.save(d / "ref.jpg", quality=95)
            (d / "meta.json").write_text(json.dumps({"category": cat, "split": "train", "garment_description": f"item {c}"}))
    prompts = []

    class FakeRedress:
        def __init__(self, *a, **k): pass
        def __call__(self, person, garment, seed, category="tops", prompt=None):
            prompts.append(prompt)
            return Image.new("RGB", (432, 768), (0, 255, 0))

    class FakeDino:
        def __call__(self, a, b): return 0.5

    old = build_real.Redresser, build_real.DinoSim
    build_real.Redresser, build_real.DinoSim = FakeRedress, FakeDino
    try:
        s = build_real.main(["--out", str(tmp_path / "data"), "--viton-dir", str(tmp_path / "viton"), "--klein", "x",
                             "--test-out", str(tmp_path / "pairs"), "--viton-n", "0", "--holdout", "2", "--test-items", "0",
                             "--windows", str(win), "--caption-style", "live", "--closeup-frac", "1.0", "--logo-frac", "1.0",
                             "--exposure-frac", "1.0", "--nochange-frac", "0", "--anchor-frac", "1.0"])
    finally:
        build_real.Redresser, build_real.DinoSim = old
    assert s["window_pairs_tried"] == 14 and s["window_pairs_kept"] == 14 and s["samples"] == 28
    assert s["window_by_type"] == {"upper": 6, "dress": 4, "lower": 4}
    assert s["closeups"] == 20 and s["logos"] == 20          # tops and dresses only, never trousers
    assert s["exposure_kept"] == 6                           # long-sleeve inputs: tops only
    assert s["plain_captions"] == 12                         # the tops' photo is plain
    assert any(p and "full-length sleeves" in p for p in prompts)
    tabs = [pq.read_table(f) for f in sorted((tmp_path / "data" / "data").glob("*.parquet"))]
    keys = sum((x.column("key").to_pylist() for x in tabs), [])
    pairs = sum((x.column("pair").to_pylist() for x in tabs), [])
    caps = sum((x.column("caption").to_pylist() for x in tabs), [])
    assert all(k.startswith("vivid_win_") for k in keys)
    from collections import Counter
    assert set(Counter(pairs).values()) == {2}               # every pair id is on exactly two rows
    top = next(c for k, c in zip(keys, caps) if "000000" in k)
    assert "It is plain: one solid colour all over" in top and "item 0" in top and "logo" not in top
    dress = next(c for k, c in zip(keys, caps) if "000003" in k)
    assert "head covering" in dress and "hijab" not in dress and "print, pattern, logo" in dress
    row = {c: tabs[0].column(c)[0].as_py() for c in ("person", "target", "mask")}
    sizes = {c: Image.open(io.BytesIO(row[c]["bytes"])).size for c in row}
    assert sizes == {"person": (288, 512), "target": (288, 512), "mask": (288, 512)}
    # fxreal6 previous frame: every window row carries the other half's frame, at the answer's size and framing
    assert s["anchors"] == 28 and all("anchor" in x.column_names for x in tabs)
    anchors = sum((x.column("anchor").to_pylist() for x in tabs), [])
    assert all(an is not None for an in anchors)
    an = Image.open(io.BytesIO(anchors[0]["bytes"]))
    assert an.size == (288, 512)
    person, target = (np.asarray(Image.open(io.BytesIO(row[c]["bytes"])).convert("RGB")).astype(int) for c in ("person", "target"))
    assert np.abs(person - target).mean() > 5                # the input differs from the answer (re-dressed, marked)


def test_pod_scripts_rebuild_the_tools_venv_and_train_with_the_fx5_recipe():
    import subprocess
    for name in ("fx6_baseline.sh", "fx6_data.sh", "fx6_train.sh"):
        sh = (ROOT / "training/fluxrt_lora" / name).read_text()
        assert subprocess.run(["bash", "-n", str(ROOT / "training/fluxrt_lora" / name)]).returncode == 0, name
        # the venv is rebuilt with its own Python 3.11 and versions that work together (numpy < 2 for MediaPipe)
        for must in ("mp_ok()", "uv venv --python 3.11", 'mediapipe==0.10.21 av pillow "numpy<2" "pyarrow==17.0.0"', "NO_MEDIAPIPE_VENV"):
            assert must in sh, (name, must)
    tr = (ROOT / "training/fluxrt_lora/fx6_train.sh").read_text()
    for must in ("AZ_INIT_LORA=$INIT", "AZ_LORA_RANK=64", "AZ_BATCH=4", "AZ_MASK_WEIGHT:-3.0", "AZ_PAIR_COLUMN=pair",
                 "AZ_TIME_SHIFT_MU=2.01", "528x704:3,384x512:1", "FLUX.2-klein-base-4B", "--prompt-style live",
                 "--blend \"0.$(printf", "FACESKIN", "FX6_TRIAL_DONE", "FX6_TRAIN_DONE", "FX6_EVAL_DONE", "TRIAL {passes} passes"):
        assert must in tr, must
    data = (ROOT / "training/fluxrt_lora/fx6_data.sh").read_text()
    for must in ("data_tools.sequences", "--windows", "--caption-style live", "data_tools.region_weight", "FX6_DATA_DONE", "AZ_MAX_MINUTES"):
        assert must in data, must
    base = (ROOT / "training/fluxrt_lora/fx6_baseline.sh").read_text()
    assert 'rm -f "$GUARD"' in base and "touch \"$GUARD\"" not in base  # an early stop never blocks the rerun



# ---------------------------------------------------------------- previous frame (owner, 2026-11-15)
def test_the_earlier_frame_comes_from_the_other_half_of_the_window(tmp_path):
    sys.path.insert(0, str(ROOT / "training" / "fluxrt_lora"))
    from fluxrt_lora.build_real import degrade_like_output, other_window_frame, write_parquet
    import random as _r
    from PIL import Image
    for suffix in ("a", "b"):
        d = tmp_path / f"clip7_w1{suffix}"
        d.mkdir()
        for tag in ("a", "b"):
            Image.new("RGB", (8, 8)).save(d / f"{tag}_target.png")
    assert other_window_frame(tmp_path / "clip7_w1a") == tmp_path / "clip7_w1b" / "b_target.png"   # frame 4 for frames 1-2
    assert other_window_frame(tmp_path / "clip7_w1b") == tmp_path / "clip7_w1a" / "a_target.png"   # frame 1 for frames 3-4
    assert other_window_frame(tmp_path / "clip8_w1a") is None and other_window_frame(tmp_path / "plain") is None
    img = Image.new("RGB", (64, 64), (200, 30, 30))
    outs = [degrade_like_output(img, _r.Random(k)) for k in range(40)]
    assert any(o is img for o in outs) and any(o is not img for o in outs)    # about half are softened
    import io as _io
    import pyarrow.parquet as pq
    b = _io.BytesIO(); img.save(b, format="JPEG")
    row = {"key": "k", "caption": "c", "pair": "", **{c: b.getvalue() for c in ("person", "garment", "target", "mask")}}
    write_parquet([row, {**row, "key": "k2", "anchor": b.getvalue()}], tmp_path / "t.parquet", anchor=True)
    t = pq.read_table(tmp_path / "t.parquet")
    assert t.column("anchor").to_pylist()[0] is None and t.column("anchor").to_pylist()[1]["bytes"]
    write_parquet([row], tmp_path / "u.parquet")                                 # older runs: no new column
    assert "anchor" not in pq.read_table(tmp_path / "u.parquet").column_names


def test_trainer_patch_adds_the_earlier_frame_as_the_third_image(tmp_path):
    import random
    import subprocess
    import types
    trainer = "train_dreambooth_lora_flux2_klein_img2img.py"
    local = Path("/home/user/repos/diffusers")
    if not (local / "examples" / "dreambooth" / trainer).exists():
        import pytest
        pytest.skip("diffusers checkout not present")
    r = subprocess.run(["git", "-C", str(local), "show", f"HEAD:examples/dreambooth/{trainer}"], capture_output=True, text=True)
    repo = tmp_path / "d"
    (repo / "examples" / "dreambooth").mkdir(parents=True)
    (repo / "examples" / "dreambooth" / trainer).write_text(r.stdout)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    order = ("garment", "timeshift", "maskloss", "fx3", "initlora", "fx5", "refsizes", "cpucache", "anchor")
    run = (ROOT / "training/fluxrt_lora/run_train.sh").read_text()
    assert [n for n in order if f"flux2-klein-{n}.patch" in run] == list(order)          # the launcher applies it last
    assert run.index("flux2-klein-anchor.patch") > run.index("flux2-klein-cpucache.patch")
    for name in order:
        rr = subprocess.run(["git", "-C", str(repo), "apply", str(ROOT / "training" / "patches" / f"flux2-klein-{name}.patch")],
                            capture_output=True, text=True)
        assert rr.returncode == 0, f"{name}: {rr.stderr}"
    src = (repo / "examples" / "dreambooth" / trainer).read_text()
    compile(src, trainer, "exec")
    for must in ('"--cond_image_column_3"', '"--cond_3_prob"', "cond_3_latents_cache.append(None)",
                 "cond_list.append(cond_3_model_input[0:1])", "Flux2KleinPipeline._pack_latents(cond_3_model_input)",
                 "ANCHOR_BATCHES", "_AZ_COND3_PROB = float(getattr(args, \"cond_3_prob\", 1.0))"):
        assert must in src, must
    # [person, garment, anchor]: the anchor goes last, after the garment, as in FluxRT's reference list
    assert src.index("cond_list.append(cond_3_model_input[0:1])") > src.index("cond_list.append(cond_2_model_input[0:1])")
    torch = __import__("pytest").importorskip("torch")
    from PIL import Image, ImageOps
    start = src.index("# Azyah fxreal6: the earlier frame")
    end = src.index("def collate_fn(examples):")
    tf = types.SimpleNamespace(
        to_tensor=lambda im: torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1),
        normalize=lambda t, mean, std: (t - mean[0]) / std[0])
    ns = {"random": random, "torch": torch, "Image": Image, "TF": tf, "exif_transpose": ImageOps.exif_transpose}
    exec(compile(src[start:end] + "\n", "anchor_block", "exec"), ns)
    t = ns["_azyah_anchor_tensor"](Image.new("RGB", (144, 256), (255, 0, 0)), 288, 512)
    assert tuple(t.shape) == (3, 512, 288) and abs(float(t[0].mean()) - 1.0) < 1e-6
    full = [{"cond_images_3": t} for _ in range(4)]
    assert ns["_azyah_use_anchor"](full) is True                                  # prob 1.0: every eligible batch
    assert ns["_azyah_use_anchor"](full[:3] + [{}]) is False                      # one row without: none of them
    ns["_AZ_COND3_PROB"] = 0.5
    ns["_AZ_COND3_RNG"].seed(1)
    share = sum(ns["_azyah_use_anchor"](full) for _ in range(2000)) / 2000
    assert abs(share - 0.5) < 0.05


def test_scripts_carry_the_earlier_frame_and_the_live_setting_stays_a_switch():
    data = (ROOT / "training/fluxrt_lora/fx6_data.sh").read_text()
    train = (ROOT / "training/fluxrt_lora/fx6_train.sh").read_text()
    assert '--anchor-frac "${AZ_ANCHOR_FRAC:-1.0}"' in data and "'anchors'" in data
    assert "removing an unfinished row build" in data                            # a stopped build never mixes in
    assert "AZ_ANCHOR_COLUMN=${AZ_ANCHOR_COLUMN:-anchor}" in train and "^ANCHOR_BATCHES" in train
    assert '--anchor-after 8 --anchor-every 12' in train and "fxreal6 + earlier frame" in train
    gen = (ROOT / "training/eval/generate_fluxrt.py").read_text()
    assert '"--anchor-after"' in gen and "images = [model_frames[i], garment] + ([anchor] if anchor is not None else [])" in gen


# ---------------------------------------------------------------- overnight chain (owner, 2026-11-15)
def _overnight(tmp_path, spent, sec="3.10", rows=8000, trial_extra="", train_extra=""):
    """Runs fx6_overnight.sh against a fake fx6_train.sh and a finished data log; returns (exit code, output, calls)."""
    import os
    import subprocess
    ws = tmp_path / "ws"
    root = ws / "azyah-vton"
    (root / "reports").mkdir(parents=True)
    (ws / "fx6_data.log").write_text('DATASET {"samples": 8000, "anchors": 6200}\nREGIONS {"rows_written": 8000}\nFX6_DATA_DONE after 150 min\n')
    calls = ws / "calls.txt"
    (ws / "fx6_train.sh").write_text(f"""#!/usr/bin/env bash
echo "$AZ_STAGE steps=${{AZ_MAX_STEPS:-}} minutes=${{AZ_MAX_MINUTES:-}}" >> {calls}
log={root}/reports/fluxlora_fxreal6_x.log
case "$AZ_STAGE" in
trial) echo "step 1 loss=0.31 detail=0.2 {trial_extra}" > $log
       echo "TRIAL {sec} s per step (batch 4), {rows} rows"; echo "FX6_TRIAL_DONE after 30 min" ;;
train) echo "loss=0.28 {train_extra}" >> $log; mkdir -p {root}/outputs/fluxrt_lora/fxreal6/checkpoint-500
       echo "FX6_TRAIN_DONE" ;;
eval)  echo "FACESKIN fx6test fx6_r528: clips 24"; echo "FX6_EVAL_DONE after 60 min" ;;
esac
""")
    env = {**os.environ, "AZ_REF": "x", "AZ_SPENT": str(spent), "AZ_WS": str(ws), "AZ_SKIP_FETCH": "1", "AZ_NO_STOP": "1"}
    r = subprocess.run(["bash", str(ROOT / "training/fluxrt_lora/fx6_overnight.sh")], capture_output=True, text=True,
                       env=env, timeout=60)
    return r.returncode, r.stdout + r.stderr, calls.read_text().splitlines() if calls.exists() else []


def test_overnight_runs_trial_then_full_training_then_eval_within_the_cap(tmp_path):
    code, out, calls = _overnight(tmp_path, spent=20)
    assert code == 0, out
    assert calls[0].startswith("trial") and calls[1].startswith("train") and calls[2].startswith("eval")
    steps = int(calls[1].split("steps=")[1].split()[0])
    minutes = int(calls[1].split("minutes=")[1])
    assert steps == 6000                                       # 3 passes of 8000 rows at batch 4, rounded down
    assert minutes <= ((45 - 20 - 2) / 2.70 - 1.5) * 60        # the time limit can never spend past the cap
    assert steps * 3.10 / 60 + 20 >= minutes                   # and it is the steps' own time when that fits
    assert "OVERNIGHT" in out and "DONE: LoRA" in out and "6200 rows with the earlier frame" in out


def test_overnight_trains_fewer_steps_when_money_is_short_and_stops_when_too_short(tmp_path):
    code, out, calls = _overnight(tmp_path, spent=33)
    assert code == 0, out
    steps = int(calls[1].split("steps=")[1].split()[0])
    assert 800 <= steps < 6000                                 # what the money left pays for
    assert steps * 3.10 / 3600 * 2.70 <= 45 - 33 - 2 - 1.5 * 2.70 + 0.5
    code, out, calls = _overnight(tmp_path / "b", spent=38.5)
    assert code != 0 and "not worth training" in out or "too much of the budget" in out
    assert not any(c.startswith("train") for c in calls)


def test_overnight_stops_on_a_bad_trial(tmp_path):
    code, out, calls = _overnight(tmp_path, spent=20, trial_extra="loss=nan")
    assert code != 0 and "NaN" in out and calls == ["trial steps= minutes="]
    code, out, calls = _overnight(tmp_path / "b", spent=20, trial_extra="torch.OutOfMemoryError: CUDA out of memory")
    assert code != 0 and "the trial hit an error" in out and len(calls) == 1


def test_overnight_resume_measures_the_training_bar_not_the_caching_bar_and_counts_only_steps_left(tmp_path):
    """2026-11-15: the trial read the "Caching latents" speed (0.20 s) as the step time, so the time limit came out
    about 6x too short. The resume run must read the "Steps:" bar only, and pay only for the steps still to do."""
    import os
    import subprocess
    ws = tmp_path / "ws"
    root = ws / "azyah-vton"
    out = root / "outputs/fluxrt_lora/fxreal6"
    (out / "checkpoint-1000").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    data = root / "data/fluxrt_lora/fxreal6"
    data.mkdir(parents=True)
    (data / "region_weight.json").write_text('{"rows_written": 7815}')
    bars = "".join(f"\rCaching latents:  {k}%|##| {k}/1954 [00:10<00:00,  5.00it/s]" for k in range(1, 60))
    bars += "".join(f"\rSteps:  {k}%|#| {k}/5800 [10:00<2:00:00,  3.50s/it, loss=0.3]" for k in range(1, 60))
    (root / "reports/fluxlora_fxreal6_x.log").write_text(bars + "\n")
    calls = ws / "calls.txt"
    (ws / "fx6_train.sh").write_text(f"""#!/usr/bin/env bash
echo "$AZ_STAGE steps=${{AZ_MAX_STEPS:-}} minutes=${{AZ_MAX_MINUTES:-}}" >> {calls}
case "$AZ_STAGE" in
train) mkdir -p {out}/checkpoint-2000; echo "FX6_TRAIN_DONE" ;;
eval)  echo "FX6_EVAL_DONE after 60 min" ;;
esac
""")
    env = {**os.environ, "AZ_REF": "x", "AZ_SPENT": "24.75", "AZ_WS": str(ws), "AZ_SKIP_FETCH": "1", "AZ_NO_STOP": "1",
           "AZ_RESUME": "1"}
    r = subprocess.run(["bash", str(ROOT / "training/fluxrt_lora/fx6_overnight.sh")], capture_output=True, text=True,
                       env=env, timeout=60)
    text = r.stdout + r.stderr
    lines = calls.read_text().splitlines()
    assert r.returncode == 0, text
    assert lines[0].startswith("train") and lines[1].startswith("eval")   # no data build, no trial
    assert "3.50 s per step" in text and "newest save step 1000" in text
    steps = int(lines[0].split("steps=")[1].split()[0])
    minutes = int(lines[0].split("minutes=")[1])
    assert 1000 < steps <= 5800
    assert (steps - 1000) * 3.5 / 60 + 20 >= minutes                 # the time limit covers the steps left
    assert minutes <= ((45 - 24.75 - 2) / 2.70 - 1.5) * 60           # and can never spend past the cap


def test_trial_speed_reads_only_the_training_bar():
    text = (ROOT / "training/fluxrt_lora/fx6_train.sh").read_text()
    assert 'startswith("Steps:")' in text and "Caching latents" in text


def test_overnight_eval_only_skips_training_and_stops_inside_the_cap(tmp_path):
    """2026-11-16: evaluate the finished weights again without training again."""
    import os
    import subprocess
    ws = tmp_path / "ws"
    root = ws / "azyah-vton"
    out = root / "outputs/fluxrt_lora/fxreal6"
    out.mkdir(parents=True)
    (out / "pytorch_lora_weights.safetensors").write_bytes(b"w")
    (root / "reports").mkdir(parents=True)
    calls = ws / "calls.txt"
    (ws / "fx6_train.sh").write_text(f"""#!/usr/bin/env bash
echo "$AZ_STAGE label=${{AZ_EVAL_LABEL:-}}" >> {calls}
echo "evaluating weights abcd1234 as fx6s5800_r528 and fx6anchors5800_r528"
echo "render fx6test/fx6s5800_r528: 24 new clips, 24 in total"; echo "FX6_EVAL_DONE after 90 min"
""")
    env = {**os.environ, "AZ_REF": "x", "AZ_SPENT": "34", "AZ_WS": str(ws), "AZ_SKIP_FETCH": "1", "AZ_NO_STOP": "1",
           "AZ_EVAL_ONLY": "1", "AZ_EVAL_LABEL": "s5800"}
    r = subprocess.run(["bash", str(ROOT / "training/fluxrt_lora/fx6_overnight.sh")], capture_output=True, text=True,
                       env=env, timeout=60)
    text = r.stdout + r.stderr
    assert r.returncode == 0, text
    assert calls.read_text().splitlines() == ["eval label=s5800"]          # nothing but the evaluation, label passed on
    assert "evaluation only" in text and "24 new clips, 24 in total" in text and "evaluating weights" in text
    limit = int(text.split("evaluation, time limit ")[1].split()[0])
    assert 20 <= limit <= 150 and limit / 60 * 2.70 <= 45 - 34 - 2 + 0.1   # never past the cap
    # no weights at all: stop before spending anything
    (out / "pytorch_lora_weights.safetensors").unlink()
    r = subprocess.run(["bash", str(ROOT / "training/fluxrt_lora/fx6_overnight.sh")], capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.returncode != 0 and "no trained weights" in r.stdout + r.stderr


def test_eval_names_its_renders_after_the_weights_and_refuses_a_reused_label(tmp_path):
    """The renders are cached per model name, so the name must follow the weights (step-1000 videos were scored
    as step-5800). The naming block of fx6_train.sh is run here on its own."""
    import subprocess
    text = (ROOT / "training/fluxrt_lora/fx6_train.sh").read_text()
    start = text.index('  SUM=$(sha256sum "$FX"')
    end = text.index('  echo "evaluating weights')
    block = text[start:end] + 'echo "NAMES $NEW $ANC $MIX"\necho "MIXES[$MIXES] BLENDS[$BLENDS] ANC[$USE_ANC]"\n'
    fx = tmp_path / "w.safetensors"
    res = tmp_path / "results"

    def run(content, label=None, **extra):
        fx.write_bytes(content)
        env = {"FX": str(fx), "R": str(res), "TAG": "r528", "PATH": "/usr/bin:/bin", **extra}
        if label:
            env["AZ_EVAL_LABEL"] = label
        return subprocess.run(["bash", "-c", block], capture_output=True, text=True, env=env)

    a = run(b"weights-step-1000")
    b = run(b"weights-step-5800")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout != b.stdout and "NAMES fx6" in a.stdout                # different weights: different names
    c = run(b"weights-step-5800", "s5800")
    assert c.returncode == 0 and "NAMES fx6s5800_r528 fx6anchors5800_r528 fx6b75s5800_r528" in c.stdout
    assert run(b"weights-step-5800", "s5800").returncode == 0               # same weights, same label: carries on
    d = run(b"other-weights", "s5800")
    assert d.returncode != 0 and "STALE_EVAL_LABEL" in d.stdout             # other weights under a used label: refused
    assert "NO_RENDERS" in text and "in total" in text
    # blends back towards favourite #2: none by default, the old switch means 75%, a list gives one model per strength
    assert "MIXES[] BLENDS[] ANC[1]" in c.stdout
    e = run(b"weights-step-5800", "s5800", AZ_EVAL_BLEND="1")
    assert "MIXES[ fx6b75s5800_r528] BLENDS[75] ANC[1]" in e.stdout
    f = run(b"weights-step-5800", "s5800", AZ_EVAL_BLENDS="75,50 05", AZ_EVAL_ANCHOR="0")
    assert "MIXES[ fx6b75s5800_r528 fx6b50s5800_r528 fx6b5s5800_r528] BLENDS[75 50 5] ANC[0]" in f.stdout
    g = run(b"weights-step-5800", "s5800", AZ_EVAL_BLENDS="100")
    assert g.returncode != 0 and "BAD_AZ_EVAL_BLENDS" in g.stdout
    assert '--blend "0.$(printf \'%02d\' "$b")"' in text and 'PANELS+=("fx6b${b}${LABEL}_$TAG=fxreal6 at ${b}%")' in text
    assert '[ "$USE_ANC" = "1" ] && render "$s" "$ANC"' in text
    # owner, 2026-11-16: no head coverings for this product: the hijab set is out of the default scores and the GATE
    assert "${AZ_EVAL_SETS:-fx6test cases5 dress10}" in text and "cases5 hijab" not in text

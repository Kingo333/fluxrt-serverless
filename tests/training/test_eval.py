"""Evaluation metrics behave sensibly on synthetic frames."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

from eval import ab_sheet, metrics, run_eval  # noqa: E402

W, H = 120, 160


def striped(color_a, color_b, period=6):
    a = np.zeros((H, W, 3), dtype=np.uint8)
    for x in range(W):
        a[:, x] = color_a if (x // period) % 2 == 0 else color_b
    return Image.fromarray(a)


def person(garment):
    """Grey background/face on top half, garment on bottom half."""
    base = Image.new("RGB", (W, H), (128, 128, 128))
    base.paste(garment.crop((0, H // 2, W, H)), (0, H // 2))
    return base


MASK = Image.new("L", (W, H), 0)
MASK.paste(255, (0, H // 2, W, H))
REF = striped((200, 20, 20), (20, 20, 200))  # red/blue stripes


def test_good_tryon_beats_bad_on_garment_metrics():
    inp = [person(Image.new("RGB", (W, H), (10, 160, 10)))] * 4
    good = [person(REF)] * 4
    blurred = [person(Image.new("RGB", (W, H), (110, 20, 110)))] * 4  # right-ish colours, stripes lost
    wrong = [person(Image.new("RGB", (W, H), (240, 240, 0)))] * 4
    g, b, w = (metrics.score_clip(inp, o, REF, MASK) for o in (good, blurred, wrong))
    assert g["garment_colour"] > w["garment_colour"]
    assert abs(g["garment_detail"] - 1.0) < 0.3 and b["garment_detail"] < 0.2
    assert g["preservation"] > 0.99 and g["flicker"] == 0.0


def test_preservation_drops_when_background_changes_and_flicker_detects_jitter():
    inp = [person(REF)] * 4
    changed_bg = [Image.new("RGB", (W, H), (0, 0, 0))] * 4
    assert metrics.preservation(inp, changed_bg, MASK) < 0.6
    jitter = [person(REF), person(Image.new("RGB", (W, H), (0, 0, 0)))] * 2
    assert metrics.flicker(inp, jitter, MASK) > 0.1


def test_run_eval_and_ab_sheet(tmp_path):
    for clip in ("c1", "c2"):
        d = tmp_path / "results" / clip
        d.mkdir(parents=True)
        person(Image.new("RGB", (W, H), (10, 160, 10))).save(d / "input.png")
        REF.save(d / "ref.png")
        MASK.save(d / "mask.png")
        person(Image.new("RGB", (W, H), (240, 240, 0))).save(d / "out_joyai-baseline.png")
        person(REF).save(d / "out_azyah-exp1.png")
        (d / "perf_azyah-exp1.json").write_text(json.dumps({"fps": 22.5}))
    r = run_eval.evaluate(str(tmp_path / "results"))
    assert r["summary"]["azyah-exp1"]["garment_colour"] > r["summary"]["joyai-baseline"]["garment_colour"]
    assert r["summary"]["azyah-exp1"]["fps"] == 22.5
    assert "| azyah-exp1 |" in run_eval.to_markdown(r)
    n = ab_sheet.make(str(tmp_path / "results"), "joyai-baseline", "azyah-exp1", str(tmp_path / "ab"))
    assert n == 2
    key = json.loads((tmp_path / "ab" / "key.json").read_text())
    rows = list(csv.DictReader(open(tmp_path / "ab" / "review.csv")))
    for row in rows:  # reviewer always picks the azyah-exp1 side
        row["choice"] = "left" if key[row["clip"]]["left"] == "azyah-exp1" else "right"
    with open(tmp_path / "ab" / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "choice", "notes"])
        w.writeheader()
        w.writerows(rows)
    s = ab_sheet.score(str(tmp_path / "ab" / "review.csv"), str(tmp_path / "ab" / "key.json"))
    assert s["win_rate"] == {"azyah-exp1": 1.0}


def test_generate_vivid_picks_held_out_pairs_with_sources(tmp_path):
    import json as _json
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "training"))
    from eval.generate_vivid import test_items
    for i, (split, src, ok) in enumerate([("test", True, True), ("train", True, True), ("test", False, True),
                                          ("test", True, False), ("test", True, True)]):
        d = tmp_path / f"vivid_{split}_{i:06d}"
        d.mkdir()
        (d / "meta.json").write_text(_json.dumps({"split": split, "pseudo_source": {"ok": ok}}))
        if src:
            (d / "source.mp4").write_bytes(b"x")
    picked = [p.name for p in test_items(tmp_path, 10)]
    assert picked == ["vivid_test_000000", "vivid_test_000004"]
    assert [p.name for p in test_items(tmp_path, 1)] == ["vivid_test_000000"]


def test_fluxrt_eval_speed_summary():
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "training"))
    from eval.generate_fluxrt import summarize
    s = summarize([0.05, 0.05, 0.10, 0.10], 12.3)
    assert s["fps"] == 13.33 and s["latency_ms_p50"] == 75.0 and s["latency_ms_p95"] == 100.0
    assert s["gpu_mem_gb"] == 12.3 and s["frames"] == 4
    assert summarize([], None)["fps"] is None


def test_target_metrics_and_no_mask():
    """P9e had no mask.png: preservation must not show a fake 1.0; vs-target scores rank a copy of
    the input below a correct try-on."""
    inp = [person(Image.new("RGB", (W, H), (10, 160, 10)))] * 3
    tgt = [person(REF)] * 3
    good = metrics.score_clip(inp, tgt, REF, None, tgt)
    lazy = metrics.score_clip(inp, inp, REF, None, tgt)
    assert "preservation" not in good
    assert lazy["change"] == 0.0 and good["change"] > 0.05
    assert good["psnr_target"] > lazy["psnr_target"] and good["colour_target"] > lazy["colour_target"]


def _mp4(path, frames):
    import av
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=8)
        st.width, st.height, st.pix_fmt = frames[0].width, frames[0].height, "yuv420p"
        for im in frames:
            for pkt in st.encode(av.VideoFrame.from_image(im)):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def test_grid_and_eval_on_p9e_layout(tmp_path):
    import pytest
    av = pytest.importorskip("av")
    from eval import grid
    for clip in ("vivid_test_000000", "vivid_test_000002"):
        d = tmp_path / "p9e" / clip
        d.mkdir(parents=True)
        _mp4(d / "input.mp4", [person(Image.new("RGB", (W, H), (10, 160, 10)))] * 6)
        _mp4(d / "target.mp4", [person(REF)] * 8)
        _mp4(d / "out_base.mp4", [person(Image.new("RGB", (W, H), (10, 160, 10)))] * 6)
        _mp4(d / "out_exp2.mp4", [person(REF)] * 6)
        REF.save(d / "ref.jpg")
    grid.main([str(tmp_path / "p9e"), str(tmp_path / "grid"), "--height", "80"])
    with av.open(str(tmp_path / "grid" / "all.mp4")) as c:
        n = sum(1 for _ in c.decode(c.streams.video[0]))
        w = c.streams.video[0].width
    assert n == 12 and w > 4 * 50
    assert (tmp_path / "grid" / "vivid_test_000002.mp4").exists()
    r = run_eval.evaluate(str(tmp_path / "p9e"))
    s = r["summary"]
    assert s["exp2"]["psnr_target"] > s["base"]["psnr_target"] and "preservation" not in s["exp2"]
    assert "No mask.png" in run_eval.to_markdown(r)


def test_person_keep_metrics_and_flags():
    """Owner requirement: flag any model that changes the background/person more than the reference."""
    inp = [person(Image.new("RGB", (W, H), (10, 160, 10)))] * 3
    good = metrics.score_clip(inp, [person(REF)] * 3, REF, MASK)
    bg_changed = [Image.new("RGB", (W, H), (0, 0, 0))] * 3
    bad = metrics.score_clip(inp, bg_changed, REF, MASK)
    assert good["border_keep"] > 0.99 and bad["border_keep"] < 0.7
    r = {"summary": {"base": {"clips": 1, **good}, "lora": {"clips": 1, **bad}},
         "per_clip": {"c1": {"base": good, "lora": bad}}}
    f = run_eval.flags(r, "base")
    assert any("FLAG lora: border_keep" in x for x in f) and any("clip c1" in x for x in f)
    r2 = {"summary": {"base": {"clips": 1, **good}, "lora": {"clips": 1, **good}}, "per_clip": {}}
    assert run_eval.flags(r2, "base")[0].startswith("no flags")


def test_border_mask_excludes_garment():
    m = np.zeros((10, 20), dtype=bool)
    m[:, :2] = True
    b = metrics.border_mask((10, 20), m, frac=0.2)
    assert not b[:, :2].any() and b[:, 2:4].all() and b[:, -4:].all() and not b[:, 5:15].any()


def test_shoe_set_and_pairs_mode(tmp_path):
    from eval import generate_fluxrt as gf, make_shoe_set
    persons = tmp_path / "persons"
    persons.mkdir()
    person(REF).save(persons / "me.jpg")
    REF.save(tmp_path / "shoe.png")
    s = tmp_path / "set.json"
    s.write_text(json.dumps({"items": [{"key": "shoe_01", "name": "Nike Shox TL sneakers in blue", "description": "Mesh upper",
                                        "image_url": (tmp_path / "shoe.png").as_uri()}]}))
    st = make_shoe_set.build(str(tmp_path / "shoes"), str(s), persons=str(persons))
    assert st["written"] == 1 and not st["failed"]
    items = gf.pair_items(tmp_path / "shoes", 10)
    assert [d.name for d in items] == ["shoe_01"]
    assert json.loads((items[0] / "meta.json").read_text())["category"] == "shoes"
    z = gf.item_zone_mask("shoes", 40, 80)
    a = np.asarray(z)
    assert a[:60].max() == 0 and a[60:].min() == 255 and gf.item_zone_mask("tops", 40, 80) is None
    assert make_shoe_set.build(str(tmp_path / "shoes"), str(s), persons=str(persons))["kept"] == 1


def test_still_grid_review_sheet(tmp_path):
    from eval import grid
    for k in range(4):
        d = tmp_path / "shoes" / f"shoe_0{k}"
        d.mkdir(parents=True)
        person(REF).save(d / "input.png")
        REF.save(d / "ref.jpg")
        person(REF).save(d / "out_base.png")
        person(REF).save(d / "out_lora.png")
    grid.main([str(tmp_path / "shoes"), str(tmp_path / "g"), "--models", "base", "lora", "--height", "60", "--max-items", "3"])
    sheet = Image.open(tmp_path / "g" / "all.png")
    assert sheet.height == 3 * (60 + grid.LABEL_H) and (tmp_path / "g" / "shoe_02.png").exists()
    assert not (tmp_path / "g" / "shoe_03.png").exists()


def test_worst_frame_and_drift_are_reported_and_flagged():
    """Owner 2026-11-05: check every frame, not only the average, and drift over the chunks."""
    inp = [person(Image.new("RGB", (W, H), (10, 160, 10)))] * 12
    steady = [person(REF)] * 12
    late_bg_break = [person(REF)] * 8 + [Image.new("RGB", (W, H), (0, 0, 0))] * 4
    a = metrics.score_clip(inp, steady, REF, MASK)
    b = metrics.score_clip(inp, late_bg_break, REF, MASK)
    assert a["border_keep_min"] > 0.99 and a["border_keep_drift"] == 0.0
    assert b["border_keep_min"] < 0.7 and b["border_keep_worst_frame"] >= 8 and b["border_keep_drift"] < -0.2
    r = {"summary": {"ref": {"clips": 1, **a}, "lora": {"clips": 1, **b}}, "per_clip": {"c": {"ref": a, "lora": b}}}
    f = run_eval.flags(r, "ref")
    assert any("worst-frame border_keep" in x and "at frame" in x for x in f)
    assert any("border_keep drifts" in x for x in f)
    md = run_eval.to_markdown({"summary": r["summary"], "per_clip": {}, "masked": True})
    assert "Every frame" in md and "border_keep_min" in md


def test_frame_strip_marks_the_worst_frame(tmp_path):
    import pytest as _p
    _p.importorskip("av")
    from eval import frame_strip
    d = tmp_path / "r" / "clip1"
    d.mkdir(parents=True)
    green = person(Image.new("RGB", (W, H), (10, 160, 10)))
    _mp4(d / "input.mp4", [green] * 8)
    _mp4(d / "out_live0811.mp4", [green] * 8)
    _mp4(d / "out_lora250.mp4", [green] * 5 + [Image.new("RGB", (W, H), (0, 0, 0))] + [green] * 2)
    REF.save(d / "ref.jpg")
    n = frame_strip.main([str(tmp_path / "r"), str(tmp_path / "s"), "--models", "live0811", "lora250",
                          "--every", "2", "--height", "40"])
    sheet = Image.open(tmp_path / "s" / "clip1_frames.png")
    assert n == 1 and sheet.height == frame_strip.HEAD_H + 3 * 44
    ch = frame_strip.person_change([green] * 8, [green] * 5 + [Image.new("RGB", (W, H), (0, 0, 0))] + [green] * 2)
    assert int(np.argmax(ch)) == 5


def test_composite_keeps_real_face_and_background():
    from eval import person_parse as pp
    real = Image.new("RGB", (20, 20), (10, 10, 200))      # camera: blue
    gen = Image.new("RGB", (20, 20), (200, 10, 10))       # generated: red
    cat_in = np.zeros((20, 20), dtype=np.uint8)           # background everywhere
    cat_in[0:5, :] = pp.MP["face"]
    cat_in[10:20, 5:15] = pp.MP["clothes"]
    cat_out = cat_in.copy()
    cat_out[5:10, 5:15] = pp.MP["clothes"]                # the new garment is longer upwards
    keep = pp.keep_mask(cat_in, cat_out, "upper")
    assert keep[0, 0] and keep[2, 10] and not keep[15, 10] and not keep[7, 10]
    out = np.asarray(pp.composite(gen, real, keep, feather=0))
    assert tuple(out[2, 10]) == (10, 10, 200) and tuple(out[15, 10]) == (200, 10, 10)
    shoes = pp.keep_mask(cat_in, cat_out, "shoes")         # shoes: face/hair only, no shoe masks
    assert shoes[2, 10] and not shoes[19, 0]


def test_tryon_agnostic_input_and_refs():
    from eval import person_parse as pp
    img = Image.new("RGB", (20, 20), (0, 200, 0))
    cat = np.zeros((20, 20), dtype=np.uint8)
    cat[5:10, 5:15] = pp.SF["upper"]
    cat[12:18, 5:15] = pp.SF["pants"]
    ag = np.asarray(pp.agnostic(img, cat))
    assert tuple(ag[7, 10]) == (128, 128, 128) and tuple(ag[0, 0]) == (0, 200, 0)
    garment = Image.new("RGB", (8, 8), (255, 0, 0))
    top, bottom = pp.tryon_refs(img, cat, garment, "upper")
    assert top is garment and bottom.size == (10, 6)       # the person's own pants
    top2, bottom2 = pp.tryon_refs(img, cat, garment, "dress")
    assert top2 is garment and bottom2 is garment

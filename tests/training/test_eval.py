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

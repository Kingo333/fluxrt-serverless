"""Step A: LoRA blends, the fixed face/skin check, the 10-clip list and the labelled video."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))


def test_blend_weights_split_the_strength_between_the_two_loras():
    from eval.generate_fluxrt import adapter_weights
    assert adapter_weights(0.5) == [0.5, 0.5]
    assert adapter_weights(0.3, 1.0) == [0.3, 0.7]
    assert adapter_weights(0.5, 0.8) == [0.4, 0.4]
    assert adapter_weights(1.0, 0.65) == [0.65, 0.0]
    with pytest.raises(ValueError):
        adapter_weights(1.2)


def test_fixed_check_compares_skin_only_where_it_stays_skin():
    np = pytest.importorskip("numpy")
    from eval import preserve_check as pc
    cat_in = np.zeros((20, 20), np.uint8)
    cat_in[0:5, :] = pc.FACE               # 100 px of face
    cat_in[5:10, :] = pc.BODY_SKIN         # 100 px of arm in the camera
    out_a = cat_in.copy()
    out_a[5:10, 10:] = 4                   # a new sleeve (clothes) now covers half the arm
    face, skin, bg = pc.common_masks(cat_in, [out_a])
    assert face.sum() == 100 and skin.sum() == 150   # the covered arm pixels are left out of the skin area
    assert bg.sum() == 200                           # rows 10-19 are background in both
    # Old behaviour (input only) counted all 200 skin pixels, sleeve included.
    inp = np.full((20, 20, 3), 100, np.uint8)
    out = inp.copy()
    out[5:10, 10:] = 20                              # the sleeve is dark fabric
    r_old = pc.region_stats(inp, out, cat_in, bg)
    r_new = pc.region_stats(inp, out, cat_in, bg, face, skin)
    assert r_old["skin"] > 10 and r_new["skin"] == 0   # fabric over an arm no longer reads as a skin change


def test_new_clips_come_from_unused_held_out_clips_with_another_clips_garment(tmp_path):
    from eval import step_a
    vivid = tmp_path / "vivid"
    cats = ["upper"] * 8 + ["dress"] * 4 + ["lower"] * 4   # 6 usable tops after the two exclusions: 3 pairs
    for i, cat in enumerate(cats):
        d = vivid / f"vivid_test_{i:06d}"
        d.mkdir(parents=True)
        (d / "target.mp4").write_bytes(b"v")
        (d / "ref.jpg").write_bytes(f"garment-{i}".encode())
        (d / "meta.json").write_text(json.dumps({"split": "test", "category": cat}))
    (vivid / "vivid_train_000001").mkdir()
    (vivid / "vivid_train_000001" / "meta.json").write_text(json.dumps({"split": "train", "category": "upper"}))
    clips = step_a.held_out_clips(vivid)
    assert len(clips) == 16
    used_photo = step_a._sha(vivid / "vivid_test_000003" / "ref.jpg")
    picked = step_a.pick_new_clips(clips, exclude={"vivid_test_000000"}, excluded_ref_shas={used_photo})
    assert len(picked) == 5
    names = {p["person"][0] for p in picked} | {p["donor"][0] for p in picked}
    assert "vivid_test_000000" not in names and "vivid_test_000003" not in names
    for p in picked:
        assert p["person"][0] != p["donor"][0]
        assert p["person"][1]["category"] == p["donor"][1]["category"]   # same type: tops on a top, etc.
    assert [p["person"][1]["category"] for p in picked] == list(step_a.NEW_CATEGORIES)
    assert len(names) == 10                                                 # no clip used twice


def _clip(path, color, n=5):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=24)
        s.width, s.height, s.pix_fmt = 288, 512, "yuv420p"
        for _ in range(n):
            for pkt in s.encode(av.VideoFrame.from_image(Image.new("RGB", (288, 512), color))):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def test_reuse_and_labelled_video(tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    from eval import step_a
    root = tmp_path
    owner = [{"key": "1_standing_x", "case": "standing", "video": "v", "ref": "r", "category": "bottoms",
              "name": None, "description": None}]
    (root / "eval/fx2/cases5").mkdir(parents=True)
    (root / "eval/fx2/cases5/clip_list.json").write_text(json.dumps(owner))
    fx4 = root / "eval/fx4/results/cases5/1_standing_x"
    fx3 = root / "eval/fx3/results/cases5/1_standing_x"
    fx4.mkdir(parents=True), fx3.mkdir(parents=True)
    _clip(fx4 / "input.mp4", (40, 40, 40)); _clip(fx4 / "out_base.mp4", (200, 0, 0)); _clip(fx4 / "out_f4_2000.mp4", (0, 0, 200))
    _clip(fx3 / "out_f3_2000.mp4", (0, 200, 0))
    Image.new("RGB", (90, 120), (9, 9, 9)).save(fx4 / "ref.jpg")
    r = step_a.reuse_renders(root)
    assert r["copied"] == 3 and not r["missing"]
    dst = root / "eval/fxa/results/clips10/1_standing_x"
    assert {p.name for p in dst.iterdir()} >= {"input.mp4", "ref.jpg", "out_base.mp4", "out_fx4.mp4", "out_fx3.mp4"}
    (root / "eval/fxa/clips10.json").write_text(json.dumps(owner))
    panels = step_a.parse_panels(["base=original", "fx3", "fx4=fx4", "blend05=blend 0.5"])
    assert panels[1] == ("fx3", "fx3") and panels[3] == ("blend05", "blend 0.5")
    rep = step_a.labelled_video(root, panels, repeats=1, cols=4, height=320)
    assert rep["items"] == ["1_standing_x"] and rep["panels"] == ["original", "fx3", "fx4", "blend 0.5"]
    assert rep["size"] == [180 * 4, 320 * 2]        # garment, camera, 3 present models, legend -> 2 rows of 4
    with av.open(rep["video"]) as c:
        n = sum(1 for _ in c.decode(video=0))
    assert n == 5                                   # one pass over the 5-frame clip


def test_owner_descriptions_rebuild_the_prompt(tmp_path, monkeypatch):
    from eval import step_a
    root = tmp_path
    (root / "eval/fx2/cases5").mkdir(parents=True)
    (root / "eval/fx2/cases5/clip_list.json").write_text(json.dumps(
        [{"key": "1_standing_x", "case": "standing", "video": str(root / "v/vivid_test_000001/target.mp4"),
          "ref": str(root / "nope.jpg"), "category": "tops", "name": None, "description": None}]))
    (root / "data/staging/vivid").mkdir(parents=True)
    (root / "eval/fxa").mkdir(parents=True)
    (root / "eval/fxa/descriptions.json").write_text(json.dumps({"1": "white strapless bandeau top with feathers"}))
    items = step_a.build_clip_list(root)
    assert items[0]["description_by"] == "owner" and "feathers" in items[0]["prompt"]
    assert "sleeveless" in items[0]["prompt"]          # the sleeve wording comes from the description


def test_pick_set_excludes_every_clip_used_before_and_needs_descriptions(tmp_path, capsys):
    from eval import step_a
    root = tmp_path
    vivid = root / "data/staging/vivid"
    cats = ["upper"] * 10 + ["dress"] * 10 + ["lower"] * 10
    for i, cat in enumerate(cats):
        d = vivid / f"vivid_test_{i:06d}"
        d.mkdir(parents=True)
        (d / "target.mp4").write_bytes(b"v")
        (d / "ref.jpg").write_bytes(f"garment-{i}".encode())
        (d / "meta.json").write_text(json.dumps({"split": "test", "category": cat}))
    # earlier lists: the owner's 5 (video clip 0, stand-in photo = clip 1's garment) and clips10 (person 2, donor 3)
    (root / "eval/fx2/cases5").mkdir(parents=True)
    (root / "eval/fx2/cases5/clip_list.json").write_text(json.dumps(
        [{"key": "1_x", "video": str(vivid / "vivid_test_000000/target.mp4"), "ref": str(vivid / "vivid_test_000001/ref.jpg")}]))
    (root / "eval/fxa").mkdir(parents=True)
    (root / "eval/fxa/clips10.json").write_text(json.dumps(
        [{"key": "6_h", "video": str(vivid / "vivid_test_000002/target.mp4"), "ref": str(vivid / "vivid_test_000003/ref.jpg"),
          "garment_from": "vivid_test_000003"}]))
    ids, shas = step_a.used_clip_ids(root)
    assert {"vivid_test_000000", "vivid_test_000001", "vivid_test_000002", "vivid_test_000003"} <= ids
    items = step_a.pick_set(root, "set2", n=6)
    assert len(items) == 6 and all(it["needs_description"] for it in items)
    used = {Path(it["video"]).parent.name for it in items} | {it["garment_from"] for it in items}
    assert not used & {"vivid_test_000000", "vivid_test_000001", "vivid_test_000002", "vivid_test_000003"}
    assert [it["category"] for it in items] == ["tops", "dresses", "bottoms"] * 2
    assert "photo:" in capsys.readouterr().out
    tmpl = json.loads((root / "eval/fxa/set2_descriptions.json").read_text())
    assert tmpl == {str(i): "" for i in range(1, 7)}
    # apply: refuses while any description is missing, builds the prompts once all are there
    tmpl["1"] = "red satin slip dress"
    (root / "eval/fxa/set2_descriptions.json").write_text(json.dumps(tmpl))
    with pytest.raises(SystemExit) as e:
        step_a.apply_descriptions(root, "set2")
    assert e.value.code == 2
    (root / "eval/fxa/set2_descriptions.json").write_text(json.dumps({str(i): f"garment number {i} in blue" for i in range(1, 7)}))
    done = step_a.apply_descriptions(root, "set2")
    assert all(it["description_by"] == "owner" and f"garment number {i} in blue" in it["prompt"] for i, it in enumerate(done, 1))
    assert "needs_description" not in done[0]


def test_video_without_legend_has_exactly_the_requested_panels(tmp_path):
    av = pytest.importorskip("av")
    from eval import step_a
    root = tmp_path
    d = root / "eval/fxa/results/set2/1_set2_x"
    d.mkdir(parents=True)
    _clip(d / "input.mp4", (40, 40, 40)); _clip(d / "out_blend05s08.mp4", (0, 0, 200)); _clip(d / "out_blend05s065.mp4", (0, 200, 0))
    Image.new("RGB", (90, 120), (9, 9, 9)).save(d / "ref.jpg")
    (root / "eval/fxa/set2.json").write_text(json.dumps([{"key": "1_set2_x", "case": "set2", "category": "tops"}]))
    panels = step_a.parse_panels(["blend05s08=blend 0.5 @0.8", "blend05s065=blend 0.5 @0.65"])
    rep = step_a.labelled_video(root, panels, repeats=1, cols=4, height=320, set_name="set2", legend=False)
    assert rep["size"] == [180 * 4, 320] and rep["video"].endswith("set2_labelled.mp4")   # one row: garment, camera, 2 models

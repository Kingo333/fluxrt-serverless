"""Step 3a (0811 LoRA refinements for the owner's step 1 review): garment-weighted + edge loss,
detail oversampling, masks, live-way inputs with 'remove' examples, captions, ViViD dress stills,
eval pairs, scripts. CPU only (numpy / PIL / av; no torch)."""
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "training"
PKG = ROOT / "joyai_0811_lora"
sys.path.insert(0, str(ROOT))


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = load(PKG / "losses.py", "losses_3a")


# ---------- loss ----------

def test_mask_downsample_is_area_average():
    np = pytest.importorskip("numpy")
    from PIL import Image
    m = np.zeros((8, 8), dtype=np.uint8)
    m[:, :4] = 255                      # left half garment
    m[0:2, 4:6] = 255                   # plus a quarter of the top-right 4x4 cell
    g = L.mask_to_grid(Image.fromarray(m), 2, 2)
    assert g.shape == (2, 2)
    assert g[0, 0] == pytest.approx(1.0) and g[1, 0] == pytest.approx(1.0)
    assert g[0, 1] == pytest.approx(0.25, abs=0.02) and g[1, 1] == pytest.approx(0.0)


def test_weighted_mse_puts_the_weight_on_the_garment():
    np = pytest.importorskip("numpy")
    t = np.zeros((1, 2, 1, 4, 4))
    mask = np.zeros((4, 4)); mask[:, :2] = 1.0
    p_in, p_out = t.copy(), t.copy()
    p_in[..., :, :2] = 1.0                # error only inside the garment
    p_out[..., :, 2:] = 1.0               # same error only outside
    w = L.weight_map(mask, 3.0)
    assert w.min() == 1.0 and w.max() == 3.0
    e_in, e_out = L.weighted_mse(p_in, t, w), L.weighted_mse(p_out, t, w)
    assert e_in == pytest.approx(3 * e_out) and e_in == pytest.approx(0.75)
    assert L.weighted_mse(p_in, t, 1.0) == pytest.approx(((p_in - t) ** 2).mean())   # no mask = plain MSE


def test_edge_term_penalises_smoothing_inside_the_garment_only():
    np = pytest.importorskip("numpy")
    stripes = np.zeros((4, 4)); stripes[:, ::2] = 1.0       # a rib / print pattern
    flat = np.full((4, 4), 0.5)                              # same average colour, detail gone
    mask = np.ones((4, 4))
    assert L.edge_l1(stripes, stripes, mask) == pytest.approx(0.0)
    assert L.edge_l1(flat, stripes, mask) > 0.4
    outside = np.zeros((4, 4)); outside[:, :1] = 1.0         # mask covers no horizontal pair
    assert L.edge_l1(flat, stripes, mask * 0) == pytest.approx(L.edge_l1(flat, stripes, None))   # empty mask -> frame
    terms = L.garment_loss(flat, stripes, mask, 3.0, 0.5)
    assert terms["total"] == pytest.approx(terms["mse"] + 0.5 * terms["edge"])
    assert L.garment_loss(flat, stripes, None, 3.0, 0.0)["edge"] == pytest.approx(0.0)


def test_trainer_reads_masks_by_sample_key_and_logs_terms():
    src = (PKG / "train.py").read_text()
    assert 'Path(d) / f"{key}.png"' in src and 'os.getenv("AZ_MASK_DIR")' in src
    assert 'float(os.getenv("AZ_MASK_WEIGHT", "3.0"))' in src and 'float(os.getenv("AZ_EDGE_WEIGHT", "0.5"))' in src
    assert "Loss/edge" in src and "masked {masked}/{used}" in src
    assert "generate_dmd_latents(" in src  # the live 2-step rollout is unchanged


# ---------- data build: masks + detail oversampling ----------

def _png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tar(path, items):
    with tarfile.open(path, "w") as tar:
        for key, files in items.items():
            for name, data in files.items():
                info = tarfile.TarInfo(f"{key}.{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def _meta():
    return json.dumps({"caption": "Substitute the current top with the garment in the reference image.",
                       "metadata": {"ref_image": "0.png", "source": "glance", "split": "train",
                                    "category": "upper"}}).encode()


def test_oversample_keys_top_third():
    msc = load(PKG / "make_static_clips.py", "msc_3a")
    scores = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.2, "e": 0.8, "f": 0.3}
    t = msc.oversample_keys(scores, 2)
    assert t == {"a": 2, "e": 2, "b": 1, "c": 1, "d": 1, "f": 1}
    assert set(msc.oversample_keys(scores, 1).values()) == {1}


def test_static_clips_write_cropped_masks_and_repeat_detailed_garments(tmp_path):
    pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from PIL import Image
    msc = load(PKG / "make_static_clips.py", "msc_3a2")
    plain = Image.new("RGB", (300, 400), (200, 0, 0))
    striped = np.zeros((400, 300, 3), dtype=np.uint8); striped[:, ::4] = 255; striped[:, 1::4] = 255
    person = Image.new("RGB", (600, 800), (0, 0, 200))
    target = Image.new("RGB", (600, 800), (200, 0, 0))
    _tar(tmp_path / "i2i.tar", {
        "glance_plain": {"0.png": _png(plain), "1.png": _png(person), "2.png": _png(target), "json": _meta()},
        "glance_rib": {"0.png": _png(Image.fromarray(striped)), "1.png": _png(person), "2.png": _png(target),
                       "json": _meta()},
        "glance_x": {"0.png": _png(plain), "1.png": _png(person), "2.png": _png(target), "json": _meta()},
    })
    md = tmp_path / "m"; md.mkdir()
    (md / "i2i.json").write_text(json.dumps({"resolved_files": [str(tmp_path / "i2i.tar")], "num_samples": 3}))
    stg = tmp_path / "staging" / "glance_rib"; stg.mkdir(parents=True)
    mask = np.zeros((800, 600), dtype=np.uint8); mask[400:, :] = 255     # garment = lower half
    Image.fromarray(mask).save(stg / "mask.png")
    out = tmp_path / "out"
    s = msc.build(str(md), str(out), frames=9, mode="commercial", staging_dirs=[str(tmp_path / "staging")],
                  detail_oversample=2)
    assert s["static_clips"] == 3 and s["oversampled_extra"] == 1 and s["masks"] == 1
    assert s["detail_scores"]["n"] == 3 and s["detail_scores"]["oversample"] == 2
    man = json.loads((out / "manifest" / "v2v.json").read_text())
    assert man["num_samples"] == 4
    keys = set()
    for shard in man["resolved_files"]:
        with tarfile.open(shard) as tar:
            keys |= {m.name.split(".")[0] for m in tar}
    assert {"glance_rib_s9", "glance_rib_s9_d2"} <= keys      # the detailed garment is repeated
    m = np.asarray(Image.open(out / "masks" / "glance_rib_s9.png"))
    assert m.shape == (840, 480) and m[:400].mean() < 10 and m[440:].mean() > 245   # same crop as the clip
    assert (out / "masks" / "glance_rib_s9_d2.png").exists()


# ---------- live-way inputs ----------

def _stage(root, key, cat, split="train", target="png"):
    from PIL import Image
    d = root / key
    d.mkdir(parents=True)
    Image.new("RGB", (60, 80), (10, 10, 10)).save(d / "ref.png")
    Image.new("RGB", (60, 80), (200, 200, 200)).save(d / f"target.{target}") if target == "png" else None
    (d / "meta.json").write_text(json.dumps({"category": cat, "split": split, "source": "glance"}))
    return d


def test_reference_is_another_item_of_the_same_category_and_fixed(tmp_path):
    from data_tools import live_source as lsrc
    for k, c in [("a", "upper"), ("b", "upper"), ("c", "lower"), ("d", "upper"), ("t", "upper")]:
        _stage(tmp_path, k, c, split="test" if k == "t" else "train")
    pool = lsrc.build_pool([str(tmp_path)])
    assert sorted(k for k, _ in pool["upper"]) == ["a", "b", "d"]            # test split never used
    r1 = lsrc.choose_reference("a", "upper", pool)
    assert r1[0] in ("b", "d") and lsrc.choose_reference("a", "upper", pool) == r1   # never itself, stable
    assert lsrc.choose_reference("c", "lower", pool) is None                  # no other lower item


def test_remove_examples_share_items_and_prompt():
    from data_tools import live_source as lsrc
    from data_tools.staging import KEEP, instruction
    keys = [f"k{i}" for i in range(2000)]
    picked = [k for k in keys if lsrc.remove_extra(k, "dress", 0.25)]
    assert 0.2 < len(picked) / len(keys) < 0.3
    assert all(lsrc.remove_extra(k, "dress", 0.25) in lsrc.EXTRA_ITEMS["dress"] for k in picked)
    assert lsrc.remove_extra("k1", "upper", 0.0) is None
    assert lsrc.live_prompt("upper") == instruction("upper")                 # the app's exact twin
    p = lsrc.live_prompt("lower", "a brown leather belt")
    assert p.startswith(instruction("lower")[: -len(KEEP)]) and "Also add a brown leather belt." in p and p.endswith(KEEP)


class _FakeEditor:
    name = "fake-0811"

    def __init__(self, change=True):
        self.calls = []
        self.change = change

    def edit(self, frames, prompt, size, ref_image=None):
        from PIL import Image
        self.calls.append((len(frames), prompt, ref_image is not None))
        return [Image.new("RGB", size, (160, 170, 170) if self.change else (200, 200, 200)) for _ in frames]


def test_live_source_writes_new_files_and_never_touches_old_source(tmp_path):
    from data_tools import live_source as lsrc
    a, b = _stage(tmp_path, "a", "upper"), _stage(tmp_path, "b", "upper")
    (a / "source.png").write_bytes(b"OLD")
    ed = _FakeEditor()
    st = lsrc.process([str(tmp_path)], ed, remove_frac=0.0)
    assert st["done"] == 2 and all(c[2] for c in ed.calls)                   # a reference photo every time
    assert (a / "source.png").read_bytes() == b"OLD" and (a / "live_source.png").exists()
    meta = json.loads((a / "meta.json").read_text())
    assert meta["live_source"]["ok"] and meta["live_source"]["ref_key"] == "b"
    assert lsrc.process([str(tmp_path)], ed)["skipped"] == 2                  # resumable
    c = _stage(tmp_path, "c", "upper")
    st = lsrc.process([str(tmp_path)], _FakeEditor(change=False), remove_frac=0.0)
    assert st["rejected"] == 1 and not (c / "live_source.png").exists()      # quality filter


def test_pack_prefers_live_source_only_when_asked(tmp_path):
    from data_tools import pack
    from PIL import Image
    d = _stage(tmp_path, "glance_x", "upper")
    Image.new("RGB", (60, 80), (1, 2, 3)).save(d / "source.png")
    Image.new("RGB", (60, 80), (9, 9, 9)).save(d / "live_source.png")
    meta = json.loads((d / "meta.json").read_text())
    meta["live_source"] = {"ok": True, "extra_item": "a black leather belt", "ref_key": "glance_y"}
    old = pack._sample(d, meta)
    new = pack._sample(d, meta, prefer_live=True)
    assert old.files["1.png"] == (d / "source.png").read_bytes()
    assert new.files["1.png"] == (d / "live_source.png").read_bytes()
    assert new.extra["input"] == "live_source" and new.extra["extra_item"] == "a black leather belt"
    meta["live_source"]["ok"] = False
    assert pack._sample(d, meta, prefer_live=True).files["1.png"] == (d / "source.png").read_bytes()


# ---------- captions ----------

def test_caption_tidy_and_app_format():
    from data_tools import caption_garments as cg
    from data_tools.staging import FIDELITY, instruction
    raw = "<think>looking</think> Description: \"White maxi dress, black floral print, bow tie neck, short puff " \
          "sleeves, ruffled hem, lightweight woven fabric with a smocked waist and tiered skirt panels everywhere\""
    t = cg.tidy_description(raw)
    assert t.startswith("White maxi dress") and "<think>" not in t and len(t.split()) <= 25
    full = instruction("dress", None, t)
    assert f"The reference garment: {t}" in full and FIDELITY in full
    assert "/no_think" in cg.DETAIL_PROMPT and "25 words" in cg.DETAIL_PROMPT


def test_caption_only_pairs_with_an_input(tmp_path):
    from data_tools import caption_garments as cg
    a, b = _stage(tmp_path, "a", "upper"), _stage(tmp_path, "b", "upper")
    (a / "source.png").write_bytes(b"x")

    class Cap:
        name = "stub"

        def describe(self, data, mime):
            return "Red knit top, ribbed crew neck."
    st = cg.process(str(tmp_path), Cap(), only_with_source=True)
    assert st["described"] == 1
    assert json.loads((a / "meta.json").read_text())["garment_description"].startswith("Red knit")
    assert "garment_description" not in json.loads((b / "meta.json").read_text())


# ---------- ViViD dress stills ----------

def test_vivid_dress_stills(tmp_path):
    pytest.importorskip("av")
    import av
    from PIL import Image
    from data_tools import vivid_stills as vs
    assert vs.pick_frame(200) == 48 and vs.pick_frame(9) == 4
    src = tmp_path / "vivid"
    for key, cat, split in [("vivid_train_000001", "dress", "train"), ("vivid_train_000002", "upper", "train"),
                            ("vivid_test_000003", "dress", "test")]:
        d = src / key; d.mkdir(parents=True)
        Image.new("RGB", (32, 48), (5, 5, 5)).save(d / "ref.jpg")
        with av.open(str(d / "target.mp4"), "w") as c:
            st = c.add_stream("libx264", rate=24)
            st.width, st.height, st.pix_fmt = 32, 48, "yuv420p"
            for k in range(9):
                for pkt in st.encode(av.VideoFrame.from_image(Image.new("RGB", (32, 48), (k * 25, 0, 0)))):
                    c.mux(pkt)
            for pkt in st.encode():
                c.mux(pkt)
        (d / "meta.json").write_text(json.dumps({"source": "vivid", "split": split, "category": cat}))
    out = tmp_path / "stills"
    assert vs.build(str(src), str(out))["written"] == 1
    d = out / "vivid_train_000001_still"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["source"] == "vivid" and meta["frame"] == 4 and meta["category"] == "dress"
    assert (d / "ref.jpg").exists() and (d / "target.png").exists() and not (d / "source.png").exists()
    assert vs.build(str(src), str(out))["kept_existing"] == 1


# ---------- eval pairs ----------

def test_pairs_from_keeps_the_regression_set_and_fills_across_categories(tmp_path):
    gv = load(ROOT / "eval" / "generate_vivid.py", "gv_3a")
    stg = tmp_path / "vivid"
    cats = {"vivid_test_000000": "dress", "vivid_test_000002": "dress", "vivid_test_000010": "dress",
            "vivid_test_000029": "dress", "g1": "dress", "u1": "upper", "u2": "upper", "l1": "lower", "l2": "lower"}
    for k, c in cats.items():
        (stg / k).mkdir(parents=True)
        (stg / k / "meta.json").write_text(json.dumps({"category": c}))
    prev = tmp_path / "p9e_cross"
    for p, g in [("vivid_test_000000", "g1"), ("vivid_test_000002", "vivid_test_000010"),
                 ("vivid_test_000010", "vivid_test_000029"), ("vivid_test_000029", "vivid_test_000000")]:
        (prev / p).mkdir(parents=True)
        (prev / p / "cross.json").write_text(json.dumps({"person": p, "garment": g}))
    fixed = gv.fixed_pairs(stg, prev)
    assert [(a.name, b.name) for a, b in fixed][0] == ("vivid_test_000000", "g1") and len(fixed) == 4
    cands = [(stg / "u1", stg / "u2"), (stg / "vivid_test_000000", stg / "g1"), (stg / "l1", stg / "l2"),
             (stg / "u2", stg / "u1"), (stg / "g1", stg / "vivid_test_000000"), (stg / "l2", stg / "l1")]
    got = gv.fill_pairs(fixed, cands, 7)
    assert [p.name for p, _ in got[:4]] == ["vivid_test_000000", "vivid_test_000002", "vivid_test_000010",
                                             "vivid_test_000029"]
    assert [p.name for p, _ in got[4:]] == ["g1", "l1", "u1"]                # one per category in turn
    assert len({p.name for p, _ in got}) == 7


# ---------- scripts ----------

@pytest.mark.parametrize("name", ["build_data.sh", "run_train.sh", "eval_saves.sh"])
def test_scripts_parse(name):
    assert subprocess.run(["bash", "-n", str(PKG / name)]).returncode == 0


def test_scripts_wire_the_owner_decisions():
    bd = (PKG / "build_data.sh").read_text()
    assert 'VIVID_DRESSES="${AZ_VIVID_DRESSES:-1}"' in bd and "--include \"$cfg/*\"" in bd
    assert "7ae72513413932dbfdb15d02c924f3be9739b0c4" in bd and "--prefer-live-source" in bd
    assert "joyai0811-commercial-train" in bd and "joyai0811-research-train" in bd and "--mode research" in bd
    assert "REFUSING" in bd and "--backend mimo" in bd and "timeout -s INT" in bd
    rt = (PKG / "run_train.sh").read_text()
    assert 'WITH_VIVID_DRESSES="${AZ_WITH_VIVID:-${AZ_WITH_VIVID_DRESSES:-1}}"' in rt and 'VIVID_STILLS="${AZ_VIVID_STILLS:-1}"' in bd and 'vivid_upper_stills' in rt and "joyai0811-commercial-train" in rt
    assert 'AZ_MASK_DIR="$DATA/masks"' in rt and "--detail-oversample" in rt and "eval/metrics.py" in rt
    assert 'AZ_DETAIL_CAPTION_FRAC:-${AZ_DETAILED_CAPTION_PROB:-0.5}' in rt and "joyai_0811_lora/losses.py" in rt
    ev = (PKG / "eval_saves.sh").read_text()
    assert 'ITEMS="${AZ_EVAL_ITEMS:-16}"' in ev and "--pairs-from" in ev and "live0811_ref1024 --ref-basesize 1024" in ev
    assert "--height 832" in ev and "vivid_test_000000 vivid_test_000002 vivid_test_000010 vivid_test_000029" in ev


# ---------- real-motion clips (owner 2026-11-05) ----------

def _moving(n, flicker=False):
    from PIL import Image
    frames = []
    for k in range(n):
        im = Image.new("RGB", (60, 80), (200, 200, 200))
        im.paste((40, 40, 40), (k * 2 % 40, 10, k * 2 % 40 + 15, 60))   # a person moving across
        if flicker:
            im.paste((255, 0, 0) if (k // 4) % 2 else (0, 0, 255), (0, 60, 60, 80))   # garment changes per chunk
        frames.append(im)
    return frames


def test_clip_check_rejects_flickering_inputs():
    from data_tools import live_source as lsrc
    real = _moving(16)
    assert abs(lsrc.extra_motion(real, real)) < 1e-6
    assert lsrc.extra_motion(_moving(16, flicker=True), real) > lsrc.MAX_EXTRA_MOTION
    from PIL import Image
    edited = [f.copy() for f in real]
    for e in edited:
        e.paste((90, 150, 90), (0, 60, 60, 80))                                          # steady new garment
    ok, em = lsrc.clip_check(real, edited, None, 0.05)
    assert ok.ok and em is not None and em <= lsrc.MAX_EXTRA_MOTION
    bad, em2 = lsrc.clip_check(real, _moving(16, flicker=True), None, 0.05)
    assert not bad.ok and bad.reason == "input video flickers"


def test_pack_only_live_source_skips_old_p8_clips(tmp_path):
    from data_tools import pack
    from PIL import Image
    a = _stage(tmp_path, "glance_a", "upper")
    b = _stage(tmp_path, "glance_b", "upper")
    for d in (a, b):
        Image.new("RGB", (60, 80), (1, 2, 3)).save(d / "source.png")
    Image.new("RGB", (60, 80), (9, 9, 9)).save(a / "live_source.png")
    m = json.loads((a / "meta.json").read_text())
    m["live_source"] = {"ok": True}
    (a / "meta.json").write_text(json.dumps(m))
    r = pack.pack(str(tmp_path), str(tmp_path / "out"), "t", identity_frac=0.0, only_live_source=True, split="train")
    assert r["samples"] == 1 and r["skipped_no_source"] == 1


def test_motion_first_picks_the_most_moving_clips(tmp_path):
    import importlib.util as u
    spec = u.spec_from_file_location("gv", Path(__file__).resolve().parents[2] / "training" / "eval" / "generate_vivid.py")
    gv = u.module_from_spec(spec)
    spec.loader.exec_module(gv)
    dirs = {}
    for k in ("f1", "c1", "c2", "c3"):
        d = tmp_path / k
        d.mkdir()
        (d / "target.mp4").write_bytes(b"x")
        dirs[k] = d
    scores = {"c1": 0.01, "c2": 0.09, "c3": 0.05}
    fixed = [(dirs["f1"], dirs["c1"])]
    cands = [(dirs[k], dirs["f1"]) for k in ("c1", "c2", "c3", "f1")]
    pairs, moving = gv.motion_first(fixed, cands, 2, score=lambda p: scores[p.name])
    assert [p.name for p, _ in pairs] == ["f1", "c2", "c3"] and moving == [("c2", 0.09), ("c3", 0.05)]


def test_scripts_wire_real_motion_clips():
    root = Path(__file__).resolve().parents[2] / "training" / "joyai_0811_lora"
    bd = (root / "build_data.sh").read_text()
    rt = (root / "run_train.sh").read_text()
    ev = (root / "eval_saves.sh").read_text()
    assert 'VIVID_CLIPS="${AZ_VIVID_CLIPS:-1}"' in bd and "--videos --categories" in bd and "--max-frames 97" in bd
    assert "--only-live-source --split train" in bd and "vivid_clips_live.catalog.json" in bd
    assert "--clip-repeat" in rt and 'CLIP_REPEAT="${AZ_CLIP_REPEAT:-3}"' in rt
    assert "--motion-first 4" in ev and "motion_keys.txt" in ev


def test_trainpipe_engine_lengths_and_wiring():
    """3b without the endpoint image (it hung three Pods): JoyAI's training pipeline, PyTorch template."""
    from data_tools import train_pipe_editor as tpe
    assert [tpe.clip_len(n) for n in (1, 9, 10, 97, 100)] == [9, 9, 9, 97, 97]
    assert tpe.fit_frames([1, 2], 4) == [1, 2, 2, 2] and tpe.fit_frames([1, 2, 3], 2) == [1, 2]
    root = Path(__file__).resolve().parents[2] / "training"
    bd = (root / "joyai_0811_lora" / "build_data.sh").read_text()
    assert 'ENGINE="${AZ_ENGINE:-deploy}"' in bd and "--engine trainpipe --config" in bd
    assert "setup_training_pod.sh" in bd and "make_fashion_config.py" in bd and "export AZ_MANIFEST_DIR=" in bd
    ls = (root / "data_tools" / "live_source.py").read_text()
    assert '"--engine", choices=["deploy", "trainpipe"]' in ls

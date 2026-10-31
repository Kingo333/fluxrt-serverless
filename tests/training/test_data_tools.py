"""End-to-end checks for the Azyah-VTON data tools with tiny synthetic data."""
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

from data_tools import convert_glance, manifests, pack, validate  # noqa: E402
from data_tools.shards import Sample, write_shards  # noqa: E402


def png(color, size=(300, 400)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def fake_glance(tmp: Path, rows=4):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for cfg, split in (("mens-upperwear", "train"), ("mens-upperwear", "test"), ("mens-lowerwear", "train")):
        d = tmp / "glance" / cfg
        d.mkdir(parents=True, exist_ok=True)
        cells = lambda c: [{"bytes": png((c + i * 10, 0, 0)), "path": None} for i in range(rows)]  # noqa: E731
        base = 0 if split == "train" else 100
        table = pa.table({
            "cloth": cells(base + 1), "image": cells(base + 2),
            "agnostic-mask": cells(base + 3), "image-densepose": cells(base + 4),
        })
        pq.write_table(table, d / f"{split}-00000-of-00001.parquet")
    return tmp / "glance"


def test_glance_to_shards_to_manifest(tmp_path):
    src = fake_glance(tmp_path)
    staging = tmp_path / "staging"
    n = convert_glance.convert(str(src), str(staging), ["mens-upperwear"])
    assert n == 8  # 4 train + 4 test; lowerwear skipped
    pairs = sorted(p for p in staging.iterdir())
    # Pretend the pseudo-source step ran for all but one pair.
    for d in pairs[:-1]:
        (d / "source.png").write_bytes(png((0, 200, 0)))
    out = tmp_path / "shards"
    result = pack.pack(str(staging), str(out), "glance_upper", identity_frac=0.0)
    assert result["samples"] == 7 and result["skipped_no_source"] == 1
    catalog = json.loads(Path(result["catalog"]).read_text())
    shards = [e["path"] for e in catalog]
    total, problems = 0, []
    for s in shards:
        n, p = validate.validate_shard(s)
        total += n
        problems += p
    assert total == 7 and problems == []
    # The loader keeps the garment photo only when metadata.ref_image is set.
    with tarfile.open(shards[0]) as t:
        rec = json.loads(next(t.extractfile(m).read() for m in t.getmembers() if m.name.endswith(".json")))
    assert rec["metadata"]["ref_image"] == "0.png"
    assert rec["caption"].startswith("Substitute the current top with the garment in the reference image.")
    sel = manifests.build(catalog, mode="commercial", split="train")
    assert sel["groups"]["i2i"]["num_samples"] == 3  # 4 train pairs, one still without a source
    assert sel["attribution"] and "Glance VTON Synthetic Dataset (2026)" in sel["attribution"][0]


def test_commercial_mode_excludes_research_only_sources(tmp_path):
    entries = [
        {"path": "a.tar", "group": "v2v", "source": "vivid", "split": "train", "num_samples": 10},
        {"path": "b.tar", "group": "i2i", "source": "glance", "split": "train", "num_samples": 5},
    ]
    com = manifests.build(entries, "commercial", "train")
    res = manifests.build(entries, "research", "train")
    assert com["used_sources"] == ["glance"] and "v2v" not in com["groups"]
    assert res["used_sources"] == ["glance", "vivid"] and res["groups"]["v2v"]["num_samples"] == 10


def test_own_recordings_need_consent(tmp_path):
    s = Sample(key="own1", task="r2i", files={"0.png": png(1), "1.png": png(2), "2.png": png(3)},
               caption="x", source="own", person_id="p1")
    with pytest.raises(ValueError, match="consent_id"):
        write_shards([s], tmp_path, "own")


def test_validator_catches_problems(tmp_path):
    same = png((5, 5, 5))
    bad = Sample(key="k1", task="r2i", files={"0.png": same, "1.png": png((9, 9, 9)), "2.png": same},
                 caption="c", source="glance")
    entry = write_shards([bad], tmp_path, "bad")[0]
    _, problems = validate.validate_shard(entry["path"])
    assert any("identical to target" in p for p in problems)
    tiny = Sample(key="k2", task="r2i", files={"0.png": png(1, (64, 64)), "1.png": png(2), "2.png": png(3)},
                  caption="c", source="glance")
    entry = write_shards([tiny], tmp_path / "t", "tiny")[0]
    _, problems = validate.validate_shard(entry["path"])
    assert any("too small" in p for p in problems)


def test_leakage_between_train_and_test(tmp_path):
    cat = tmp_path / "c.json"
    cat.write_text(json.dumps([
        {"split": "train", "garment_ids": ["g1", "g2"], "person_ids": ["p1"]},
        {"split": "test", "garment_ids": ["g2"], "person_ids": ["p9"]},
    ]))
    problems = validate.check_leakage([str(cat)])
    assert len(problems) == 1 and "garment leakage" in problems[0]


def test_split_comes_from_inside_the_dataset_folder():
    assert convert_glance._cfg_split(Path("mens-upperwear/train-00000.parquet")) == ("mens-upperwear", "train")
    assert convert_glance._cfg_split(Path("womens-lowerwear/test-00001.parquet")) == ("womens-lowerwear", "test")
    assert convert_glance._cfg_split(Path("data/womens-upperwear/train/0000.parquet")) == ("womens-upperwear", "train")


def test_vivid_inspect_and_stem_pairing(tmp_path):
    from data_tools import convert_vivid
    root = tmp_path / "ViViD" / "upper_body"
    (root / "videos").mkdir(parents=True)
    (root / "cloth").mkdir()
    for i in range(5):
        (root / "videos" / f"{i:05d}_00.mp4").write_bytes(b"\x00" * 2048 + bytes([i]))
        (root / "cloth" / f"{i:05d}_00.jpg").write_bytes(png((i * 40, 10, 10)))
    (root / "videos" / "orphan.mp4").write_bytes(b"\x00" * 2048)
    rep = convert_vivid.inspect(str(tmp_path))
    assert rep["videos"] == 6 and rep["cloth_images"] == 5 and rep["videos_with_same_stem_cloth"] == 5
    res = convert_vivid.convert(str(tmp_path), str(tmp_path / "staging"), "stem")
    assert res == {"pairs_found": 5, "staged": 5}
    metas = [json.loads((d / "meta.json").read_text()) for d in (tmp_path / "staging").iterdir()]
    assert all(m["source"] == "vivid" and m["task"] == "iv2v" for m in metas)


class FakeEditor:
    """Recolours the garment area (masked lower half) like a real edit would."""
    name = "fake"

    def __init__(self, recolour_everything=False):
        self.calls = []
        self.everything = recolour_everything

    def edit(self, frames, prompt, size):
        from PIL import Image
        self.calls.append((len(frames), prompt, size))
        out = []
        for f in frames:
            im = f.convert("RGB").resize(size)
            w, h = im.size
            box = (0, 0, w, h) if self.everything else (0, h // 2, w, h)
            im.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (10, 200, 30)), box[:2])
            out.append(im)
        return out


def _stage_still(staging, key):
    from PIL import Image
    from data_tools.staging import stage_pair
    mask = Image.new("L", (300, 400), 0)
    mask.paste(255, (0, 200, 300, 400))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    stage_pair(staging, key, {"ref.png": png((1, 2, 3)), "target.png": png((120, 120, 120)), "mask.png": buf.getvalue()},
               {"source": "glance", "split": "train", "category": "upper", "task": "r2i"})


def test_pseudo_source_keeps_good_edits_and_rejects_bad(tmp_path):
    from data_tools import pseudo_source
    staging = tmp_path / "staging"
    _stage_still(staging, "good")
    res = pseudo_source.process(str(staging), FakeEditor())
    assert res["done"] == 1 and (staging / "good" / "source.png").exists()
    meta = json.loads((staging / "good" / "meta.json").read_text())
    assert meta["pseudo_source"]["ok"] and meta["pseudo_source"]["prompt"].startswith("Substitute the current top with a plain")
    _stage_still(staging, "bad")
    res = pseudo_source.process(str(staging), FakeEditor(recolour_everything=True))
    assert res["rejected"] == 1 and res["skipped_has_source"] == 1
    assert not (staging / "bad" / "source.png").exists()
    assert "background" in json.loads((staging / "bad" / "meta.json").read_text())["pseudo_source"]["reason"]


def test_edit_prompt_is_reproducible_and_category_aware():
    from data_tools import pseudo_source
    assert pseudo_source.edit_prompt("k1", "upper") == pseudo_source.edit_prompt("k1", "upper")
    assert "pants or skirt" in pseudo_source.edit_prompt("k2", "lower")
    assert pseudo_source._session_size(768, 1024) == (624, 840)
    assert pseudo_source._session_size(832, 624) == (840, 624)


def test_ingest_own_respects_consent_and_holds_out_test(tmp_path):
    from data_tools import ingest_own
    raw = tmp_path / "raw"
    (raw / "garments").mkdir(parents=True)
    (raw / "consent.csv").write_text(
        "person_id,consent_id,train,test,commercial\n"
        "p01,C1,yes,yes,yes\np02,C2,yes,yes,no\np03,C3,no,no,no\np05,C5,no,yes,yes\n")
    (raw / "garments" / "garments.csv").write_text("garment_id,category,name,description\n"
                                                   "g01,upper,Tee,Navy tee\ng02,upper,Shirt,\ng09,upper,Test top,\n")
    for g in ("g01", "g02", "g09"):
        (raw / "garments" / f"{g}.jpg").write_bytes(png((3, 3, 3)))
    for person, garment in (("p01", "g01"), ("p02", "g02"), ("p03", "g01"), ("p05", "g09"), ("p04", "g01")):
        d = raw / "clips" / person / garment
        d.mkdir(parents=True)
        (d / "phone.mp4").write_bytes(b"\x00" * 4096)
    r = ingest_own.ingest(str(raw), str(tmp_path / "st"), test_people={"p05"}, test_garments={"g09"})
    assert r["staged"] == {"train": 2, "test": 1} and r["leakage"] == 0
    assert any("p03" in s and "does not cover training" in s for s in r["skipped"])
    assert any("p04" in s and "no consent row" in s for s in r["skipped"])
    metas = {d.name: json.loads((d / "meta.json").read_text()) for d in (tmp_path / "st").iterdir()}
    sources_used = {m["person_id"]: m["source"] for m in metas.values()}
    assert sources_used["own-p02"] == "own_research" and sources_used["own-p01"] == "own"
    assert all(m["consent_id"] for m in metas.values())


def test_mask_box_picker():
    from data_tools.masks import pick_box
    assert pick_box([[0, 0, 1, 1], [1, 1, 2, 2]], [0.2, 0.9]) == ([1.0, 1.0, 2.0, 2.0], 0.9)
    assert pick_box([[0, 0, 1, 1]], [0.1]) is None


def test_training_caption_matches_the_app_instruction_exactly():
    # Same case and expected string as azyah-shopping
    # src/components/ai-studio/live-cam/__tests__/buildTryOnInstruction.test.ts (review R2).
    from data_tools.staging import instruction
    assert instruction("upper", "Striped linen shirt", "Navy and white vertical stripes, mandarin collar, two chest pockets") == (
        "Substitute the current top with the garment in the reference image. "
        "The reference garment is the Striped linen shirt: Navy and white vertical stripes, mandarin collar, two chest pockets. "
        "Match its exact color, pattern, print, logo, fabric and construction details. "
        "Keep the person's face, body, pose and the background unchanged."
    )
    assert instruction("upper") == (
        "Substitute the current top with the garment in the reference image. "
        "Keep the person's face, body, pose and the background unchanged."
    )
    assert instruction("upper", None, "Red tee!").startswith(
        "Substitute the current top with the garment in the reference image. The reference garment: Red tee.")


class FakeCaptioner:
    name = "fake-vlm"

    def describe(self, image_bytes, mime):
        return "Navy crew-neck tee with a white chest logo. <b>Soft</b> jersey."


def test_captioner_then_pack_writes_both_caption_styles(tmp_path):
    from data_tools import caption_garments
    staging = tmp_path / "st"
    _stage_still(staging, "p1")
    (staging / "p1" / "source.png").write_bytes(png((0, 200, 0)))
    res = caption_garments.process(str(staging), FakeCaptioner())
    assert res == {"described": 1, "skipped": 0, "failed": 0}
    meta = json.loads((staging / "p1" / "meta.json").read_text())
    assert meta["garment_description"] == "Navy crew-neck tee with a white chest logo. Soft jersey."
    out = pack.pack(str(staging), str(tmp_path / "sh"), "t")
    shard = json.loads(Path(out["catalog"]).read_text())[0]["path"]
    with tarfile.open(shard) as t:
        rec = json.loads(next(t.extractfile(m).read() for m in t.getmembers() if m.name.endswith(".json")))
    assert rec["caption"].startswith("Substitute the current top") and "Navy" not in rec["caption"]
    assert "The reference garment: Navy crew-neck tee with a white chest logo. Soft jersey." in rec["caption_detailed"]
    assert caption_garments.process(str(staging), FakeCaptioner())["skipped"] == 1


def test_identity_pairs_use_the_real_target_as_clean_input(tmp_path):
    staging = tmp_path / "st"
    for i in range(20):
        _stage_still(staging, f"p{i:02d}")
        (staging / f"p{i:02d}" / "source.png").write_bytes(png((0, 200, 0)))
    out = pack.pack(str(staging), str(tmp_path / "sh"), "t", identity_frac=0.5)
    assert 0 < out["identity_pairs"] < 20 and out["samples"] == 20 + out["identity_pairs"]
    shards = [e["path"] for e in json.loads(Path(out["catalog"]).read_text())]
    found = 0
    for s in shards:
        with tarfile.open(s) as t:
            members = {m.name: t.extractfile(m).read() for m in t.getmembers()}
        for name, data in members.items():
            if name.endswith("_id.json"):
                key = name[:-len(".json")]
                assert json.loads(data)["metadata"]["identity_pair"] is True
                assert members[f"{key}.1.png"] == members[f"{key}.2.png"]
                found += 1
        _, problems = validate.validate_shard(s)
        assert problems == []
    assert found == out["identity_pairs"]


def test_vivid_zip_inspect_and_sampled_extract_across_groups(tmp_path):
    import zipfile as zf
    from data_tools import convert_vivid, vivid_zip
    zp = tmp_path / "ViViD.zip"
    with zf.ZipFile(zp, "w") as z:
        for group in ("upper_body", "dresses"):
            for i in range(6):
                z.writestr(f"ViViD/{group}/videos/{i:05d}.mp4", b"\x00" * 2048 + bytes([i]))
                z.writestr(f"ViViD/{group}/cloth/{i:05d}.jpg", png((i * 30, 5, 5)))
        z.writestr("ViViD/upper_body/videos/orphan.mp4", b"\x00" * 2048)
    rep = vivid_zip.inspect(str(zp))
    assert rep["videos"] == 13 and rep["pairs_by_stem"] == 12
    out = vivid_zip.extract(str(zp), str(tmp_path / "sub"), sample=5, seed=1)
    assert out == {"pairs_available": 12, "extracted": 5}
    res = convert_vivid.convert(str(tmp_path / "sub"), str(tmp_path / "st"), "stem")
    assert res["staged"] == 5
    cats = {json.loads((d / "meta.json").read_text())["category"] for d in (tmp_path / "st").iterdir()}
    assert cats <= {"upper", "dress"}


def test_glance_per_config_subset_is_balanced(tmp_path):
    src = fake_glance(tmp_path, rows=6)
    n = convert_glance.convert(str(src), str(tmp_path / "st"), ["mens-upperwear", "mens-lowerwear"], per_config=2)
    metas = [json.loads((d / "meta.json").read_text()) for d in (tmp_path / "st").iterdir()]
    by = {}
    for m in metas:
        by[(m["config"], m["split"])] = by.get((m["config"], m["split"]), 0) + 1
    assert by == {("mens-upperwear", "train"): 2, ("mens-upperwear", "test"): 1, ("mens-lowerwear", "train"): 2}
    assert n == 5

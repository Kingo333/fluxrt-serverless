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
    result = pack.pack(str(staging), str(out), "glance_upper")
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

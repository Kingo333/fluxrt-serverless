"""inventory.py: counts on the training volume (fxreal6 plan, step 3). Standard library only."""
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
from data_tools import inventory as inv  # noqa: E402


def _stage(root: Path, name: str, key: str, meta: dict, files=("ref.jpg", "target.mp4")):
    d = root / "data" / "staging" / name / key
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta))
    for f in files:
        (d / f).write_bytes(b"x")


def _volume(tmp_path: Path) -> Path:
    root = tmp_path / "azyah-vton"
    _stage(root, "vivid", "a", {"category": "dress", "split": "train", "garment_description": "red dress"})
    _stage(root, "vivid", "b", {"category": "upper", "split": "train"}, files=("ref.jpg", "target.mp4", "source.png"))
    _stage(root, "vivid", "c", {"category": "dress", "split": "heldout"})
    (root / "data/staging/vivid/broken").mkdir()
    (root / "data/staging/vivid/broken/meta.json").write_text("{not json")
    raw = root / "data/raw"
    (raw / "viton_hd").mkdir(parents=True)
    (raw / "viton_hd/.downloaded").write_text("")
    (raw / "viton_hd/train-0.parquet").write_bytes(b"p")
    with zipfile.ZipFile(raw / "ViViD.zip", "w") as z:
        for n in ("ViViD/dresses/videos/1.mp4", "ViViD/dresses/videos/2.mp4", "ViViD/upper_body/videos/3.mp4",
                  "ViViD/lower_body/videos/4.mp4", "ViViD/dresses/cloth/1.jpg"):
            z.writestr(n, b"v")
    ds = root / "data/fluxrt_lora/fxreal5"
    ds.mkdir(parents=True)
    (ds / "dataset_summary.json").write_text(json.dumps({"samples": 4321}))
    fav = root / "outputs/fluxrt_lora/blend05s065"
    fav.mkdir(parents=True)
    (fav / "pytorch_lora_weights.safetensors").write_bytes(b"weights")
    (fav / "selection.json").write_text("{}")
    (root / "checkpoints/FLUX.2-klein-base-4B").mkdir(parents=True)
    ev = root / "eval"
    (ev / "fxa").mkdir(parents=True)
    (ev / "fxa/dress10.json").write_text(json.dumps([1, 2, 3]))
    (ev / "fxa/dress10_descriptions.json").write_text("{}")
    (ev / "hijab_clips").mkdir()
    (ev / "hijab_clips/h1.mp4").write_bytes(b"v")
    return root


def test_counts_every_part_of_the_volume(tmp_path):
    r = inv.inventory(_volume(tmp_path))
    v = r["staging"]["vivid"]
    assert v["items"] == 3 and v["by_type"] == {"dress": 2, "upper": 1}
    assert v["by_split"] == {"heldout": 1, "train": 2}
    assert v["with_video"] == 3 and v["with_input"] == 1 and v["with_description"] == 1
    z = r["zips"]["data/raw/ViViD.zip"]
    assert z["videos"] == 4 and z["by_type"] == {"dress": 2, "lower": 1, "upper": 1}
    assert r["viton_hd"]["downloaded"] is True and r["viton_hd"]["parquet_files"] == 1
    assert r["datasets_built"]["fxreal5"]["samples"] == 4321
    fav = r["loras"]["blend05s065"]
    assert fav["selection_json"] is True and len(fav["sha256"]) == 64
    assert "FLUX.2-klein-base-4B" in r["models"]
    assert r["eval_sets"] == {"fxa/dress10.json": 3}
    assert r["hijab_clips"] == 1 and r["own_clips"] == 0
    assert r["disk"]["free_gb"] > 0


def test_prints_a_table_and_one_json_line(tmp_path, capsys):
    assert inv.main([str(_volume(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "vivid: 3; {'dress': 2, 'upper': 1}" in out
    line = [l for l in out.splitlines() if l.startswith("INVENTORY_JSON=")]
    assert len(line) == 1 and json.loads(line[0].split("=", 1)[1])["own_clips"] == 0


def test_missing_volume_and_empty_parts_do_not_crash(tmp_path, capsys):
    assert inv.main([str(tmp_path / "nope")]) == 1
    assert "NO_VOLUME" in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    r = inv.inventory(tmp_path / "empty")
    assert r["staging"] == {} and r["zips"] == {} and r["loras"] == {} and r["eval_sets"] == {}
    assert inv.probe_sample(tmp_path / "empty", 5) == {"probed": 0}
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    assert "error" in inv.zip_videos(bad)

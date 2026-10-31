import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from data_tools import manifests  # noqa: E402
from data_tools.fluxrt_prompt import fluxrt_prompt  # noqa: E402
from data_tools.shards import Sample, write_shards  # noqa: E402
from fluxrt_lora import make_dataset  # noqa: E402


def png(color, size=(300, 400)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def make_manifest(tmp_path, n=3):
    samples = [Sample(key=f"k{i}", task="r2i", source="glance", caption="x",
                      files={"0.png": png((255, 0, 0), (500, 700)), "1.png": png((0, 255, 0)), "2.png": png((0, 0, 255))},
                      extra={"category": "upper"}) for i in range(n)]
    cat = write_shards(samples, tmp_path / "shards", "t", per_shard=2)
    cat += write_shards([Sample(key="noref", task="i2i", source="glance", caption="x",
                                files={"0.png": png((1, 1, 1)), "1.png": png((2, 2, 2))})], tmp_path / "shards", "u")
    sel = manifests.build(cat, "research", "train")
    manifests.write(sel, tmp_path / "m")
    return tmp_path / "m"


def test_make_dataset_loads_in_hf_datasets(tmp_path):
    datasets = pytest.importorskip("datasets")
    mdir = make_manifest(tmp_path)
    summary = make_dataset.main(["--manifest-dir", str(mdir), "--out", str(tmp_path / "out"), "--per-file", "2"])
    assert summary["samples"] == 3 and summary["files"] == 2  # the i2i sample without a garment is skipped
    assert (tmp_path / "out" / "selection.json").exists()
    ds = datasets.load_dataset(summary["dataset_name"])
    row = ds["train"][0]
    assert set(ds["train"].column_names) == {"key", "caption", "person", "garment", "target"}
    assert row["person"].size == (288, 512) and row["target"].size == (288, 512)
    assert row["garment"].size == (500, 700)
    assert row["caption"] == fluxrt_prompt("tops")
    r, g, b = row["target"].convert("RGB").getpixel((10, 10))
    assert b > 200 and r < 50


def test_patch_applies_to_pinned_diffusers():
    trainer = Path("/home/user/repos/diffusers/examples/dreambooth/train_dreambooth_lora_flux2_klein_img2img.py")
    if not trainer.exists():
        pytest.skip("diffusers checkout not present")
    patch = ROOT / "training" / "patches" / "flux2-klein-garment.patch"
    repo = trainer.parents[2]
    r = subprocess.run(["git", "-C", str(repo), "apply", "--check", "-R", str(patch)], capture_output=True, text=True)
    r2 = subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch)], capture_output=True, text=True)
    assert r.returncode == 0 or r2.returncode == 0, r.stderr + r2.stderr


def test_run_script_args():
    sh = (ROOT / "training" / "fluxrt_lora" / "run_train.sh").read_text()
    for flag in ("--cond_image_column=person", "--cond_image_column_2=garment", "--image_column=target",
                 "--caption_column=caption", "--checkpointing_steps", "--resume_from_checkpoint=latest"):
        assert flag in sh
    assert "--random_flip" not in sh  # flipping would mirror logos against the unflipped garment


def _mp4(tmp_path, name, color):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    out = tmp_path / name
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:s=320x576:d=1:r=12",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out.read_bytes()


def test_video_stills_from_clip_pairs(tmp_path):
    jpg = io.BytesIO()
    Image.new("RGB", (300, 400), (255, 0, 0)).save(jpg, format="JPEG")
    s = Sample(key="v1", task="iv2v", source="vivid", caption="x",
               files={"0.jpg": jpg.getvalue(), "0.mp4": _mp4(tmp_path, "a.mp4", "green"), "1.mp4": _mp4(tmp_path, "b.mp4", "blue")},
               extra={"category": "dress"})
    cat = write_shards([s], tmp_path / "shards", "v")
    manifests.write(manifests.build(cat, "research", "train"), tmp_path / "m")
    items = list(make_dataset.iter_video_stills(tmp_path / "m", 3))
    assert [k for k, *_ in items] == ["v1_f0", "v1_f1", "v1_f2"]
    _, g, p, t, meta = items[0]
    person, garment, target = make_dataset.convert(g, p, t, (288, 512))
    r, gg, b = Image.open(io.BytesIO(target)).getpixel((5, 5))
    assert b > 150 and r < 80
    assert make_dataset.caption_for(meta) == fluxrt_prompt("dresses")
    sel = json.loads((tmp_path / "m" / "selection.json").read_text())
    assert sel["trained_with_vivid"] is True and sel["commercial_use_cleared"] is False

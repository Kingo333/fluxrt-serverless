"""Half-written files a stopped Pod leaves (2026-11-22): they must be found and moved aside, never taken for a
finished render or segmentation, and never deleted."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))


def _npz(p: Path, whole: bool = True):
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, cat=np.zeros((3, 8, 8), np.uint8))
    if not whole:
        b = p.read_bytes()
        p.write_bytes(b[: len(b) // 2])


def _job(yard: Path, name: str, sets=("setx",)):
    d = yard / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"name": name, "model": "fav2", "seed": 54}))
    for m in (".rendered", ".scored"):
        (d / m).touch()
    for s in sets:
        (d / s).mkdir()
        for f in ("garment.json", "face_skin.json", "stability.json"):
            (d / s / f).write_text("{}")
    return d


def test_a_half_segmentation_file_is_moved_aside_and_its_model_is_done_again(tmp_path, capsys):
    from eval.half_files import check
    res, yard = tmp_path / "results", tmp_path / "yard"
    clip = res / "setx/c1"
    _npz(clip / "seg_camera.npz")
    _npz(clip / "seg_fav2seed54_r528.npz", whole=False)
    _npz(clip / "seg_other.npz", whole=False)                                  # a model the yard does not know
    job = _job(yard, "fav2seed54_r528")
    (job / ".failed").touch()
    r = check([res], yard, videos=False)
    assert sorted(Path(b).name for b in r["bad"]) == ["seg_fav2seed54_r528.npz", "seg_other.npz"] and r["again"] == ["fav2seed54_r528"]
    assert (clip / "seg_fav2seed54_r528.npz.bad").exists() and not (clip / "seg_fav2seed54_r528.npz").exists()
    assert (clip / "seg_camera.npz").exists()
    # the model's marks are set aside (kept as .old), its scores stay: they were made before the file broke
    assert not (job / ".scored").exists() and (job / ".scored.old").exists() and (job / ".failed.old").exists()
    assert (job / "setx/garment.json").exists()
    assert "HALVES 3 files written in the last 72 hours opened (videos NOT checked): 2 cannot be read and were moved aside" in capsys.readouterr().out
    # nothing left to find the second time
    assert check([res], yard, videos=False)["bad"] == []


def test_only_recent_files_are_opened_and_a_dry_run_changes_nothing(tmp_path, capsys):
    from eval.half_files import main
    res = tmp_path / "results"
    old, new, now = res / "setx/c1/seg_old.npz", res / "setx/c1/seg_new.npz", res / "setx/c1/seg_now.npz"
    for p in (old, new, now):
        _npz(p, whole=False)
    t = time.time()
    os.utime(old, (t - 100 * 3600, t - 100 * 3600))                             # four days old: read by an earlier scoring
    os.utime(new, (t - 3600, t - 3600))
    r = main([str(res), str(tmp_path / "nosuchfolder"), "--no-videos", "--min-age", "120", "--dry-run"])
    assert [Path(b).name for b in r["bad"]] == ["seg_new.npz"] and new.exists() and not (res / "setx/c1/seg_new.npz.bad").exists()
    assert "1 cannot be read (listed only): " in capsys.readouterr().out
    r = main([str(res), "--no-videos", "--hours", "1000"])
    assert sorted(Path(b).name for b in r["bad"]) == ["seg_new.npz", "seg_now.npz", "seg_old.npz"]


def test_a_video_cut_off_in_the_middle_is_found_and_the_clip_is_rendered_and_scored_again(tmp_path, capsys):
    av = pytest.importorskip("av")
    from PIL import Image
    from eval.half_files import check, readable

    def mp4(p: Path, n: int = 12):
        p.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(p), "w") as c:
            st = c.add_stream("libx264", rate=24)
            st.width, st.height, st.pix_fmt = 64, 96, "yuv420p"
            for k in range(n):
                for pkt in st.encode(av.VideoFrame.from_image(Image.new("RGB", (64, 96), (10 * k, 80, 120)))):
                    c.mux(pkt)
            for pkt in st.encode():
                c.mux(pkt)

    res, yard = tmp_path / "results", tmp_path / "yard"
    clip = res / "setx/c1"
    mp4(clip / "input.mp4")
    mp4(clip / "out_fav2_r528.mp4")
    mp4(clip / "out_fav2seed54_r528.mp4")
    whole = (clip / "out_fav2seed54_r528.mp4").read_bytes()
    (clip / "out_fav2seed54_r528.mp4").write_bytes(whole[: len(whole) * 2 // 3])    # the Pod stopped while it was written
    (clip / "empty.mp4").write_bytes(b"")
    assert readable(clip / "input.mp4", av) and not readable(clip / "out_fav2seed54_r528.mp4", av)
    job = _job(yard, "fav2seed54_r528", sets=("setx", "sety"))
    r = check([res], yard)
    assert r["videos_checked"] and sorted(Path(b).name for b in r["bad"]) == ["empty.mp4", "out_fav2seed54_r528.mp4"]
    assert (clip / "out_fav2seed54_r528.mp4.bad").read_bytes() == whole[: len(whole) * 2 // 3]      # kept, not deleted
    assert (clip / "input.mp4").exists() and (clip / "out_fav2_r528.mp4").exists()
    # the clip is rendered again (no .rendered) and that set scored again (its scores set aside); the other set stays
    assert not (job / ".rendered").exists() and not (job / ".scored").exists()
    assert not (job / "setx/garment.json").exists() and (job / "setx/garment.json.old").exists() and (job / "sety/garment.json").exists()
    assert "HALVES 4 files written in the last 72 hours opened: 2 cannot be read and were moved aside to be made again" in capsys.readouterr().out

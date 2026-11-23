"""The same garments as really worn (2026-11-22): the real videos of the garment photos' own clips, laid out like a
render and scored by the usual garment scorer alone, then read by the yardstick next to favourite #2. Real video
files and the real scorer (the segmenter's output is written by hand: MediaPipe is not needed for the sums)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
W, H, N = 72, 128, 10
BACKGROUND, FACE, CLOTHES = 0, 3, 4


def _dress(light: int, dark: int, w: int = 160, h: int = 120, shift: int = 0):
    """A landscape frame: grey room, a person in the middle with a face and a red striped dress (light and dark
    stripes: the fabric's light-dark spread)."""
    from PIL import Image
    px = np.full((h, w, 3), 128, np.uint8)
    x0, x1 = w // 2 - 18 + shift, w // 2 + 18 + shift
    px[10:28, x0 + 8:x1 - 8] = (220, 180, 160)                                   # the face
    for y in range(30, h - 6):
        px[y, x0:x1] = (light, 30, 40) if (y // 6) % 2 else (dark, 20, 30)       # the dress
    return Image.fromarray(px)


def _photo(light: int = 150, dark: int = 135):
    """A garment photo: white backdrop, the dress in the middle with faint stripes (little light-dark spread)."""
    from PIL import Image
    px = np.full((120, 90, 3), 246, np.uint8)
    for y in range(12, 110):
        px[y, 20:70] = (light, 25, 35) if (y // 6) % 2 else (dark, 20, 30)
    return Image.fromarray(px)


def _mp4(path: Path, frames):
    import av
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = frames[0].width, frames[0].height, "yuv420p"
        st.options = {"crf": "12"}
        for im in frames:
            for pkt in st.encode(av.VideoFrame.from_image(im)):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def _n_frames(path: Path) -> tuple[int, int, int]:
    import av
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        return sum(1 for _ in c.decode(v)), v.width, v.height


@pytest.fixture
def volume(tmp_path):
    pytest.importorskip("av")
    from PIL import Image
    root = tmp_path / "azyah-vton"
    vivid = root / "data/staging/vivid"
    for name in ("donor_a", "donor_b", "person_a"):
        (vivid / name).mkdir(parents=True)
    # donor a: its photo is an evenly lit red dress, its video shows the dress in strong light and shade (worn)
    _photo().save(vivid / "donor_a/ref.jpg", quality=95)
    _mp4(vivid / "donor_a/target.mp4", [_dress(200, 100, shift=k % 3) for k in range(30)])
    _photo().save(vivid / "donor_b/ref.jpg", quality=95)
    _mp4(vivid / "donor_b/target.mp4", [_dress(200, 100)])                         # one frame only: no video to speak of
    _mp4(vivid / "person_a/target.mp4", [_dress(90, 60) for _ in range(12)])
    own = tmp_path / "owner_photos"
    own.mkdir()
    Image.fromarray(np.full((120, 90, 3), (20, 60, 150), np.uint8)).save(own / "blue.png")
    person = str(vivid / "person_a/target.mp4")
    items = [{"key": "1_setx_person_a", "case": "setx", "video": person, "ref": str(vivid / "donor_a/ref.jpg"), "category": "dresses",
              "garment_from": "donor_a"},
             {"key": "2_setx_person_a", "case": "setx", "video": person, "ref": str(vivid / "donor_b/ref.jpg"), "category": "dresses",
              "garment_from": "donor_b"},
             {"key": "3_owner", "case": "setx", "video": person, "ref": str(own / "blue.png"), "category": "dresses"}]
    (root / "eval/fxa").mkdir(parents=True)
    (root / "eval/fxa/setx.json").write_text(json.dumps(items))
    return root


def test_the_real_video_of_a_garment_is_laid_out_like_a_render_of_it(volume, capsys):
    from eval.worn_reference import build, main
    rep = main([str(volume), "--sets", "setx", "nosuchset", "--frames", str(N), "--size", f"{W}x{H}"])
    out = capsys.readouterr().out
    # the donor with a video is used; the one-frame video and the owner's own photo are not
    assert rep == {"setx": {"clips": 1, "new": 1, "without_a_video": 2}}
    assert "WORN_SETS setx: 1 real videos of the test garments (1 new, 2 garments have no video)" in out
    d = volume / "eval/fxa/worn/setx/1_setx_person_a"
    assert sorted(p.name for p in d.iterdir()) == ["input.mp4", "item.json", "out_worn.mp4", "ref.jpg"]
    assert _n_frames(d / "input.mp4") == (N, W, H) and _n_frames(d / "out_worn.mp4") == (N, W, H)   # cut and sized like a test clip
    item = json.loads((d / "item.json").read_text())
    assert item["worn_from"] == "donor_a" and item["category"] == "dresses" and item["key"] == "1_setx_person_a"
    assert not (volume / "eval/fxa/worn/setx/2_setx_person_a").exists() and not (volume / "eval/fxa/worn/setx/3_owner").exists()
    # it shows the DONOR's video (the bright striped dress), not the test person's
    import av
    with av.open(str(d / "out_worn.mp4")) as c:
        frame = np.asarray(next(c.decode(video=0)).to_image().convert("RGB"))
    assert frame[H // 2:, W // 2 - 4:W // 2 + 4, 0].max() > 170
    # started again: kept as it is
    stamp = (d / "out_worn.mp4").stat().st_mtime_ns
    assert build(volume, ["setx"], N, (W, H)) == {"setx": {"clips": 1, "new": 0, "without_a_video": 2}}
    assert (d / "out_worn.mp4").stat().st_mtime_ns == stamp
    # nothing at all: said, not an error
    assert build(volume, ["nosuchset"], N, (W, H)) == {} and "WORN_SETS none" in capsys.readouterr().out


def test_the_usual_scorer_measures_the_real_video_alone_and_the_yardstick_reads_it(volume, tmp_path, capsys):
    from eval import garment_report, yardstick
    from eval.worn_reference import build
    # three test clips with the same donor garment, so the real videos make a line of their own
    items = json.loads((volume / "eval/fxa/setx.json").read_text())
    one = items[0]
    (volume / "eval/fxa/setx.json").write_text(json.dumps([{**one, "key": f"{k}_setx_person_a"} for k in (1, 2, 3)]))
    build(volume, ["setx"], N, (W, H))
    worn = volume / "eval/fxa/worn/setx"
    # the segmenter's part, by hand: where the person, the face and the clothes are in every frame of the real video
    import av
    for d in sorted(worn.iterdir()):
        with av.open(str(d / "out_worn.mp4")) as c:
            frames = [np.asarray(f.to_image().convert("RGB")).astype(int) for f in c.decode(video=0)]
        cats = []
        for f in frames:
            cat = np.full(f.shape[:2], BACKGROUND, np.uint8)
            cat[(f[..., 0] > 70) & (f[..., 1] < 70)] = CLOTHES                    # the red dress
            cat[(f[..., 0] > 190) & (f[..., 1] > 140)] = FACE
            cats.append(cat)
        np.savez_compressed(d / "seg_camera.npz", cat=np.stack(cats))
        np.savez_compressed(d / "seg_worn.npz", cat=np.stack(cats))
    yard = volume / "eval/fxa/yard"
    (yard / "worn/setx").mkdir(parents=True)
    res = garment_report.main(["score", str(worn), "--models", "worn", "--baseline", "worn", "--no-dino",
                               "--out", str(yard / "worn/setx/garment.md"), "--json", str(yard / "worn/setx/garment.json")])
    assert res["scorecard"] == [] and set(res["items"]) == {f"{k}_setx_person_a" for k in (1, 2, 3)}
    rec = res["items"]["1_setx_person_a"]["worn"]
    # the photo is evenly lit, the worn dress stands in strong light and shade: far more light-dark spread than the photo
    assert rec["valid_frames"] == N and rec["fabric_ratio"] > 3 and rec["fabric"] == pytest.approx(abs(np.log(rec["fabric_ratio"])), abs=1e-3)
    assert 0 < rec["colour_de"] < 10 and rec["on_body"] > 0.9 and "flicker_de" in rec and "jitter" in rec and "length_ratio" in rec
    # a yard with favourite #2's own numbers on the same clips: its dress is nearer the photo's flat red
    capsys.readouterr()
    e = yard / "fav2seed54_r528"
    (e / "setx").mkdir(parents=True)
    (e / "meta.json").write_text(json.dumps({"name": "fav2seed54_r528", "label": "seed 54", "model": "fav2", "seed": 54}))
    fav = {"colour_de": rec["colour_de"] + 4.0, "fabric": 0.2, "fabric_ratio": 1.22, "jitter": 0.03, "flicker_de": 1.5}
    (e / "setx/garment.json").write_text(json.dumps({"items": {c: {"fav2seed54_r528": fav, "fav2_r528": fav} for c in res["items"]}}))
    (e / "setx/face_skin.json").write_text(json.dumps({"per_clip": {}}))
    (e / "setx/stability.json").write_text("{}")
    out = yardstick.main([str(yard), "--baseline", "fav2_r528", "--old", "setx", "--clips", str(volume / "eval/fxa"),
                          "--worn", str(yard / "worn"), "--boot", "50"])
    text = capsys.readouterr().out
    line = next(l for l in text.splitlines() if l.startswith("YARD_REAL "))
    assert "the same garments as really worn (3 real videos) next to favourite #2 at seed 52 on the same garments: " in line
    assert f"colour distance from the photo: real {rec['colour_de']:.1f}, favourite #2 {rec['colour_de'] + 4.0:.1f}" in line
    assert f"light-dark spread (1 = the photo's): real x{rec['fabric_ratio']:.2f}, favourite #2 x1.22" in line
    assert "colour flicker: real " in line and "shape jitter: real " in line
    assert "YARD_REAL_GARMENT dresses, 3 real videos:" in text
    assert out["real"]["all"]["fabric_ratio"]["real"] == pytest.approx(rec["fabric_ratio"], rel=1e-3)
    assert f"; the same garments as really worn: x{rec['fabric_ratio']:.2f}" in text

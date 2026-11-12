"""fxreal4: longer-to-shorter examples and the item-5-only video."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))


def test_extend_down_adds_a_band_below_the_hem_only():
    np = pytest.importorskip("numpy")
    from PIL import Image
    from fluxrt_lora import build_real
    m = np.zeros((100, 60), dtype=np.uint8)
    m[20:50, 15:45] = 255                      # the garment: rows 20-49, columns 15-44
    whole, band = build_real.extend_down(Image.fromarray(m, "L"), frac=0.2, widen=0.05)
    w, b = np.asarray(whole) > 0, np.asarray(band) > 0
    rows = np.where(b.any(axis=1))[0]
    assert rows.min() == 50 and rows.max() == 69   # 20% of 100 rows, starting right under the hem
    cols = np.where(b.any(axis=0))[0]
    assert cols.min() == 12 and cols.max() == 47   # the hem's width plus 5% of 60 px each side
    assert w[20:50, 15:45].all() and not b[20:50].any()  # garment kept, band only below it
    assert not w[:20].any()                        # nothing added above the garment


def test_extend_down_stops_at_the_bottom_edge_and_handles_an_empty_mask():
    np = pytest.importorskip("numpy")
    from PIL import Image
    from fluxrt_lora import build_real
    m = np.zeros((50, 40), dtype=np.uint8)
    m[30:48, 10:30] = 255
    whole, band = build_real.extend_down(Image.fromarray(m, "L"), frac=0.5)
    assert (np.asarray(band) > 0)[48:].any() and np.asarray(whole).shape == (50, 40)
    empty, eband = build_real.extend_down(Image.new("L", (10, 10), 0))
    assert not (np.asarray(empty) > 0).any() and not (np.asarray(eband) > 0).any()


def test_longer_prompt_asks_for_a_longer_garment_and_keeps_the_person():
    from fluxrt_lora import build_real
    p = build_real.LONGER_PROMPT.lower()
    assert "longer" in p and "thighs" in p and "face" in p


def test_fx4_script_uses_the_new_settings():
    sh = (ROOT / "training" / "fluxrt_lora" / "fx4.sh").read_text()
    assert 'EXP="${AZ_EXP:-fxreal4}"' in sh and "--longer-frac" in sh
    assert "AZ_DETAIL_WEIGHT=${AZ_DETAIL_WEIGHT:-0.03}" in sh
    assert "--name item5_fx4 --only 5" in sh and "--name five_test_fx4" in sh
    assert "fx3" not in sh.replace("fxreal3", "")


def test_builder_makes_longer_examples_and_their_inputs_differ_below_the_hem(tmp_path):
    import io
    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image
    from fluxrt_lora import build_real

    def jb(img):
        b = io.BytesIO(); img.save(b, format="JPEG"); return {"bytes": b.getvalue(), "path": "x.jpg"}
    t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    rows = {"cloth": [], "image": [], "agnostic": []}
    for i in range(6):
        person = Image.new("RGB", (768, 1024), (200, 30 + i * 20, 40))
        agn = person.copy(); agn.paste((128, 128, 128), (200, 300, 560, 600))   # top ends at y=600
        rows["cloth"].append(jb(Image.new("RGB", (300, 400), (i * 30, 90, 200))))
        rows["image"].append(jb(person)); rows["agnostic"].append(jb(agn))
    (tmp_path / "viton").mkdir()
    pq.write_table(pa.table({k: pa.array(v, type=t) for k, v in rows.items()}), tmp_path / "viton" / "a.parquet")
    prompts = []

    class FakeRedress:
        def __init__(self, *a, **k): pass
        def __call__(self, person, garment, seed, category="tops", prompt=None):
            prompts.append(prompt)
            return Image.new("RGB", (432, 768), (0, 255, 0))   # a long green garment everywhere

    class FakeDino:
        def __call__(self, a, b): return 0.5

    old = build_real.Redresser, build_real.DinoSim
    build_real.Redresser, build_real.DinoSim = FakeRedress, FakeDino
    try:
        s = build_real.main(["--out", str(tmp_path / "data"), "--viton-dir", str(tmp_path / "viton"), "--klein", "x",
                             "--test-out", str(tmp_path / "pairs"), "--viton-n", "4", "--holdout", "1",
                             "--test-items", "0", "--longer-frac", "1.0"])
    finally:
        build_real.Redresser, build_real.DinoSim = old
    assert s["longer_tried"] == 4 and s["longer_kept"] == 4
    assert all(p == build_real.LONGER_PROMPT for p in prompts)
    tab = pq.read_table(next((tmp_path / "data" / "data").glob("*.parquet")))
    row = {c: tab.column(c)[0].as_py() for c in tab.column_names}
    assert row["key"].endswith("_long")
    person = np.asarray(Image.open(io.BytesIO(row["person"]["bytes"])).convert("RGB"), dtype=int)
    target = np.asarray(Image.open(io.BytesIO(row["target"]["bytes"])).convert("RGB"), dtype=int)
    h = person.shape[0]
    below = slice(int(h * 0.62), int(h * 0.72))   # just under the real hem: long garment in, real clothes out
    assert np.abs(person[below] - target[below]).mean() > 20
    assert np.abs(person[: int(h * 0.15)] - target[: int(h * 0.15)]).mean() < 8   # the head area is untouched

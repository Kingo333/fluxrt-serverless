"""fxreal3: same-clip pairs, light jitter, face weight, detail and pair loss terms."""

from __future__ import annotations

import ast
import io
import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
TRAINER = "train_dreambooth_lora_flux2_klein_img2img.py"
PATCHES = ("flux2-klein-garment.patch", "flux2-klein-timeshift.patch", "flux2-klein-maskloss.patch",
           "flux2-klein-fx3.patch")


def _patched_trainer(tmp_path) -> str:
    local = Path("/home/user/repos/diffusers")
    if not (local / "examples" / "dreambooth" / TRAINER).exists():
        pytest.skip("diffusers checkout not present")
    r = subprocess.run(["git", "-C", str(local), "show", f"HEAD:examples/dreambooth/{TRAINER}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("pinned diffusers file not available")
    repo = tmp_path / "d"
    (repo / "examples" / "dreambooth").mkdir(parents=True)
    (repo / "examples" / "dreambooth" / TRAINER).write_text(r.stdout)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name in PATCHES:
        r = subprocess.run(["git", "-C", str(repo), "apply", str(ROOT / "training" / "patches" / name)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{name}: {r.stderr}"
    return (repo / "examples" / "dreambooth" / TRAINER).read_text()


def test_fx3_patch_applies_after_the_others(tmp_path):
    src = _patched_trainer(tmp_path)
    compile(src, TRAINER, "exec")
    for s in ("--pair_column", "--pair_weight", "--detail_weight", "_azyah_pair_batches", "pair_cache.append",
              "noise[_j] = noise[_i]", "_unpatchify_latents(_x0_hat)", "**_az_logs"):
        assert s in src, s


def _function(src: str, name: str):
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"random": random}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "x", "exec"), ns)
    return ns[name]


def test_pairs_share_a_batch_and_nothing_is_lost(tmp_path):
    batches_of = _function(_patched_trainer(tmp_path), "_azyah_pair_batches")
    ids = ["", "c1", "", "c2", "c1", "", "c2", "", "c3", "", "lonely", ""]
    random.seed(0)
    batches = batches_of(list(range(len(ids))), ids, 4)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(ids)))  # every row once
    for pid in ("c1", "c2"):
        rows = [i for i, p in enumerate(ids) if p == pid]
        assert any(set(rows) <= set(b) for b in batches), pid
    assert all(len(b) <= 4 for b in batches)


def _loss_block(src: str) -> str:
    a = src.index("                # Azyah fxreal3: detail (edge) term")
    b = src.index("                accelerator.backward(loss)", a)
    return textwrap.dedent(src[a:b])


def test_detail_and_pair_terms_are_zero_for_a_perfect_prediction_and_train_otherwise(tmp_path):
    torch = pytest.importorskip("torch")
    block = _loss_block(_patched_trainer(tmp_path))

    class Pipe:
        @staticmethod
        def _unpatchify_latents(latents):
            b, c, h, w = latents.shape
            latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
            return latents.reshape(b, c // 4, h * 2, w * 2)

    class Args:
        detail_weight, pair_weight = 0.1, 1.0

    torch.manual_seed(0)
    x0 = torch.randn(4, 128, 6, 4)
    noise = torch.randn_like(x0)
    noise[1] = noise[0]
    sig = torch.tensor([0.3, 0.3, 0.9, 0.5]).reshape(4, 1, 1, 1)
    noisy = (1 - sig) * x0 + sig * noise
    m = torch.zeros(4, 1, 6, 4)
    m[:, :, 1:5, 1:3] = 1.0

    def run(pred):
        ns = {"torch": torch, "Flux2KleinPipeline": Pipe, "args": Args, "sigmas": sig, "noisy_model_input": noisy,
              "model_pred": pred, "model_input": x0, "m": m, "loss_mask": m, "loss": torch.zeros(()),
              "_az_pairs": [(0, 1)]}
        exec(block, ns)
        return ns["loss"], ns["_az_logs"]

    perfect = (noise - x0).clone()
    loss, logs = run(perfect)
    assert logs["detail"] == pytest.approx(0.0, abs=1e-5) and logs["pair"] == pytest.approx(0.0, abs=1e-5)

    wrong = (noise - x0 + 0.3 * torch.randn_like(x0)).requires_grad_(True)
    loss, logs = run(wrong)
    assert logs["detail"] > 0.01 and logs["pair"] > 0.001
    loss.backward()
    assert wrong.grad is not None and float(wrong.grad.abs().sum()) > 0
    assert float(wrong.grad[0][m[0].expand(128, -1, -1) == 0].abs().sum()) < float(wrong.grad[0].abs().sum())


def test_photo_jitter_is_identical_for_input_and_answer():
    from PIL import Image
    from fluxrt_lora import build_real
    rng = random.Random(1)
    assert build_real.jitter_factors(rng, 0.0) is None
    f = build_real.jitter_factors(random.Random(1), 1.0)
    assert 0.6 <= f[0] <= 1.4 and 0.75 <= f[1] <= 1.25
    img = Image.new("RGB", (8, 8), (120, 80, 60))
    a, b = build_real.photo_jitter(img, *f), build_real.photo_jitter(img.copy(), *f)
    assert list(a.getdata()) == list(b.getdata()) and list(a.getdata()) != list(img.getdata())


def test_vivid_pair_items_and_parquet_pair_column(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image
    from fluxrt_lora import build_real
    d = tmp_path / "pairs" / "vivid_train_000001"
    d.mkdir(parents=True)
    for f in ("a_target.png", "b_target.png", "a_mask.png", "b_mask.png"):
        Image.new("RGB", (8, 8)).save(d / f)
    Image.new("RGB", (8, 8)).save(d / "ref.jpg")
    (d / "meta.json").write_text(json.dumps({"category": "lower", "split": "train"}))
    (tmp_path / "pairs" / "incomplete").mkdir()
    items = list(build_real.vivid_pair_items(tmp_path / "pairs"))
    assert len(items) == 1 and items[0][0] == "vivid_train_000001" and items[0][1] == "lower"
    row = {"key": "k", "caption": "c", "person": b"p", "garment": b"g", "target": b"t", "mask": b"m"}
    build_real.write_parquet([dict(row, pair="clip1"), row], tmp_path / "x.parquet")
    t = pq.read_table(tmp_path / "x.parquet")
    assert t.column("pair").to_pylist() == ["clip1", ""]


def test_face_is_added_to_the_weight_mask():
    np = pytest.importorskip("numpy")
    from PIL import Image
    from data_tools import face_weight

    def seg(img):
        c = np.zeros((img.height, img.width), dtype=np.uint8)
        c[0:4, 0:4] = face_weight.FACE
        return c
    mask = Image.new("L", (16, 16), 0)
    mask.paste(255, (8, 8, 16, 16))
    buf_m, buf_t = io.BytesIO(), io.BytesIO()
    mask.save(buf_m, format="PNG")
    Image.new("RGB", (16, 16)).save(buf_t, format="JPEG")
    out, n = face_weight.merge_face(buf_m.getvalue(), buf_t.getvalue(), seg)
    m = np.asarray(Image.open(io.BytesIO(out)))
    assert n > 0 and m[1, 1] == 255 and m[12, 12] == 255 and m[8, 0] == 0


def test_pick_frames_stays_inside_the_clip():
    from data_tools.vivid_pairs import pick_frames
    rng = random.Random(0)
    for n in (12, 25, 49):
        t0, t1 = pick_frames(n, 3, rng)
        assert t1 - t0 == 3 and 0 <= t0 and t1 < n
    assert pick_frames(4, 3, rng) is None

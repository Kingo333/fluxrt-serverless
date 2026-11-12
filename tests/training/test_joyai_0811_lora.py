"""0811 LoRA (training/joyai_0811_lora): live schedule, config, static-clip data, trainer checks.
CPU only; the trainer is imported with stub torch/upstream modules."""
import importlib.util
import io
import json
import random
import sys
import tarfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2] / "training"
PKG = ROOT / "joyai_0811_lora"
UPSTREAM_STAGE1 = Path("/home/user/repos/upstream/JoyAI-Video-Edit/training/configs/stage1_ar.py")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ls = load(PKG / "live_schedule.py", "live_schedule")


# ---------- live schedule ----------

def test_live_timesteps_match_the_live_scheduler():
    """FlowMatchDiscreteScheduler(shift=5.159).set_timesteps(2): sigmas shift(linspace(1, 0, 3))."""
    sig = ls.live_sigmas()
    assert sig[0] == pytest.approx(1.0) and sig[-1] == pytest.approx(0.0)
    assert sig[1] == pytest.approx(5.159 * 0.5 / (1 + 4.159 * 0.5))
    ts = ls.live_timesteps()
    assert ts == pytest.approx([1000.0, 837.64], abs=0.01)
    assert ls.matches_live([1000.0, 837.636])
    assert not ls.matches_live([1000.0, 966.7, 900.0])  # 30-step Stage 1 default would fail
    assert not ls.matches_live([1000.0, 500.0])


def test_exit_policy():
    rng = random.Random(0)
    assert ls.choose_exit("last", 2, rng) == 1
    picks = {ls.choose_exit("random", 2, rng) for _ in range(50)}
    assert picks == {0, 1}
    assert ls.choose_exit("random", 1, rng) == 0
    with pytest.raises(ValueError):
        ls.choose_exit("first", 2, rng)


def test_static_clip_length_and_live_windows():
    assert ls.latent_frames(25) == 4 and ls.latent_frames(9) == 2
    with pytest.raises(ValueError):
        ls.latent_frames(24)
    # first chunk is the sink; window 3 = sink + the 2 most recent chunks
    assert ls.chunk_windows(4) == [[0], [0, 1], [0, 1, 2], [0, 2, 3]]


# ---------- config ----------

def _synthetic_upstream(mc, fashion) -> str:
    olds = [old for old, _ in fashion.REPLACEMENTS]
    produced = {new for _, new in fashion.REPLACEMENTS}
    olds += [old for old, _ in mc.EXTRA if not any(old in p for p in produced)]
    text = "\n".join(olds) + "\n"
    return text.replace("class Stage1ARConfig(ExpConfig):", "\n@dataclass\nclass Stage1ARConfig(ExpConfig):")


def test_config_applies_live_settings_on_top_of_the_fashion_config():
    mc = load(PKG / "make_config.py", "make_config_0811")
    fashion = mc._fashion_module()
    out = mc.build(_synthetic_upstream(mc, fashion), fashion)
    for _, new in mc.EXTRA:
        assert new in out
    assert "TRAIN_RESOLUTION = (480, 840)" in out and "(480, 832)" not in out
    assert "class AzyahJoyAI0811LoraConfig(ExpConfig):" in out and "AzyahFashionStage1Config" not in out
    assert "dmd_generator_num_inference_steps: int = 2" in out and "max_temporal_ids: int = 8" in out
    assert out.index(mc.ANCHOR) < out.index("dmd_use_lora: bool = True")
    assert out.index("def _manifest(group):") < out.index("class AzyahJoyAI0811LoraConfig")


@pytest.mark.skipif(not UPSTREAM_STAGE1.exists(), reason="upstream checkout not present")
def test_config_on_the_real_upstream_file_compiles(tmp_path):
    mc = load(PKG / "make_config.py", "make_config_0811b")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "stage1_ar.py").write_text(UPSTREAM_STAGE1.read_text())
    out = mc.main(str(tmp_path))
    compile(out.read_text(), str(out), "exec")


def test_config_fails_loudly_when_upstream_changes(tmp_path):
    mc = load(PKG / "make_config.py", "make_config_0811c")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "stage1_ar.py").write_text("TRAIN_RESOLUTION = (720, 1248)\n")
    with pytest.raises(SystemExit):
        mc.main(str(tmp_path))


# ---------- static-clip data ----------

def _png(color, size):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mp4(n=9, size=(64, 96)):
    import av
    from PIL import Image
    out = io.BytesIO()
    with av.open(out, "w", format="mp4") as c:
        st = c.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = size[0], size[1], "yuv420p"
        for k in range(n):
            for pkt in st.encode(av.VideoFrame.from_image(Image.new("RGB", size, (k * 20, 0, 0)))):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    return out.getvalue()


def _tar(path: Path, items: dict[str, dict[str, bytes]]):
    with tarfile.open(path, "w") as tar:
        for key, files in items.items():
            for name, data in files.items():
                info = tarfile.TarInfo(f"{key}.{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def _meta(source, split="train", consent=None):
    return json.dumps({"caption": "Substitute the current top with the garment in the reference image.",
                       "caption_detailed": "Substitute the current top ... The reference garment is the Tee: red.",
                       "metadata": {"ref_image": "0.png", "source": source, "split": split, "category": "upper",
                                    "consent_id": consent}}).encode()


def _manifest_dir(tmp_path):
    md = tmp_path / "manifests"
    md.mkdir()
    still = {"0.png": _png((200, 0, 0), (300, 400)), "1.png": _png((0, 0, 200), (600, 800)),
             "2.png": _png((200, 0, 0), (600, 800))}
    _tar(tmp_path / "i2i.tar", {
        "glance_a": {**still, "json": _meta("glance")},
        "glance_b": {**still, "1.png": _png((0, 200, 0), (300, 400)), "json": _meta("glance")},  # other size
        "glance_test": {**still, "json": _meta("glance", split="test")},
        "vivid_still": {**still, "json": _meta("vivid")},
    })
    _tar(tmp_path / "v2v.tar", {"vivid_clip": {"0.jpg": _png((1, 2, 3), (64, 96)), "0.mp4": _mp4(), "1.mp4": _mp4(),
                                               "json": _meta("vivid")}})
    (md / "i2i.json").write_text(json.dumps({"resolved_files": [str(tmp_path / "i2i.tar")], "num_samples": 4}))
    (md / "v2v.json").write_text(json.dumps({"resolved_files": [str(tmp_path / "v2v.tar")], "num_samples": 1}))
    return md


def _frames(data: bytes):
    import av
    with av.open(io.BytesIO(data)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def test_static_clips_are_steady_live_sized_and_licence_filtered(tmp_path):
    pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    mod = load(PKG / "make_static_clips.py", "make_static_clips")
    out = tmp_path / "out"
    s = mod.build(str(_manifest_dir(tmp_path)), str(out), frames=9, mode="commercial", detail_oversample=1)
    assert s["static_clips"] == 2 and s["vivid_videos"] == 0 and s["dropped_by_mode"] == 1
    assert s["commercial_use_cleared"] is True and s["trained_with_vivid"] is False
    manifest = json.loads((out / "manifest" / "v2v.json").read_text())
    assert manifest["num_samples"] == 2 and not (out / "manifest" / "i2i.json").exists()
    files = {}
    for shard in manifest["resolved_files"]:
        with tarfile.open(shard) as tar:
            for m in tar:
                files[m.name] = tar.extractfile(m).read()
    assert {"glance_a_s9.0.jpg", "glance_a_s9.0.mp4", "glance_a_s9.1.mp4", "glance_a_s9.json"} <= set(files)
    rec = json.loads(files["glance_a_s9.json"])
    assert rec["caption"].startswith("Substitute") and "Tee" in rec["caption_detailed"]
    assert rec["metadata"]["ref_image"] == "0.jpg" and rec["metadata"]["static_clip"] is True
    for name in ("glance_a_s9.0.mp4", "glance_a_s9.1.mp4", "glance_b_s9.0.mp4"):
        fr = _frames(files[name])
        assert len(fr) == 9 and fr[0].shape == (840, 480, 3)  # live 480x840 portrait
        assert all(np.abs(f.astype(int) - fr[0].astype(int)).mean() < 2 for f in fr)  # steady
    tgt = _frames(files["glance_a_s9.1.mp4"])[0]
    src = _frames(files["glance_a_s9.0.mp4"])[0]
    assert tgt[..., 0].mean() > 150 and src[..., 2].mean() > 150  # target red, person source blue


def test_vivid_videos_only_on_explicit_opt_in(tmp_path):
    pytest.importorskip("av")
    mod = load(PKG / "make_static_clips.py", "make_static_clips2")
    s = mod.build(str(_manifest_dir(tmp_path)), str(tmp_path / "o"), frames=9, mode="research", vivid_videos=True,
                  detail_oversample=1)
    assert s["vivid_videos"] == 1 and s["static_clips"] == 3
    assert s["trained_with_vivid"] is True and s["commercial_use_cleared"] is False
    with pytest.raises(SystemExit):
        mod.build(str(tmp_path / "manifests"), str(tmp_path / "o2"), frames=24)


def test_cover_box_keeps_aspect_and_centres():
    mod = load(PKG / "make_static_clips.py", "make_static_clips3")
    scale, box = mod.cover_box(600, 800, 480, 840)
    assert scale == pytest.approx(1.05) and box == (75, 0, 555, 840)


# ---------- trainer (stub torch / upstream) ----------

@pytest.fixture
def trainer(monkeypatch):
    names = ["torch", "torch.distributed", "torch.nn", "torch.nn.functional", "torch.utils", "torch.utils.tensorboard",
             "src", "src.train_t2v_dmd", "src.config", "src.dataset", "src.dataset.webdataset", "src.distributed",
             "src.distributed.communications", "src.distributed.parallel_states", "src.models", "src.train_t2v",
             "src.utils", "src.utils.checkpoint", "src.utils.logging"]
    attrs = {"torch.utils.tensorboard": ["SummaryWriter"], "src.config": ["ExpConfig", "load_config_class_from_pyfile"],
             "src.dataset.webdataset": ["build_webdataset_dataloader"], "src.distributed.communications": ["broadcast_item"],
             "src.distributed.parallel_states": ["clean_dist_env"], "src.models": ["load_pipeline"],
             "src.train_t2v": ["prepare_inputs", "setup_distributed_training", "setup_model_and_optimizer"],
             "src.utils": ["find_latest_checkpoint", "seed_everything"],
             "src.utils.checkpoint": ["load_checkpoint", "save_checkpoint"], "src.utils.logging": ["setup_logger"]}
    for n in names:
        m = types.ModuleType(n)
        m.__path__ = []
        for a in attrs.get(n, []):
            setattr(m, a, object)
        monkeypatch.setitem(sys.modules, n, m)
    sys.modules["torch"].distributed = sys.modules["torch.distributed"]
    sys.modules["src"].train_t2v_dmd = sys.modules["src.train_t2v_dmd"]
    sys.modules["src.train_t2v_dmd"]._sample_exit_index = lambda n: 0
    monkeypatch.setitem(sys.modules, "live_schedule", ls)
    monkeypatch.setitem(sys.modules, "losses", load(PKG / "losses.py", "losses"))
    monkeypatch.setenv("AZ_TOOLS_DIR", str(PKG))
    return load(PKG / "train.py", "azyah_train_0811_lora")


class _Sched:
    def __init__(self, n_to_ts):
        self.n_to_ts = n_to_ts

    def set_timesteps(self, n, device=None):
        self.timesteps = SimpleNamespace(tolist=lambda: self.n_to_ts(n))


def _live_cfg(**kw):
    base = dict(dmd_generator_num_inference_steps=2, max_temporal_ids=8, ref_image_basesize=768,
                ref_image_bucket_configs=None, cfg_rate=0.0, use_vit=True, stage1_lora=True, sft_exit="random")
    return SimpleNamespace(**{**base, **kw})


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(m)


def test_trainer_self_check_accepts_live_and_rejects_drift(trainer):
    pipe = SimpleNamespace(scheduler=_Sched(lambda n: ls.live_timesteps(n)))
    model = SimpleNamespace(config=SimpleNamespace(chunk_size=1, causal=True, local_window_size=3, global_sink_chunk=True),
                            _azyah_stage1_lora=True)
    log = _Log()
    trainer.self_check(_live_cfg(), pipe, model, log)
    assert "self-check ok" in log.lines[-1]
    for bad in (dict(dmd_generator_num_inference_steps=30), dict(ref_image_basesize=480), dict(cfg_rate=0.01),
                dict(max_temporal_ids=16), dict(stage1_lora=False)):
        with pytest.raises(SystemExit):
            trainer.self_check(_live_cfg(**bad), pipe, model, log)
    with pytest.raises(SystemExit):  # a bidirectional (non-causal) checkpoint is not the live model
        trainer.self_check(_live_cfg(), pipe, SimpleNamespace(config=SimpleNamespace(chunk_size=1, causal=False),
                                                              _azyah_stage1_lora=True), log)


def test_trainer_exit_policy_drives_the_upstream_rollout(trainer):
    rng = random.Random(1)
    trainer.install_exit_policy(_live_cfg(sft_exit="last"), rng)
    assert trainer.dmd._sample_exit_index(2) == 1
    trainer.install_exit_policy(_live_cfg(sft_exit="random"), rng)
    assert {trainer.dmd._sample_exit_index(2) for _ in range(40)} == {0, 1}
    with pytest.raises(ValueError):
        trainer.install_exit_policy(_live_cfg(sft_exit="sometimes"), rng)


def test_trainer_uses_the_patched_lora_save_and_keeps_every_save():
    src = (PKG / "train.py").read_text()
    assert "from src.utils.checkpoint import load_checkpoint, save_checkpoint" in src  # patched: refuses empty saves
    assert "setup_model_and_optimizer(cfg, device)" in src  # patched: frozen base + trainable-count assert
    assert "generate_dmd_latents(" in src and "requires_grad=True" in src and "rollout_num_chunks_override=-1" in src
    assert "garment_loss(pred.float(), target.float(), mask, MASK_WEIGHT, EDGE_WEIGHT)" in src
    assert "prune_checkpoints" not in src
    patch = (ROOT / "patches" / "stage1-lora.patch").read_text()
    assert "refusing to write a \"" in patch and "checkpoint without the adapter weights" in patch
    assert "has NO adapter weights (empty generator_lora)" in patch


def test_run_script_guards_the_live_model_and_pins_settings():
    sh = (PKG / "run_train.sh").read_text()
    assert "/runpod-volume/*) echo \"REFUSING" in sh and "is not on the training volume" in sh
    for knob in ('AZ_MAX_STEPS="${AZ_MAX_STEPS:-1000}"', 'AZ_LR="${AZ_LR:-3e-5}"', 'AZ_LORA_RANK="${AZ_LORA_RANK:-32}"',
                 'AZ_CKPT_INTERVAL="${AZ_CKPT_INTERVAL:-250}"', 'AZ_STATIC_FRAMES="${AZ_STATIC_FRAMES:-25}"',
                 'VIVID_VIDEOS="${AZ_VIVID_VIDEOS:-$WITH_VIVID_DRESSES}"'):
        assert knob in sh
    assert "prune_checkpoints" not in sh and "GOOD_SAVES" in sh
    ev = (PKG / "eval_saves.sh").read_text()
    assert "--live --cross" in ev and '--base "$BASE"' in ev and "--require-all" in ev


def test_lora_save_merges_into_an_0811_style_base():
    np = pytest.importorskip("numpy")
    mm = load(ROOT / "scripts" / "merge_lora.py", "merge_lora_0811")
    rng = np.random.default_rng(3)
    w = rng.standard_normal((8, 6)).astype("float32")
    a = rng.standard_normal((4, 6)).astype("float32")
    b = rng.standard_normal((8, 4)).astype("float32")
    base = mm.unwrap_backbone({"model": {"double_blocks.3.img_mlp.fc1.weight": w}})
    save = {"step": 250, "stage1_lora": True, "generator_lora": {"double_blocks.3.img_mlp.fc1.lora_A.weight": a,
                                                                 "double_blocks.3.img_mlp.fc1.lora_B.weight": b},
            "generator_lora_config": {"r": 4, "lora_alpha": 4}}
    lora = mm.find_lora(save)
    merged, n = mm.merge(base, lora, 4, 4.0)
    assert n == 1 and np.allclose(merged["double_blocks.3.img_mlp.fc1.weight"], w + b @ a)


def test_generate_vivid_can_fuse_a_save_and_sends_the_full_instruction():
    src = (ROOT / "eval" / "generate_vivid.py").read_text()
    assert 'cfg.dmd_base_ckpt = a.base' in src and "cfg.stage1_lora = False" in src
    assert 'gmeta.get("garment_name"), gmeta.get("garment_description")) if a.live' in src


def test_real_frame_sources_are_taken_first(tmp_path):
    """Owner 2026-11-05: ViViD real frames first, Glance fills the rest of --max."""
    pytest.importorskip("av")
    mod = load(PKG / "make_static_clips.py", "make_static_clips3")
    s = mod.build(str(_manifest_dir(tmp_path)), str(tmp_path / "o"), frames=9, max_items=1, mode="research",
                  detail_oversample=1)
    assert s["by_source"] == {"vivid": 1} and s["trained_with_vivid"] is True
    s2 = mod.build(str(_manifest_dir(tmp_path / "x") if (tmp_path / "x").mkdir() is None else None), str(tmp_path / "o2"),
                   frames=9, max_items=3, mode="research", detail_oversample=1, prefer_sources=())
    assert sum(s2["by_source"].values()) == 3


def test_motion_clips_are_repeated(tmp_path):
    pytest.importorskip("av")
    mod = load(PKG / "make_static_clips.py", "make_static_clips4")
    s = mod.build(str(_manifest_dir(tmp_path)), str(tmp_path / "o"), frames=9, mode="research", vivid_videos=True,
                  detail_oversample=1, clip_repeat=3)
    assert s["vivid_videos"] == 3

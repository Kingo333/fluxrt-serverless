"""Compile phase (owner, 2026-11-14): fewer CPU-GPU waits (FLUXRT_FAST_MASK), compile settings, compile practice
frames at warm-up, and sizes that are rounded safely instead of stopping the worker. Everything is set in env."""
import importlib
import importlib.util
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = ("FLUXRT_FAST_MASK", "FLUXRT_COMPILE", "FLUXRT_COMPILE_CACHE_DIR", "FLUXRT_COMPILE_WARMUP",
       "FLUXRT_COMPILE_WARMUP_PASSES", "FLUXRT_COMPILE_RECOMPILE_LIMIT", "TORCHDYNAMO_SUPPRESS_ERRORS",
       "FLUXRT_WIDTH", "FLUXRT_HEIGHT", "FLUXRT_REF_WIDTH", "FLUXRT_REF_HEIGHT", "FLUXRT_CACHE_MASK",
       "FLUXRT_PRECISION", "FLUXRT_LORA_PATH", "FLUXRT_ROOT", "ENABLE_REFERENCE_IMAGE", "FLUXRT_SHARPEN")


def load(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("SESSION_SIGNING_SECRET", "s")
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FLUXRT_ROOT", str(tmp_path / "fluxrt"))
    (tmp_path / "fluxrt").mkdir(exist_ok=True)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("server", None)
    return importlib.import_module("server")


def config(s):
    return json.loads(s.write_runtime_config().read_text())


# ---- the FluxRT patch ----------------------------------------------------------------------------------------

def test_speed_patch_touches_the_right_places():
    patch = (ROOT / "fluxrt/patches/fluxrt-speed.patch").read_text()
    # the index of changed tokens is worked out once per model call, the original path stays when off
    assert "AZYAH_FAST_MASK = False" in patch and "def _azyah_active_index(seq_mask" in patch
    assert "@torch.compiler.disable" in patch and "_azyah_reset_active_index()" in patch
    assert patch.count("if AZYAH_FAST_MASK:") == 3                      # MLP, attention, reset in forward
    assert "+    if seq_mask.any():" not in patch                         # FluxRT's own lines are kept, not replaced
    # the leftover timing helper stops waiting with the fast mask
    assert "AZYAH_PROFILE_SYNC = True" in patch and "if not AZYAH_PROFILE_SYNC:" in patch
    assert 'self.config.get("fast_active_index", False)' in patch
    # compile: errors fall back to normal running, a higher recompile limit, both from the worker's config
    assert 'self.config.get("compile_suppress_errors", True)' in patch
    assert 'self.config.get("compile_recompile_limit", 64)' in patch and '"recompile_limit", "cache_size_limit"' in patch


@pytest.mark.skipif(not os.environ.get("FLUXRT_SRC"), reason="set FLUXRT_SRC to a FluxRT checkout with our patches applied")
def test_fast_mask_gives_the_same_pictures_in_fluxrt():
    """FluxRT's own transformer (tiny config, CPU): identical outputs with the fast mask on, far fewer waits."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    path = Path(os.environ["FLUXRT_SRC"]) / "src/fluxrt/stream_processor/transformer_flux2.py"
    spec = importlib.util.spec_from_file_location("azyah_tf_check", path)
    tf = importlib.util.module_from_spec(spec)
    sys.modules["azyah_tf_check"] = tf
    spec.loader.exec_module(tf)
    torch.manual_seed(0)
    model = tf.Flux2Transformer2DModel(
        patch_size=1, in_channels=8, out_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=16, timestep_guidance_channels=16, mlp_ratio=2.0,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
    T, N = 5, 23
    x, c = torch.randn(1, N, 8), torch.randn(1, T, 16)
    ids = torch.zeros(1, N, 4)
    ids[..., 1] = torch.arange(N)
    g = torch.Generator().manual_seed(1)
    masks = [torch.full((1, T + N), 2, dtype=torch.int32)]
    for p in (0.3, 0.7, 0.0, 0.5):
        m = (torch.rand(1, T + N, generator=g) < p).int() * 2
        m[0, :T] = 0
        masks.append(m)
    masks.append(torch.randint(0, 3, (1, T + N), generator=g, dtype=torch.int32))

    def run(fast):
        tf.AZYAH_FAST_MASK = fast
        cache = tf.SpatialCache(image_seq_len=N, text_seq_len=T, output_channels=8, attention_head_dim=8,
                                num_attention_heads=2, num_layers=1, num_single_layers=1, device="cpu",
                                dtype=torch.float32)
        calls = [0]
        real = torch.Tensor.nonzero

        def counting(self, *a, **k):
            calls[0] += 1
            return real(self, *a, **k)
        torch.Tensor.nonzero = counting
        try:
            with torch.no_grad():
                outs = [model(hidden_states=x + 0.1 * i, encoder_hidden_states=c, timestep=torch.tensor([0.5]),
                              img_ids=ids, txt_ids=torch.zeros(1, T, 4), return_dict=False, mask=m.clone(),
                              spatial_cache=cache)[0].clone() for i, m in enumerate(masks)]
        finally:
            torch.Tensor.nonzero = real
        return outs, calls[0]

    slow, n_slow = run(False)
    fast, n_fast = run(True)
    assert all(torch.equal(a, b) for a, b in zip(slow, fast))
    assert n_fast <= 3 * len(masks) < n_slow                        # at most one wait per mask kind per call


# ---- worker settings -----------------------------------------------------------------------------------------

def test_new_switches_defaults_and_env(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path)
    cfg = config(s)
    assert s.FAST_MASK is True and cfg["fast_active_index"] is True     # same pictures, so on by default
    assert cfg["compile_models"] is False and cfg["compile_suppress_errors"] is True
    assert cfg["compile_recompile_limit"] == 64
    assert (s.COMPILE_WARMUP, s.COMPILE_WARMUP_PASSES) == (True, 4)
    sw = s.live_switches(False)
    assert sw["fast_mask"] is True and sw["compile_warmup_passes"] == 0 and sw["compile_cache"] == "off"
    s = load(monkeypatch, tmp_path, FLUXRT_FAST_MASK="0", FLUXRT_COMPILE="1", FLUXRT_COMPILE_WARMUP_PASSES="7",
             TORCHDYNAMO_SUPPRESS_ERRORS="0", FLUXRT_COMPILE_RECOMPILE_LIMIT="128")
    cfg = config(s)
    assert cfg["fast_active_index"] is False and cfg["compile_models"] is True
    assert cfg["compile_suppress_errors"] is False and cfg["compile_recompile_limit"] == 128
    sw = s.live_switches(False)
    assert (sw["fast_mask"], sw["compile_warmup_passes"], sw["compile_suppress_errors"],
            sw["compile_recompile_limit"]) == (False, 7, False, 128)
    s = load(monkeypatch, tmp_path, FLUXRT_COMPILE="1", FLUXRT_COMPILE_WARMUP="off", FLUXRT_COMPILE_WARMUP_PASSES="99",
             FLUXRT_COMPILE_RECOMPILE_LIMIT="3", TORCHDYNAMO_SUPPRESS_ERRORS="1")
    assert s.COMPILE_WARMUP is False and s.COMPILE_WARMUP_PASSES == 12 and s.COMPILE_RECOMPILE_LIMIT == 8
    assert s.live_switches(False)["compile_warmup_passes"] == 0 and s.COMPILE_SUPPRESS_ERRORS is True
    assert load(monkeypatch, tmp_path, FLUXRT_COMPILE_WARMUP_PASSES="lots").COMPILE_WARMUP_PASSES == 4   # typo: default


def test_sizes_are_rounded_safely_and_logged(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path, FLUXRT_WIDTH="304", FLUXRT_HEIGHT="544")
    assert (s.WIDTH, s.HEIGHT) == (304, 544) and s.SETTING_NOTES == []    # the owner's ~10% step, as asked
    cfg = config(s)
    assert cfg["resolution"] == {"width": 304, "height": 544}
    s = load(monkeypatch, tmp_path, FLUXRT_WIDTH="300", FLUXRT_HEIGHT="2000", FLUXRT_REF_WIDTH="530",
             FLUXRT_REF_HEIGHT="seven hundred")
    assert (s.WIDTH, s.HEIGHT, s.REF_WIDTH, s.REF_HEIGHT) == (304, 1024, 528, 512)
    assert s.SETTING_NOTES == ["FLUXRT_WIDTH=300 -> 304 (multiples of 16, 128..1024)",
                               "FLUXRT_HEIGHT=2000 -> 1024 (multiples of 16, 128..1024)",
                               "FLUXRT_REF_WIDTH=530 -> 528 (multiples of 16, 128..1024)",
                               "FLUXRT_REF_HEIGHT='seven hundred' is not a number; using 512"]
    sw = s.live_switches(False)
    assert sw["size"] == "304x1024" and sw["ref_size"] == "528x512" and len(sw["setting_notes"]) == 4
    assert load(monkeypatch, tmp_path, FLUXRT_WIDTH="40").WIDTH == 128                  # never below 128
    assert load(monkeypatch, tmp_path, FLUXRT_WIDTH="296").WIDTH == 304                  # halfway rounds up
    assert load(monkeypatch, tmp_path, FLUXRT_WIDTH="288.0").SETTING_NOTES == []         # same value, no note


# ---- compile practice at warm-up -----------------------------------------------------------------------------

class _PracticeProc:
    """FluxRT at warm-up: a background 'model loop' finishes a frame every few ms and records what it was given."""

    def __init__(self, h=64, w=48, frame_s=0.004):
        self.h, self.w = h, w
        self.masks, self.refs, self.frames = [], [], []
        self.t = 0.0
        self.input = types.SimpleNamespace(copy_from=lambda f: self.frames.append(f.copy()))
        self._stop = False
        self._thread = threading.Thread(target=self._loop, args=(frame_s,), daemon=True)
        self._thread.start()

    def _loop(self, frame_s):
        while not self._stop:
            time.sleep(frame_s)
            self.t += 0.001                                         # a new "last processing time" per frame

    def stop(self):
        self._stop = True

    def get_resolution(self): return {"width": self.w, "height": self.h}
    def get_input_tensor(self): return self.input
    def get_last_processing_time(self): return self.t
    def set_mask(self, m): self.masks.append(np.array(m))
    def set_reference_image(self, r): self.refs.append(None if r is None else r.shape)


def test_compile_practice_with_the_seg_mask(monkeypatch, tmp_path, caplog):
    s = load(monkeypatch, tmp_path, FLUXRT_COMPILE="1", FLUXRT_CACHE_MASK="seg", FLUXRT_REF_WIDTH="528",
             FLUXRT_REF_HEIGHT="704")
    proc = _PracticeProc(h=320, w=160)                                  # 20 x 10 cells of 16 pixels
    try:
        with caplog.at_level("INFO", logger="fluxrt-lb"):
            s.compile_warmup(proc, 4, 60.0)
    finally:
        proc.stop()
    shares = [round(float((m == 2).mean()), 2) for m in proc.masks]
    assert shares == [0.6, 0.15, 0.0, 0.85, 1.0]                        # 4 practice passes, then a full mask
    assert all(m.shape == (320 // 16, 160 // 16) and m.dtype == np.int32 for m in proc.masks)
    assert proc.refs == [(704, 528, 3), None]                           # stand-in garment, then cleared
    assert len(proc.frames) == 4 and all(f.shape == (320, 160, 3) and f.dtype == np.uint8 for f in proc.frames)
    assert "warmup: compile practice 1/4 (60% changing): 3 frames in" in caplog.text
    assert "warmup: compile practice done: 4 passes in" in caplog.text


def test_compile_practice_without_the_seg_mask_changes_part_of_the_frame(monkeypatch, tmp_path):
    s = load(monkeypatch, tmp_path, FLUXRT_COMPILE="1", ENABLE_REFERENCE_IMAGE="false")
    proc = _PracticeProc(h=40, w=32)
    try:
        s.compile_warmup(proc, 3, 60.0)
    finally:
        proc.stop()
    assert proc.masks == [] and proc.refs == []                         # FluxRT's own motion check, no garment slot
    first = proc.frames[0]
    changed = [float((f != first).any(axis=(1, 2)).mean()) for f in proc.frames[1:]]
    # every frame shares the same base below the changed rows: 15% then 0% of the rows differ from the base
    base_rows = (proc.frames[2] == proc.frames[1]).all(axis=(1, 2))
    assert proc.frames[2].shape == (40, 32, 3) and base_rows[int(round(0.15 * 40)):].all()
    assert changed[1] > 0                                               # pass 1 (60%) differs from pass 3 (0%)


def test_compile_practice_never_blocks_the_warm_up(monkeypatch, tmp_path, caplog):
    s = load(monkeypatch, tmp_path, FLUXRT_COMPILE="1")
    stuck = _PracticeProc()
    stuck.stop()                                                       # no frames ever finish
    time.sleep(0.02)
    with caplog.at_level("INFO", logger="fluxrt-lb"):
        t0 = time.time()
        s.compile_warmup(stuck, 4, 6.0)
    assert time.time() - t0 < 8.0                                       # bounded by the warm-up time left
    assert "only 0 frames" in caplog.text and "compile practice" in caplog.text


class _FakeStreamProcessor(_PracticeProc):
    def __init__(self, config_path):
        super().__init__()
        self.cfg = json.loads(Path(config_path).read_text())
        self.model_inference_subprocess = types.SimpleNamespace(process=None)
        self.output_scheduler_subprocess = types.SimpleNamespace(process=None)

    def start(self):
        Path(self.cfg["precision_marker"]).write_text(json.dumps({"precision": "fp8", "fp8_layers": 100}))

    def is_ready(self):
        return True


@pytest.mark.parametrize("env, practiced", [({"FLUXRT_COMPILE": "1"}, True),
                                            ({"FLUXRT_COMPILE": "1", "FLUXRT_COMPILE_WARMUP": "0"}, False),
                                            ({}, False)])
def test_warm_up_runs_the_practice_only_with_compile(monkeypatch, tmp_path, caplog, env, practiced):
    s = load(monkeypatch, tmp_path, FLUXRT_PRECISION="fp8", FLUXRT_COMPILE_WARMUP_PASSES="2", **env)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "fluxrt", types.SimpleNamespace(StreamProcessor=_FakeStreamProcessor))
    monkeypatch.setitem(sys.modules, "fluxrt.utils", types.SimpleNamespace(crop_maximal_rectangle=lambda *a: None))
    monkeypatch.setattr(s, "ensure_models", lambda: None)
    with caplog.at_level("INFO", logger="fluxrt-lb"):
        s.warmup_blocking()
    s.STATE["processor"].stop()
    assert s.STATE["status"] == "ready" and s.STATE["precision_active"] == "fp8 (100 layers)"
    assert ("warmup: compile practice done: 2 passes" in caplog.text) is practiced
    line = [l for l in caplog.text.splitlines() if "warmup: ready; live switches" in l][0]
    for key in ("'fast_mask': True", "'size': '576x320'", "'sharpen': 0.0", "'precision': 'fp8'",
                "'compile_warmup_passes': %d" % (2 if practiced else 0), "'setting_notes': []"):
        assert key in line, key


def test_a_failing_practice_still_ends_ready(monkeypatch, tmp_path, caplog):
    s = load(monkeypatch, tmp_path, FLUXRT_COMPILE="1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "fluxrt", types.SimpleNamespace(StreamProcessor=_FakeStreamProcessor))
    monkeypatch.setitem(sys.modules, "fluxrt.utils", types.SimpleNamespace(crop_maximal_rectangle=lambda *a: None))
    monkeypatch.setattr(s, "ensure_models", lambda: None)

    def boom(*a):
        raise RuntimeError("practice broke")
    monkeypatch.setattr(s, "compile_warmup", boom)
    with caplog.at_level("INFO", logger="fluxrt-lb"):
        s.warmup_blocking()
    s.STATE["processor"].stop()
    assert s.STATE["status"] == "ready"
    assert "compile practice failed (practice broke); continuing without it" in caplog.text

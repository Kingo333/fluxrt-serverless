"""FluxRT live-quality switches: off = today's behaviour; on = the documented effects."""
import base64
import importlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from session_token import sign

SECRET = "s"
SWITCHES = ("FLUXRT_OUTPUT_PACING", "FLUXRT_OUTPUT_POLL_HZ", "FLUXRT_COMPOSITE", "FLUXRT_REF_CLEAN",
            "FLUXRT_CACHE_MASK", "FLUXRT_ANCHOR", "FLUXRT_MU", "FLUXRT_ANCHOR_REFRESH_S", "FLUXRT_FACE_PASTE",
            "FLUXRT_FACE_PASTE_FEATHER", "FLUXRT_FACE_PASTE_SHRINK", "FLUXRT_FACE_PASTE_MIN_OVERLAP", "FLUXRT_INPUT_SCALE", "FLUXRT_SEG_MODEL",
            "ENABLE_PLAYBACK_SCHEDULER", "PLAYBACK_TARGET_FPS", "REPEAT_LAST_FRAME_WHEN_IDLE",
            "FLUXRT_WIDTH", "FLUXRT_HEIGHT", "FLUXRT_CLIENT_HINTS", "FLUXRT_LORA_PATH", "FLUXRT_REF_CLEAN_DRESS",
            "LIVECAM_WEBRTC", "LIVECAM_WEBRTC_MAX_KBPS", "LIVECAM_WEBRTC_START_KBPS")


def load(monkeypatch, **env):
    monkeypatch.setenv("SESSION_SIGNING_SECRET", SECRET)
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("server", None)
    return importlib.import_module("server")


def token():
    return sign(f"u1.g1.standard.300.{int(time.time()) + 300}", SECRET)


def test_defaults_are_all_off(monkeypatch):
    s = load(monkeypatch)
    assert not s.OUTPUT_PACING and not s.COMPOSITE and not s.REF_CLEAN and not s.ANCHOR
    assert s.CACHE_MASK == "auto" and s.MU_OVERRIDE is None and s.INPUT_SCALE == 1 and not s.SEG_NEEDED
    sw = s.live_switches(False)
    assert sw == {"output_pacing": False, "output_poll_hz": None, "composite": False, "ref_clean": False,
                  "ref_clean_dress": False, "cache_mask": "auto", "anchor": False, "anchor_refresh_s": None, "face_paste": False, "mu": None,
                  "input_scale": 1, "segmenter_loaded": False, "webrtc": False, "webrtc_max_kbps": None}


def test_ready_message_unchanged_when_off_and_extended_when_on(monkeypatch):
    s = load(monkeypatch, FLUXRT_WIDTH="288", FLUXRT_HEIGHT="512")
    m = s.ready_message(10.0)
    assert set(m) == {"type", "session_deadline", "engine", "input_width", "input_height", "output_width",
                      "output_height", "input_fps", "input_jpeg_quality"}
    assert (m["input_width"], m["input_height"]) == (288, 512)
    s = load(monkeypatch, FLUXRT_WIDTH="288", FLUXRT_HEIGHT="512", FLUXRT_OUTPUT_PACING="1",
             FLUXRT_OUTPUT_POLL_HZ="24", FLUXRT_INPUT_SCALE="2")
    m = s.ready_message(10.0)
    assert m["playback_buffer"] is True and m["playback_fps"] == 24
    assert (m["input_width"], m["input_height"]) == (576, 1024)      # sharp input for the composite
    assert (m["output_width"], m["output_height"]) == (576, 1024)    # model size x upscaler, unchanged


def test_runtime_config_keys_only_when_switched_on(monkeypatch, tmp_path):
    s = load(monkeypatch, FLUXRT_WIDTH="288", FLUXRT_HEIGHT="512")
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    cfg = json.loads(s.write_runtime_config().read_text())
    assert "mask_calculation_method" not in cfg and "use_anchor_image" not in cfg and "mu_override" not in cfg
    s = load(monkeypatch, FLUXRT_WIDTH="288", FLUXRT_HEIGHT="512", FLUXRT_CACHE_MASK="seg", FLUXRT_ANCHOR="1",
             FLUXRT_MU="1.9")
    monkeypatch.setattr(s, "FLUXRT_ROOT", tmp_path)
    cfg = json.loads(s.write_runtime_config().read_text())
    assert cfg["mask_calculation_method"] == "manual" and cfg["use_anchor_image"] is True and cfg["mu_override"] == 1.9


def test_bad_cache_mask_value_falls_back_to_auto(monkeypatch):
    s = load(monkeypatch, FLUXRT_CACHE_MASK="weird")
    assert s.CACHE_MASK == "auto"


def test_composite_lag_env_parsing(monkeypatch):
    monkeypatch.delenv("FLUXRT_COMPOSITE_LAG_MS", raising=False)
    assert load(monkeypatch).COMPOSITE_LAG_S is None
    assert load(monkeypatch, FLUXRT_COMPOSITE_LAG_MS="250").COMPOSITE_LAG_S == 0.25
    assert load(monkeypatch, FLUXRT_COMPOSITE_LAG_MS="abc").COMPOSITE_LAG_S is None   # never crashes a session
    assert load(monkeypatch, FLUXRT_MU="abc").MU_OVERRIDE is None                       # typo never stops the worker


def test_numeric_switch_typos_use_defaults(monkeypatch):
    s = load(monkeypatch, FLUXRT_OUTPUT_POLL_HZ="fast", FLUXRT_COMPOSITE_FEATHER="x", FLUXRT_COMPOSITE_OUT_SEG_EVERY="?",
             FLUXRT_CACHE_MASK_DILATE="two", FLUXRT_CACHE_MASK_REFRESH="", FLUXRT_ANCHOR_DELAY_S="soon",
             FLUXRT_INPUT_SCALE="2x")
    assert (s.OUTPUT_POLL_HZ, s.COMPOSITE_FEATHER, s.COMPOSITE_OUT_SEG_EVERY) == (30.0, 3, 1)
    assert (s.CACHE_MASK_DILATE, s.CACHE_MASK_REFRESH, s.ANCHOR_DELAY_S, s.INPUT_SCALE) == (2, 48, 1.5, 1)
    assert load(monkeypatch, FLUXRT_INPUT_SCALE="2", FLUXRT_OUTPUT_POLL_HZ="24").INPUT_SCALE == 2
    assert load(monkeypatch, FLUXRT_OUTPUT_POLL_HZ="24").OUTPUT_POLL_HZ == 24.0


def png_b64(w, h, rgb):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_reference_clean_needs_switch_and_segmenter(monkeypatch):
    from livecam import segment as sg

    def fake_seg(rgb):
        cat = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cat[: rgb.shape[0] // 4, :] = sg.FACE                       # a model's face on top
        cat[rgb.shape[0] // 4:, :] = sg.CLOTHES                      # the garment below
        return cat

    s = load(monkeypatch)                                            # switch off: segmenter ignored
    arr = s.b64_to_reference_rgb(png_b64(384, 512, (200, 0, 0)), fake_seg)
    assert arr[20, 192][0] > 190                                     # face region still red
    s = load(monkeypatch, FLUXRT_REF_CLEAN="1")
    arr = s.b64_to_reference_rgb(png_b64(384, 512, (200, 0, 0)), fake_seg)
    assert (arr[20, 192] == 255).all() and arr[400, 192][0] > 190    # face cut to white, garment kept
    arr = s.b64_to_reference_rgb(png_b64(384, 512, (200, 0, 0)), None)
    assert arr[20, 192][0] > 190                                     # no segmenter: untouched


def _abaya_seg(rgb):
    """An on-model abaya photo: hijab top above the chin, face, a necklace, the abaya below."""
    from livecam import segment as sg
    cat = np.zeros(rgb.shape[:2], dtype=np.uint8)
    h = rgb.shape[0]
    cat[: h // 8, :] = sg.CLOTHES                                   # hijab over the head
    cat[h // 8: h // 4, :] = sg.FACE
    cat[h // 4:, :] = sg.CLOTHES                                    # the abaya
    cat[h // 4 + 10: h // 4 + 30, :] = sg.OTHERS                    # a necklace on the chest
    return cat


def test_dress_reference_drops_hijab_top_and_jewellery(monkeypatch):
    s = load(monkeypatch, FLUXRT_REF_CLEAN="1")
    assert s.REF_CLEAN_DRESS and s.live_switches(True)["ref_clean_dress"] is True
    b64 = png_b64(384, 512, (90, 40, 20))
    general = s.b64_to_reference_rgb(b64, _abaya_seg)
    dress = s.b64_to_reference_rgb(b64, _abaya_seg, dress=True)
    assert general[30, 192][0] < 200 and general[150, 192][0] < 200    # general rule keeps hijab and necklace
    assert (dress[30, 192] == 255).all() and (dress[150, 192] == 255).all()
    assert dress[400, 192][0] < 200                                     # the abaya itself is kept
    s = load(monkeypatch, FLUXRT_REF_CLEAN="1", FLUXRT_REF_CLEAN_DRESS="0")
    assert not s.REF_CLEAN_DRESS
    assert s.b64_to_reference_rgb(b64, _abaya_seg, dress=True)[30, 192][0] < 200
    s = load(monkeypatch)                                              # cleaner off: nothing changes
    assert not s.REF_CLEAN_DRESS


def test_dress_category_recleans_the_reference_at_set_prompt(monkeypatch):
    """The app sends the photo first and the category with set_prompt: a dress is re-cleaned then."""
    from fastapi.testclient import TestClient
    s = load(monkeypatch, FLUXRT_REF_CLEAN="1", FLUXRT_WIDTH="16", FLUXRT_HEIGHT="16")
    proc = FakeProc()
    calls = []
    real = s.reference_array
    monkeypatch.setattr(s, "reference_array", lambda img, seg=None, dress=False: calls.append(dress) or real(img, seg, dress))
    s.STATE.update(status="ready", processor=proc, crop_fn=lambda f, h, w: f, segmenter=_abaya_seg)
    monkeypatch.setattr(s, "start_warmup_background", lambda: None)
    with TestClient(s.app).websocket_connect(f"/ws?token={token()}") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_reference_image", "image_b64": png_b64(384, 512, (90, 40, 20))})
        assert ws.receive_json() == {"type": "ack", "name": "set_reference_image"}
        ws.send_json({"type": "set_prompt", "prompt": "abaya", "category": "dresses"})
        assert ws.receive_json() == {"type": "ack", "name": "set_prompt"}
        ws.send_json({"type": "set_prompt", "prompt": "abaya again", "category": "dresses"})
        assert ws.receive_json() == {"type": "ack", "name": "set_prompt"}
        ws.send_json({"type": "set_prompt", "prompt": "a top", "category": "tops"})
        assert ws.receive_json() == {"type": "ack", "name": "set_prompt"}
    assert calls == [False, True, False]          # general, dress once, back to general for a top
    assert len(proc.refs) == 3
    time.sleep(0.2)


def test_dress_cleaner_works_without_the_general_cleaner(monkeypatch):
    """Owner's favourite profile keeps FLUXRT_REF_CLEAN off: FLUXRT_REF_CLEAN_DRESS=1 alone cleans only dress
    photos; every other photo goes to FluxRT as sent."""
    s = load(monkeypatch, FLUXRT_REF_CLEAN_DRESS="1")
    assert s.REF_CLEAN_DRESS and not s.REF_CLEAN and s.SEG_NEEDED
    sw = s.live_switches(True)
    assert sw["ref_clean_dress"] is True and sw["ref_clean"] is False
    b64 = png_b64(384, 512, (90, 40, 20))
    dress = s.b64_to_reference_rgb(b64, _abaya_seg, dress=True)
    assert (dress[30, 192] == 255).all() and (dress[150, 192] == 255).all() and dress[400, 192][0] < 200
    other = s.b64_to_reference_rgb(b64, _abaya_seg)
    assert other[30, 192][0] < 200 and other[150, 192][0] < 200       # not a dress: used as sent
    s = load(monkeypatch)
    assert not s.REF_CLEAN_DRESS and not s.SEG_NEEDED                 # default: nothing new switched on


def test_dress_photo_sent_again_keeps_the_current_category(monkeypatch):
    """The app skips a repeated set_prompt, so a photo sent again for the same abaya is still cleaned as a
    dress; a set_reference_image that names its own category uses it."""
    from fastapi.testclient import TestClient
    s = load(monkeypatch, FLUXRT_REF_CLEAN_DRESS="1", FLUXRT_WIDTH="16", FLUXRT_HEIGHT="16")
    proc = FakeProc()
    calls = []
    real = s.reference_array
    monkeypatch.setattr(s, "reference_array", lambda img, seg=None, dress=False: calls.append(dress) or real(img, seg, dress))
    s.STATE.update(status="ready", processor=proc, crop_fn=lambda f, h, w: f, segmenter=_abaya_seg)
    monkeypatch.setattr(s, "start_warmup_background", lambda: None)
    photo = png_b64(384, 512, (90, 40, 20))
    with TestClient(s.app).websocket_connect(f"/ws?token={token()}") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_reference_image", "image_b64": photo})
        assert ws.receive_json() == {"type": "ack", "name": "set_reference_image"}
        ws.send_json({"type": "set_prompt", "prompt": "abaya", "category": "dresses"})
        assert ws.receive_json() == {"type": "ack", "name": "set_prompt"}
        ws.send_json({"type": "set_reference_image", "image_b64": photo})
        assert ws.receive_json() == {"type": "ack", "name": "set_reference_image"}
        ws.send_json({"type": "set_reference_image", "image_b64": photo, "category": "tops"})
        assert ws.receive_json() == {"type": "ack", "name": "set_reference_image"}
        # an abaya filed as "modestwear": the app's try-on type says dress; a hijab filed the same way does not
        ws.send_json({"type": "set_reference_image", "image_b64": photo, "category": "modestwear", "garment_type": "dresses"})
        assert ws.receive_json() == {"type": "ack", "name": "set_reference_image"}
        ws.send_json({"type": "set_prompt", "prompt": "hijab", "category": "modestwear", "garment_type": "accessories"})
        assert ws.receive_json() == {"type": "ack", "name": "set_prompt"}
    assert calls == [False, True, True, False, True, False]
    time.sleep(0.2)


def test_anchor_patch_applies_to_pinned_fluxrt(tmp_path):
    patch_file = Path(__file__).resolve().parents[1] / "fluxrt" / "patches" / "fluxrt-anchor-reference.patch"
    text = patch_file.read_text()
    for marker in ("def set_anchor_image", "use_anchor_image_mask", "mu_override", "anchor_image_seq_len",
                   "self.pipe.spatial_cache.clear()"):
        assert marker in text
    src = Path("/home/user/repos/FluxRT/src/fluxrt/stream_processor")
    if not src.is_dir():
        pytest.skip("pinned FluxRT checkout not present")
    tree = tmp_path / "src" / "fluxrt" / "stream_processor"
    tree.mkdir(parents=True)
    for name in ("model_inference_subprocess.py", "update_controller.py", "pipeline.py", "stream_processor.py"):
        (tree / name).write_bytes((src / name).read_bytes())
    r = subprocess.run(["patch", "--dry-run", "-p1", "-d", str(tmp_path)], input=text, text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr


class FakeTensor:
    def __init__(self, shape):
        self.frame = np.zeros(shape, dtype=np.uint8)
        self.writes = 0

    def copy_from(self, arr):
        self.writes += 1

    def to_numpy(self):
        return self.frame.copy()


class FakeProc:
    """Stands in for FluxRT's StreamProcessor: its scheduler 'writes' a new output every tick."""

    def __init__(self):
        self.input = FakeTensor((8, 8, 3))
        self.output = FakeTensor((8, 8, 3))
        self.prompts, self.refs = [], []

    def get_input_tensor(self): return self.input
    def get_output_tensor(self): return self.output
    def get_resolution(self): return {"width": 8, "height": 8}
    def is_ready(self): return True
    def get_last_processing_time(self): return 0.05
    def set_prompt(self, p): self.prompts.append(p)
    def set_reference_image(self, r): self.refs.append(r.shape)

    def tick(self):
        self.output.frame[:] = (self.output.frame[0, 0, 0] + 1) % 250


def _frames_until(ws, n, budget_s=4.0):
    got, deadline = [], time.time() + budget_s
    while len(got) < n and time.time() < deadline:
        m = ws.receive_json()
        if m.get("type") == "frame":
            got.append(m)
    return got


def test_pacing_on_with_playback_scheduler_streams_without_camera_frames(monkeypatch):
    """Owner's recommended combination: FLUXRT_OUTPUT_PACING=1 + ENABLE_PLAYBACK_SCHEDULER=true.
    The poller refreshes latest_output on its own clock; the scheduler paces the sends."""
    from fastapi.testclient import TestClient
    s = load(monkeypatch, FLUXRT_OUTPUT_PACING="1", FLUXRT_OUTPUT_POLL_HZ="60",
             ENABLE_PLAYBACK_SCHEDULER="true", PLAYBACK_TARGET_FPS="50", FLUXRT_WIDTH="16", FLUXRT_HEIGHT="16")
    proc = FakeProc()
    s.STATE.update(status="ready", processor=proc, crop_fn=lambda f, h, w: f)
    monkeypatch.setattr(s, "start_warmup_background", lambda: None)
    client = TestClient(s.app)
    with client.websocket_connect(f"/ws?token={token()}") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["playback_buffer"] is True and ready["playback_fps"] == 60
        ws.send_json({"type": "set_prompt", "prompt": "p", "category": "tops"})
        assert ws.receive_json() == {"type": "ack", "name": "set_prompt"}
        # Before any camera frame FluxRT's initial output is never published.
        for _ in range(3):
            proc.tick()
            time.sleep(0.05)
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
        # ONE camera frame, then FluxRT keeps producing: the app must get many frames from it.
        ws.send_json({"type": "frame", "frame_b64": png_b64(16, 16, (1, 2, 3))})
        for _ in range(6):
            proc.tick()
            time.sleep(0.06)
        got = _frames_until(ws, 4)
        assert len(got) >= 4 and proc.input.writes == 1
        # The frames differ: real new outputs, not repeats of one stale read.
        assert len({m["frame_b64"] for m in got}) >= 3
    time.sleep(0.2)


def test_pacing_off_reads_output_only_with_camera_frames(monkeypatch):
    from fastapi.testclient import TestClient
    s = load(monkeypatch, FLUXRT_WIDTH="16", FLUXRT_HEIGHT="16")
    proc = FakeProc()
    s.STATE.update(status="ready", processor=proc, crop_fn=lambda f, h, w: f)
    monkeypatch.setattr(s, "start_warmup_background", lambda: None)
    client = TestClient(s.app)
    with client.websocket_connect(f"/ws?token={token()}") as ws:
        ready = ws.receive_json()
        assert "playback_buffer" not in ready
        for _ in range(3):
            proc.tick()
            time.sleep(0.05)
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"            # no frame arrived without a camera frame
        ws.send_json({"type": "frame", "frame_b64": png_b64(16, 16, (1, 2, 3))})
        got = _frames_until(ws, 1)
        assert len(got) == 1 and proc.input.writes == 1
    time.sleep(0.2)


def test_face_paste_and_anchor_refresh_switches(monkeypatch):
    s = load(monkeypatch, FLUXRT_FACE_PASTE="1", FLUXRT_ANCHOR="1", FLUXRT_ANCHOR_REFRESH_S="4")
    assert s.FACE_PASTE and s.SEG_NEEDED and s.ANCHOR_REFRESH_S == 4.0
    assert (s.FACE_PASTE_FEATHER, s.FACE_PASTE_SHRINK, s.FACE_PASTE_MIN_OVERLAP) == (8, 4, 0.6)
    sw = s.live_switches(True)
    assert sw["face_paste"] is True and sw["anchor_refresh_s"] == 4.0
    assert s.live_switches(False)["face_paste"] is False            # no segmenter: off, said so
    s = load(monkeypatch, FLUXRT_FACE_PASTE="1", FLUXRT_COMPOSITE="1")
    assert s.live_switches(True)["face_paste"] is False              # the full composite wins


def test_face_paste_values_are_clamped_and_typos_fall_back(monkeypatch):
    s = load(monkeypatch, FLUXRT_ANCHOR_REFRESH_S="0.5", FLUXRT_FACE_PASTE_MIN_OVERLAP="3",
             FLUXRT_FACE_PASTE_FEATHER="soft", FLUXRT_FACE_PASTE_SHRINK="-2")
    assert s.ANCHOR_REFRESH_S == 2.0 and s.FACE_PASTE_MIN_OVERLAP == 1.0
    assert s.FACE_PASTE_FEATHER == 8 and s.FACE_PASTE_SHRINK == 0
    assert s.live_switches(True)["anchor_refresh_s"] is None         # anchor itself off

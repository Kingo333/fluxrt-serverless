"""server.py with LIVECAM_WEBRTC: off = unchanged; on = the app can move camera and generated frames
onto a WebRTC connection negotiated over the existing WebSocket, with JPEG frames as the fallback."""
import asyncio
import fractions
import time

import numpy as np
import pytest

from livecam import webrtc
from tests.test_fluxrt_live_switches import FakeProc, load, png_b64, token


def test_off_by_default_ready_unchanged_and_offer_refused(monkeypatch):
    from fastapi.testclient import TestClient
    s = load(monkeypatch, FLUXRT_WIDTH="16", FLUXRT_HEIGHT="16")
    assert not s.LIVECAM_WEBRTC and not s.WEBRTC_ON
    assert "webrtc" not in s.ready_message(10.0)
    proc = FakeProc()
    s.STATE.update(status="ready", processor=proc, crop_fn=lambda f, h, w: f)
    monkeypatch.setattr(s, "start_warmup_background", lambda: None)
    with TestClient(s.app).websocket_connect(f"/ws?token={token()}") as ws:
        assert "webrtc" not in ws.receive_json()
        ws.send_json({"type": "webrtc_offer", "sdp": "v=0"})
        assert ws.receive_json() == {"type": "webrtc_unavailable"}
        # The JPEG path is untouched.
        ws.send_json({"type": "frame", "frame_b64": png_b64(16, 16, (1, 2, 3))})
        while True:
            m = ws.receive_json()
            if m.get("type") == "frame":
                break
    time.sleep(0.2)


def test_switch_on_without_aiortc_stays_on_jpeg(monkeypatch):
    monkeypatch.setattr(webrtc, "_IMPORT_ERROR", "ModuleNotFoundError: aiortc")
    s = load(monkeypatch, LIVECAM_WEBRTC="1")
    assert s.LIVECAM_WEBRTC and not s.WEBRTC_ON
    assert "webrtc" not in s.ready_message(10.0)
    assert s.live_switches(False)["webrtc"] is False


@pytest.mark.skipif(not webrtc.available(), reason="aiortc not installed")
def test_switch_on_checks_udp_once_at_startup(monkeypatch):
    from fastapi.testclient import TestClient
    calls = []

    async def fake_probe(*a, **k):
        calls.append(a)
        return True

    s = load(monkeypatch, LIVECAM_WEBRTC="1")
    monkeypatch.setattr(s.livecam_webrtc, "outbound_udp_ok", fake_probe)
    with TestClient(s.app):
        time.sleep(0.1)
    assert len(calls) == 1
    s = load(monkeypatch)  # off: no check
    calls.clear()
    monkeypatch.setattr(s.livecam_webrtc, "outbound_udp_ok", fake_probe)
    with TestClient(s.app):
        time.sleep(0.1)
    assert calls == []


@pytest.mark.skipif(not webrtc.available(), reason="aiortc not installed")
def test_switch_on_ready_says_so_and_bitrates_follow_env(monkeypatch):
    from aiortc.codecs import h264
    saved = (h264.MAX_BITRATE, h264.DEFAULT_BITRATE)
    try:
        s = load(monkeypatch, LIVECAM_WEBRTC="1", LIVECAM_WEBRTC_MAX_KBPS="3500", LIVECAM_WEBRTC_START_KBPS="abc")
        assert s.WEBRTC_ON and s.ready_message(10.0)["webrtc"] is True
        assert s.live_switches(False)["webrtc"] is True and s.live_switches(False)["webrtc_max_kbps"] == 3500
        assert h264.MAX_BITRATE == 3_500_000 and h264.DEFAULT_BITRATE == 2_000_000   # typo -> default 2000
    finally:
        h264.MAX_BITRATE, h264.DEFAULT_BITRATE = saved


@pytest.mark.skipif(not webrtc.available(), reason="aiortc not installed")
def test_offer_answer_media_both_ways_and_no_jpeg_while_connected(monkeypatch):
    """A real aiortc 'app' negotiates over the WebSocket; its camera frames reach FluxRT's input,
    generated frames come back on the track, and the socket carries no JPEG frames meanwhile."""
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamTrack
    import av
    from fastapi.testclient import TestClient

    s = load(monkeypatch, LIVECAM_WEBRTC="1", FLUXRT_WIDTH="16", FLUXRT_HEIGHT="16")
    proc = FakeProc()
    s.STATE.update(status="ready", processor=proc, crop_fn=lambda f, h, w: f)
    monkeypatch.setattr(s, "start_warmup_background", lambda: None)

    class Camera(MediaStreamTrack):
        kind = "video"

        def __init__(self):
            super().__init__()
            self.n = 0

        async def recv(self):
            await asyncio.sleep(1 / 15)
            proc.tick()  # FluxRT keeps producing
            img = np.full((32, 32, 3), 40 + (self.n % 50), np.uint8)
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts, f.time_base = self.n * 6000, fractions.Fraction(1, 90000)
            self.n += 1
            return f

    loop = asyncio.new_event_loop()
    app_pc = RTCPeerConnection()
    got = []
    done = asyncio.Event()

    @app_pc.on("track")
    def on_track(track):
        async def read():
            while len(got) < 5:
                f = await track.recv()
                got.append((f.width, f.height))
            done.set()
        asyncio.ensure_future(read())

    async def make_offer():
        app_pc.addTransceiver(Camera(), direction="sendrecv")
        await app_pc.setLocalDescription(await app_pc.createOffer())
        return app_pc.localDescription.sdp

    try:
        with TestClient(s.app).websocket_connect(f"/ws?token={token()}") as ws:
            ready = ws.receive_json()
            assert ready["webrtc"] is True
            ws.send_json({"type": "webrtc_offer", "sdp": loop.run_until_complete(make_offer()), "ice_servers": []})
            answer = ws.receive_json()
            assert answer["type"] == "webrtc_answer" and answer["sdp"].startswith("v=0") and answer["turn"] == "none"
            loop.run_until_complete(app_pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer")))
            loop.run_until_complete(asyncio.wait_for(done.wait(), timeout=25))
            assert proc.input.writes >= 3                 # camera frames reached FluxRT's input
            assert got[-1][0] > 0 and got[-1][1] > 0       # generated frames came back as video
            ws.send_json({"type": "ping"})
            seen = []
            while True:
                m = ws.receive_json()
                seen.append(m["type"])
                if m["type"] == "pong":
                    break
            assert "frame" not in seen                     # no JPEG over the socket while WebRTC carries frames
            assert "webrtc_state" in seen
            # Back to JPEG: the app closes WebRTC and sends a frame the old way.
            ws.send_json({"type": "webrtc_close"})
            while ws.receive_json().get("type") != "webrtc_state":
                pass
            ws.send_json({"type": "frame", "frame_b64": png_b64(16, 16, (1, 2, 3))})
            while ws.receive_json().get("type") != "frame":
                pass
    finally:
        loop.run_until_complete(app_pc.close())
        loop.close()
    time.sleep(0.2)

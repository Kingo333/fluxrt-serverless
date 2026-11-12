"""livecam/webrtc.py: STUN probe, ICE server choice for aiortc, and a real in-process loopback
(an aiortc 'app' sends 288x512 camera frames, the bridge hands them on and sends 576x1024 back)."""

import asyncio
import os

import numpy as np
import pytest

from livecam import webrtc

CF_ICE = [
    {"urls": ["stun:stun.cloudflare.com:3478", "stun:stun.cloudflare.com:53"]},
    {
        "urls": [
            "turn:turn.cloudflare.com:3478?transport=udp",
            "turn:turn.cloudflare.com:53?transport=udp",
            "turn:turn.cloudflare.com:443?transport=udp",
            "turn:turn.cloudflare.com:3478?transport=tcp",
            "turn:turn.cloudflare.com:80?transport=tcp",
            "turns:turn.cloudflare.com:5349?transport=tcp",
            "turns:turn.cloudflare.com:443?transport=tcp",
        ],
        "username": "u-test",
        "credential": "c-test",
    },
]


def test_stun_request_and_success_check():
    txid = bytes(range(12))
    req = webrtc.stun_binding_request(txid)
    assert len(req) == 20 and req[:2] == b"\x00\x01" and req[8:] == txid
    ok = b"\x01\x01\x00\x00" + req[4:8] + txid
    assert webrtc.is_stun_binding_success(ok, txid)
    assert not webrtc.is_stun_binding_success(ok, bytes(12))
    assert not webrtc.is_stun_binding_success(b"\x01\x01", txid)
    with pytest.raises(ValueError):
        webrtc.stun_binding_request(b"short")


def test_ice_choice_prefers_udp_when_it_works():
    servers, chosen = webrtc.choose_ice_servers(CF_ICE, udp_ok=True)
    assert chosen == "turn:turn.cloudflare.com:3478?transport=udp"
    assert servers[0] == {"urls": ["stun:stun.cloudflare.com:3478"]}
    assert servers[1] == {"urls": [chosen], "username": "u-test", "credential": "c-test"}
    assert webrtc.describe_turn(chosen) == "udp"


def test_ice_choice_uses_tls_443_when_udp_is_blocked():
    servers, chosen = webrtc.choose_ice_servers(CF_ICE, udp_ok=False)
    assert chosen == "turns:turn.cloudflare.com:443?transport=tcp"
    assert servers == [{"urls": [chosen], "username": "u-test", "credential": "c-test"}]  # no STUN: it is UDP
    assert webrtc.describe_turn(chosen) == "tls443"


def test_ice_choice_drops_port_53_bad_urls_and_missing_credentials():
    only53 = [{"urls": ["turn:turn.cloudflare.com:53?transport=udp"], "username": "u", "credential": "c"}]
    assert webrtc.choose_ice_servers(only53, udp_ok=True) == ([], None)
    nocred = [{"urls": ["turn:turn.cloudflare.com:3478?transport=udp"]}]
    assert webrtc.choose_ice_servers(nocred, udp_ok=True) == ([], None)
    assert webrtc.choose_ice_servers("nonsense", udp_ok=True) == ([], None)
    assert webrtc.choose_ice_servers([{"urls": ["http://x"]}, 5], udp_ok=True) == ([], None)
    assert webrtc.describe_turn(None) == "none"
    assert webrtc.parse_ice_url("turns:turn.cloudflare.com")["port"] == 5349


COTURN_ICE = [{
    "urls": ["turn:turn.example.com:3478?transport=udp", "turn:turn.example.com:3478?transport=tcp",
             "turns:turn.example.com:443?transport=tcp", "turns:turn.example.com:5349?transport=tcp"],
    "username": "1794500000:session-1", "credential": "abc=",
}]


def test_works_the_same_with_a_self_hosted_coturn():
    servers, chosen = webrtc.choose_ice_servers(COTURN_ICE, udp_ok=True)
    assert chosen == "turn:turn.example.com:3478?transport=udp" and servers[-1]["username"] == "1794500000:session-1"
    assert webrtc.choose_ice_servers(COTURN_ICE, udp_ok=False)[1] == "turns:turn.example.com:443?transport=tcp"


def test_udp_check_goes_to_the_relay_the_app_will_use():
    assert webrtc.probe_target(CF_ICE) == ("stun.cloudflare.com", 3478)
    assert webrtc.probe_target(COTURN_ICE) == ("turn.example.com", 3478)   # no STUN URL: the TURN UDP port
    assert webrtc.probe_target([]) == (webrtc.STUN_HOST, webrtc.STUN_PORT)
    assert webrtc.probe_target([{"urls": "stun:x.example:53"}, {"urls": ["stun:y.example:3479"]}]) == ("y.example", 3479)


def _fake_stun_server(loop):
    class Answer(asyncio.DatagramProtocol):
        def connection_made(self, transport):
            self.transport = transport

        def datagram_received(self, data, addr):
            if len(data) >= 20 and data[:2] == b"\x00\x01":
                self.transport.sendto(b"\x01\x01\x00\x00" + data[4:20], addr)

    return loop.create_datagram_endpoint(Answer, local_addr=("127.0.0.1", 0))


def test_udp_probe_true_with_an_answering_server_false_without():
    async def run():
        loop = asyncio.get_running_loop()
        transport, _ = await _fake_stun_server(loop)
        port = transport.get_extra_info("sockname")[1]
        try:
            assert await webrtc.probe_outbound_udp("127.0.0.1", port, timeout=0.5)
        finally:
            transport.close()
        # Nobody listens here: no answer within the timeout.
        assert not await webrtc.probe_outbound_udp("127.0.0.1", port, timeout=0.2, tries=1)
        # The cached check: one probe per relay, then remembered.
        transport, _ = await _fake_stun_server(loop)
        port2 = transport.get_extra_info("sockname")[1]
        try:
            assert await webrtc.outbound_udp_ok(("127.0.0.1", port2))
        finally:
            transport.close()
        assert await webrtc.outbound_udp_ok(("127.0.0.1", port2))           # cached: no server needed now
        assert not await webrtc.outbound_udp_ok(("127.0.0.1", port2), force=True)

    asyncio.run(run())


@pytest.mark.skipif(not webrtc.available(), reason="aiortc not installed")
def test_loopback_camera_in_generated_frames_out():
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamTrack
    import av
    import fractions

    class Camera(MediaStreamTrack):
        kind = "video"

        def __init__(self):
            super().__init__()
            self.n = 0

        async def recv(self):
            await asyncio.sleep(1 / 15)
            img = np.zeros((512, 288, 3), np.uint8)
            img[:, :, 1] = 90
            img[(self.n * 7) % 480:(self.n * 7) % 480 + 32, 100:140, :] = 255  # a moving block
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts, f.time_base = self.n * 6000, fractions.Fraction(1, 90000)
            self.n += 1
            return f

    async def run():
        got_in = []
        bridge = None

        async def on_frame(bgr, ts):
            got_in.append(bgr.shape)
            out = np.repeat(np.repeat(bgr[:, :, ::-1], 2, axis=0), 2, axis=1)  # stands in for FluxRT: 2x
            bridge.push_output(out)

        states = []

        async def on_state(s):
            states.append(s)

        bridge = webrtc.RtcBridge(on_frame, on_state)
        app = RTCPeerConnection()
        app.addTransceiver(Camera(), direction="sendrecv")
        got_out = []
        done = asyncio.Event()

        @app.on("track")
        def on_track(track):
            async def read():
                while len(got_out) < 5:
                    f = await track.recv()
                    got_out.append((f.width, f.height))
                done.set()

            asyncio.ensure_future(read())

        offer = await app.createOffer()
        await app.setLocalDescription(offer)
        answer = await bridge.accept_offer(app.localDescription.sdp, [], udp_ok=False)
        assert "H264" in answer  # output codec offered first
        await app.setRemoteDescription(RTCSessionDescription(sdp=answer, type="answer"))
        try:
            await asyncio.wait_for(done.wait(), timeout=25)
        finally:
            await app.close()
            await bridge.close()
        assert got_in and got_in[0] == (512, 288, 3)
        assert got_out[-1] == (576, 1024)
        assert "connected" in states
        assert bridge.frames_in >= 5 and bridge.last_in_size == (288, 512)
        assert bridge.state == "closed" and not bridge.live

    asyncio.run(run())


def test_bitrate_ceiling_is_raised_for_the_576x1024_output():
    if not webrtc.available():
        pytest.skip("aiortc not installed")
    from aiortc.codecs import h264, vpx

    saved = (h264.MAX_BITRATE, h264.DEFAULT_BITRATE, vpx.MAX_BITRATE, vpx.DEFAULT_BITRATE)
    try:
        out = webrtc.configure_bitrates(3500, 2000)
        assert out == {"max_bps": 3_500_000, "start_bps": 2_000_000}
        assert (h264.MAX_BITRATE, vpx.MAX_BITRATE, h264.DEFAULT_BITRATE) == (3_500_000, 3_500_000, 2_000_000)
        assert webrtc.configure_bitrates(100, 9000) == {"max_bps": 500_000, "start_bps": 500_000}
    finally:
        h264.MAX_BITRATE, h264.DEFAULT_BITRATE, vpx.MAX_BITRATE, vpx.DEFAULT_BITRATE = saved


def test_unavailable_bridge_refuses_cleanly(monkeypatch):
    monkeypatch.setattr(webrtc, "_IMPORT_ERROR", "ModuleNotFoundError: aiortc")
    assert not webrtc.available()
    assert "aiortc" in webrtc.unavailable_reason()

    async def nop(*_a):
        return None

    with pytest.raises(RuntimeError):
        webrtc.RtcBridge(nop)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))

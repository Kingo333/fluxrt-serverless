import base64
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from joyai.adapter.app import create_app
from joyai.adapter.config import Settings
from session_token import sign
from tests.joyai.fake_joyai import FakeJoyAI

SECRET = "test-secret"


def tok(tier="premium", exp_in=360, max_s=300):
    return sign(f"u1.g1.{tier}.{max_s}.{int(time.time()) + exp_in}", SECRET)


def settings(port, **kw):
    base = dict(session_signing_secret=SECRET, upstream_command="", upstream_port=port,
                upstream_boot_timeout_s=10, load_timeout_s=10)
    base.update(kw)
    return Settings(**base)


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_refuses_to_start_without_secret():
    with pytest.raises(SystemExit):
        create_app(Settings(session_signing_secret="", upstream_command=""))


def test_ping_204_then_200():
    with FakeJoyAI(load_delay=0.5) as fake, TestClient(create_app(settings(fake.port))) as c:
        assert c.get("/ping").status_code == 204
        assert wait_for(lambda: c.get("/ping").status_code == 200)


def test_ping_503_when_load_fails():
    with FakeJoyAI(load_ok=False) as fake, TestClient(create_app(settings(fake.port))) as c:
        assert wait_for(lambda: c.get("/ping").status_code == 503)
        assert c.get("/health").json()["status"] == "error"


def test_ping_204_when_idle_and_admin_routes_need_token():
    with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port, auto_load=False))) as c:
        assert c.get("/ping").status_code == 204
        for path in ("/warmup", "/api/load"):
            assert c.post(path).status_code == 401
        assert c.get("/debug/startup").status_code == 401
        assert c.post("/warmup?wait=true", headers={"X-Session-Token": tok()}).json()["status"] == "ready"
        assert c.get("/debug/startup", params={"token": tok("standard")}).status_code == 200


def test_ws_rejects_bad_tokens():
    with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port))) as c:
        for bad in ["", "u1.g1.nosig", tok(exp_in=-5), tok().replace("premium", "standard", 1)]:
            with pytest.raises(WebSocketDisconnect):
                with c.websocket_connect(f"/ws?token={bad}") as ws:
                    ws.receive_json()


def test_accepts_standard_and_legacy_by_default_and_respects_accepted_tiers():
    legacy = sign(f"u1.g1.{int(time.time()) + 60}", SECRET)
    with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port))) as c:
        for good in (tok("standard"), legacy):
            with c.websocket_connect(f"/ws?token={good}") as ws:
                assert ws.receive_json()["type"] in ("warming", "ready")
    with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port, accepted_tiers=frozenset({"premium"})))) as c:
        for bad in (tok("standard"), legacy):
            with pytest.raises(WebSocketDisconnect):
                with c.websocket_connect(f"/ws?token={bad}") as ws:
                    ws.receive_json()


def handshake(ws, prompt="Put the red dress from the reference image on the subject."):
    msgs = []
    while True:
        m = ws.receive_json()
        msgs.append(m)
        if m["type"] == "ready":
            break
    ws.send_json({"type": "set_reference_image", "image_b64": base64.b64encode(b"REFJPEG").decode()})
    assert ws.receive_json() == {"type": "ack", "name": "set_reference_image"}
    ws.send_json({"type": "set_prompt", "prompt": prompt})
    assert ws.receive_json() == {"type": "ack", "name": "set_prompt"}
    return msgs


def send_frame_until_echo(ws, fake, starts=1, payload=b"JPEG1"):
    """Frames sent before upstream replies 'started' are dropped (no reply), so wait first."""
    assert wait_for(lambda: len(fake.starts) >= starts)
    time.sleep(0.3)  # let the 'started' reply reach the adapter
    ws.send_json({"type": "frame", "frame_b64": base64.b64encode(payload).decode()})
    m = ws.receive_json()
    while m["type"] != "frame":
        m = ws.receive_json()
    return m, base64.b64decode(m["frame_b64"])


def test_end_to_end_session_and_start_payload():
    with FakeJoyAI(load_delay=0.3) as fake, TestClient(create_app(settings(fake.port))) as c:
        with c.websocket_connect(f"/ws?token={tok()}") as ws:
            msgs = handshake(ws)
            assert msgs[0]["type"] in ("warming", "ready")
            assert wait_for(lambda: len(fake.starts) == 1)
            _, out = send_frame_until_echo(ws, fake)
            assert out == b"OUT:JPEG1"
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
        start = fake.starts[0]
        assert start["width"] == 480 and start["height"] == 840
        assert start["input_codec"] == "mjpeg" and start["output_codec"] == "mjpeg"
        assert start["use_pe"] is False and start["gate_enabled"] is True
        assert start["kv_reset_frames"] == 600 and start["num_inference_steps"] == 2
        assert start["ref_image"] == base64.b64encode(b"REFJPEG").decode()
        assert "red dress" in start["prompt"]


def test_status_passthrough_and_set_param_restarts():
    with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port))) as c:
        fake.emit_waiting_face = True
        with c.websocket_connect(f"/ws?token={tok()}") as ws:
            handshake(ws)
            assert ws.receive_json() == {"type": "status", "status": "waiting_face", "reason": "show_full_face"}
            ws.send_json({"type": "set_param", "name": "steps", "value": 3})
            assert ws.receive_json() == {"type": "ack", "name": "set_param"}
            assert wait_for(lambda: len(fake.starts) == 2)
            assert fake.starts[1]["num_inference_steps"] == 3
            ws.send_json({"type": "set_param", "name": "nope", "value": 1})
            m = ws.receive_json()
            while m["type"] == "status":  # the restart re-emits waiting_face
                m = ws.receive_json()
            assert m["type"] == "error" and "unknown_param" in m["message"]


def test_reconnects_after_upstream_session_timeout():
    with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port))) as c:
        fake.timeout_after_frames = 1
        with c.websocket_connect(f"/ws?token={tok()}") as ws:
            handshake(ws)
            send_frame_until_echo(ws, fake)
            # Upstream released the session: adapter tells the client and reconnects.
            m = ws.receive_json()
            while not (m["type"] == "status" and m["status"] == "session_reset"):
                m = ws.receive_json()
            assert wait_for(lambda: len(fake.starts) == 2)
            _, out = send_frame_until_echo(ws, fake, starts=2, payload=b"J2")
            assert out == b"OUT:J2"
            assert fake.connections == 2 and len(fake.starts) == 2


def test_session_cap_closes():
    with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port, max_session_seconds=1))) as c:
        with c.websocket_connect(f"/ws?token={tok()}") as ws:
            handshake(ws)
            got = None
            for _ in range(50):
                m = ws.receive_json()
                if m.get("type") == "status" and m.get("status") == "session_limit":
                    got = m
                    break
            assert got is not None


def test_instruction_preferred_and_garment_swap_is_one_restart():
    with FakeJoyAI() as fake, TestClient(create_app(settings(fake.port, max_temporal_ids=4))) as c:
        with c.websocket_connect(f"/ws?token={tok()}") as ws:
            while ws.receive_json()["type"] != "ready":
                pass
            ws.send_json({"type": "set_reference_image", "image_b64": base64.b64encode(b"TOP1").decode()})
            assert ws.receive_json()["name"] == "set_reference_image"
            ws.send_json({"type": "set_prompt", "prompt": "long fluxrt prompt",
                          "instruction": "Substitute the current top with the garment in the reference image."})
            assert ws.receive_json()["name"] == "set_prompt"
            assert wait_for(lambda: len(fake.starts) == 1)
            first = fake.starts[0]
            assert first["prompt"].startswith("Substitute the current top")
            assert first["max_temporal_ids"] == 4

            # Mid-session swap: new image + new prompt -> exactly one restart.
            ws.send_json({"type": "set_reference_image", "image_b64": base64.b64encode(b"DRESS2").decode()})
            assert ws.receive_json()["name"] == "set_reference_image"
            ws.send_json({"type": "set_prompt", "prompt": "long dress prompt",
                          "instruction": "Substitute the current outfit with the dress in the reference image."})
            assert ws.receive_json()["name"] == "set_prompt"
            assert wait_for(lambda: len(fake.starts) == 2)
            time.sleep(0.6)
            assert len(fake.starts) == 2
            second = fake.starts[1]
            assert second["ref_image"] == base64.b64encode(b"DRESS2").decode()
            assert second["prompt"].startswith("Substitute the current outfit")

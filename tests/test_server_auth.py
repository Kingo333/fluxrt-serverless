import importlib
import sys
import time

import pytest
from fastapi.testclient import TestClient

from session_token import sign

SECRET = "test-secret"


def load_server(monkeypatch, secret=SECRET):
    if secret is None:
        monkeypatch.delenv("SESSION_SIGNING_SECRET", raising=False)
    else:
        monkeypatch.setenv("SESSION_SIGNING_SECRET", secret)
    sys.modules.pop("server", None)
    return importlib.import_module("server")


def token(tier="standard", exp_in=300):
    return sign(f"u1.g1.{tier}.300.{int(time.time()) + exp_in}", SECRET)


def test_refuses_to_start_without_secret(monkeypatch):
    with pytest.raises(SystemExit):
        load_server(monkeypatch, secret=None)


@pytest.fixture
def client(monkeypatch):
    server = load_server(monkeypatch)
    monkeypatch.setattr(server, "start_warmup_background", lambda: None)
    return TestClient(server.app)


def test_ping_stays_public(client):
    assert client.get("/ping").status_code == 200


@pytest.mark.parametrize("method,path", [
    ("post", "/warmup"), ("post", "/api/load"), ("get", "/debug/startup"),
])
def test_admin_routes_need_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(path, headers={"X-Session-Token": token()}).status_code == 200
    assert getattr(client, method)(f"{path}?token={token()}").status_code == 200


def test_predict_needs_token(client):
    assert client.post("/api/predict", json={}).status_code == 401
    # Authorized but model not loaded -> 503, not 401.
    assert client.post("/api/predict", json={}, headers={"X-Session-Token": token()}).status_code == 503


def test_premium_and_expired_tokens_rejected(client):
    assert client.post("/warmup", headers={"X-Session-Token": token("premium")}).status_code == 401
    assert client.post("/warmup", headers={"X-Session-Token": token(exp_in=-5)}).status_code == 401


def test_ws_rejects_bad_tokens(client):
    from starlette.websockets import WebSocketDisconnect
    for bad in ["", "u1.g1.nosig", token("premium"), token(exp_in=-5)]:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={bad}") as ws:
                ws.receive_json()


def test_ws_session_cap_closes_stream(monkeypatch):
    server = load_server(monkeypatch)

    class FakeTensor:
        def copy_from(self, _): pass
        def to_numpy(self): return None

    class FakeProc:
        def get_input_tensor(self): return FakeTensor()
        def get_output_tensor(self): return FakeTensor()
        def get_resolution(self): return {"width": 8, "height": 8}

    server.STATE.update(status="ready", processor=FakeProc(), crop_fn=lambda f, h, w: f)
    monkeypatch.setattr(server, "MAX_SESSION_SECONDS", 1)
    client = TestClient(server.app)
    with client.websocket_connect(f"/ws?token={token()}") as ws:
        assert ws.receive_json()["type"] == "ready"
        assert ws.receive_json() == {"type": "status", "status": "session_limit"}

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_bootstrap():
    spec = importlib.util.spec_from_file_location("bootstrap_under_test", os.path.join(ROOT, "bootstrap.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_engine_is_fluxrt(monkeypatch):
    monkeypatch.delenv("LIVECAM_ENGINE", raising=False)
    load_bootstrap()._maybe_exec_engine()  # returns: FluxRT runs in-process


def test_joyai_engine_execs_adapter(monkeypatch, tmp_path):
    fake_py = tmp_path / "python"
    fake_py.write_text("")
    calls = []
    monkeypatch.setenv("LIVECAM_ENGINE", "JoyAI")
    monkeypatch.setenv("JOYAI_PYTHON", str(fake_py))
    b = load_bootstrap()
    monkeypatch.setattr(b.os, "execv", lambda p, argv: calls.append((p, argv)))
    monkeypatch.setattr(b.os, "chdir", lambda d: None)
    b._maybe_exec_engine()
    assert calls == [(str(fake_py), [str(fake_py), "-u", "-m", "joyai.adapter.app"])]


def test_joyai_without_venv_stops(monkeypatch):
    monkeypatch.setenv("LIVECAM_ENGINE", "joyai")
    monkeypatch.setenv("JOYAI_PYTHON", "/nonexistent/python")
    with pytest.raises(SystemExit):
        load_bootstrap()._maybe_exec_engine()


def test_unknown_engine_stops(monkeypatch):
    monkeypatch.setenv("LIVECAM_ENGINE", "sdxl")
    with pytest.raises(SystemExit):
        load_bootstrap()._maybe_exec_engine()

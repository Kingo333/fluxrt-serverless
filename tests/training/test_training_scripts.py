"""Fast checks for the training helper scripts (no GPU, no torch needed)."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "training" / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_config_applies_every_replacement(tmp_path):
    mod = load("make_smoke_config")
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "stage1_ar.py").write_text("\n".join(old for old, _ in mod.REPLACEMENTS) + "\n")
    mod.main(str(tmp_path))
    out = (cfg / "azyah_smoke_stage1.py").read_text()
    for old, new in mod.REPLACEMENTS:
        assert new in out
    assert "class AzyahSmokeStage1Config(ExpConfig):" in out
    assert "(480, 832)" in out


def test_smoke_config_fails_loudly_when_upstream_changes(tmp_path):
    mod = load("make_smoke_config")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "stage1_ar.py").write_text("TRAIN_RESOLUTION = (720, 1280)\n")
    with pytest.raises(SystemExit):
        mod.main(str(tmp_path))


def _keys(path, keys):
    path.write_text(json.dumps({"path": str(path), "tensors": len(keys), "params": 0, "keys": keys}))
    return str(path)


def test_compare_same_and_different_layouts(tmp_path, capsys):
    mod = load("dump_checkpoint_keys")
    a = _keys(tmp_path / "a.json", {"w": {"shape": [2, 2], "dtype": "torch.bfloat16"}})
    b = _keys(tmp_path / "b.json", {"w": {"shape": [2, 2], "dtype": "torch.float8_e4m3fn"}})
    c = _keys(tmp_path / "c.json", {"w": {"shape": [2, 3], "dtype": "torch.bfloat16"}, "x": {"shape": [1], "dtype": "torch.bfloat16"}})
    assert mod.compare(a, b) == 0  # same names/shapes; dtype differences are reported but allowed
    assert "dtype differs 1" in capsys.readouterr().out
    assert mod.compare(a, c) == 2

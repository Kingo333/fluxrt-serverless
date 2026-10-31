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


def test_fashion_config_applies_every_replacement(tmp_path):
    mod = load("make_fashion_config")
    cfg = tmp_path / "configs"
    cfg.mkdir()
    upstream = "\n".join(old for old, _ in mod.REPLACEMENTS)
    # the generator inserts its helper before the config class
    upstream = upstream.replace("class Stage1ARConfig(ExpConfig):", "\n@dataclass\nclass Stage1ARConfig(ExpConfig):")
    (cfg / "stage1_ar.py").write_text(upstream + "\n")
    out = mod.main(str(tmp_path)).read_text()
    for _, new in mod.REPLACEMENTS:
        assert new in out
    assert "def _manifest(group):" in out and out.index("def _manifest") < out.index("class AzyahFashionStage1Config")


def test_estimator_matches_the_audit():
    est = load("estimate")
    rows = {(r["gpu"], r["count"]): r for r in est.options("stage1", 20, 45)}
    assert rows[("rtxpro6000", 1)]["fits"] == "no"   # one 96 GB card cannot hold Stage 1
    assert rows[("b200", 2)]["fits"] == "yes"
    assert rows[("h100-80", 4)]["fits"] == "maybe"
    s2 = {(r["gpu"], r["count"]): r for r in est.options("stage2", 10, 60)}
    assert s2[("b200", 1)]["fits"] == "yes"


def test_lora_merge_math_and_layout_check(tmp_path):
    np = pytest.importorskip("numpy")
    mod = load("merge_lora")
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 4)).astype("float32")
    a = rng.standard_normal((2, 4)).astype("float32")
    b = rng.standard_normal((6, 2)).astype("float32")
    base = mod.unwrap_backbone({"model": {"module.blocks.0.attn.q.weight": w, "blocks.0.norm.weight": np.ones(4)}})
    assert set(base) == {"blocks.0.attn.q.weight", "blocks.0.norm.weight"}
    lora = {"base_model.model.blocks.0.attn.q.lora_A.weight": a, "base_model.model.blocks.0.attn.q.lora_B.weight": b}
    merged, n = mod.merge(base, lora, r=2, alpha=4)
    assert n == 1 and np.allclose(merged["blocks.0.attn.q.weight"], w + 2.0 * (b @ a))
    keys = tmp_path / "serving.json"
    keys.write_text(json.dumps({"keys": {"blocks.0.attn.q.weight": {"shape": [6, 4]}, "blocks.0.norm.weight": {"shape": [4]}}}))
    assert mod.check_keys(merged, str(keys)) == []
    keys.write_text(json.dumps({"keys": {"blocks.0.attn.q.weight": {"shape": [6, 5]}}}))
    problems = mod.check_keys(merged, str(keys))
    assert any("shape" in p for p in problems) and any("not in serving" in p for p in problems)
    with pytest.raises(SystemExit, match="not in the Stage 1 backbone"):
        mod.merge(base, {"x.lora_A.weight": a, "x.lora_B.weight": b}, 2, 4)


def test_prune_keeps_latest_and_marked_saves(tmp_path):
    import os
    import time
    mod = load("prune_checkpoints")
    run = tmp_path / "run" / "checkpoints"
    old = time.time() - 600
    for step in (250, 500, 750, 1000, 1250):
        d = run / f"global_step{step}"
        d.mkdir(parents=True)
        f = d / f"step_{step}.pth"
        f.write_text("x")
        os.utime(f, (old, old))
    (run / "global_step500" / "KEEP").write_text("")
    # a save still being written: no settle time yet
    d = run / "global_step1500"
    d.mkdir()
    (d / "step_1500.pth").write_text("x")
    removed = mod.prune(str(tmp_path / "run"), keep_last=2)
    left = sorted(p.name for p in run.iterdir())
    assert left == ["global_step1000", "global_step1250", "global_step1500", "global_step500"]
    assert len(removed) == 2


def test_model_card_states_vivid_and_blocks_commercial_use():
    import sys as _sys
    _sys.path.insert(0, str(SCRIPTS.parent))
    from data_tools import manifests
    mod = load("model_card")
    entries = [{"path": "a.tar", "group": "v2v", "source": "vivid", "split": "train", "num_samples": 10},
               {"path": "b.tar", "group": "i2i", "source": "glance", "split": "train", "num_samples": 5}]
    res = manifests.build(entries, "research", "train")
    assert res["trained_with_vivid"] and not res["commercial_use_cleared"] and res["research_only_sources"] == ["vivid"]
    card, entry = mod.build(res, "dit.pth", "azyah-vton-experimental", "stage1 step 2000", "n", "abc")
    assert "Not cleared for commercial use" in card and "Trained with ViViD: **yes**" in card
    assert "Glance VTON Synthetic Dataset (2026)" in card
    assert entry["commercial_use_cleared"] is False and entry["trained_with_vivid"] is True
    com = manifests.build(entries, "commercial", "train")
    assert not com["trained_with_vivid"] and com["commercial_use_cleared"] and com["notice"] == ""
    card2, entry2 = mod.build(com, "dit.pth", "azyah-vton-candidate", "x", "n", "abc")
    assert "Not cleared" not in card2 and entry2["commercial_use_cleared"] is True

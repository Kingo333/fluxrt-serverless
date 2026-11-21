"""A full practice run of fx6b_continue.sh with stand-in tools (no GPU, no network, no RunPod).

Why (2026-11-17): the script's mistakes so far were all in the orchestration, and each cost real Pod time: a
bare `wait` that hung for 40 minutes, a memory limit read from the wrong place, a start line that fired before
it was swapped. Here the whole script runs against a fake volume: every Python tool, the trainer, nvidia-smi,
curl, git, runpodctl and sleep are stand-ins that only write the files and lines the script looks for. The
real pick_best_save runs, so the ranking, the chance rule and the verdict are the real ones.

What a run must do: finish (no hang), stop its Pod once, do the checks in the right order, hand the trainer the
right start and settings, never run two scorers at once, and leave the best file where the live worker reads it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("safetensors")                # the real strength tool reads real LoRA files here
from safetensors.numpy import save_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "training/fluxrt_lora/fx6b_continue.sh"
SETS = ("fx6test", "cases5", "dress10")
TAG = "r528"
# what the two stand-in clips of each test set are (the real lists: 8 tops, 8 dresses, 8 bottoms; 10 dresses; the owner's 5)
CLIP_TYPES = {"fx6test": ("tops", "bottoms"), "dress10": ("dresses", "dresses"), "cases5": ("dresses", "tops")}

FAKE_PY = r'''#!/bin/bash
# stand-in for the training venv's python
REAL="__REAL__"
if [ "$1" = "-c" ]; then
  case "$2" in
    *"import torch, diffusers, pyarrow"*) exit 0;;
    *safe_open*) echo 64; exit 0;;
    *pyarrow.parquet*) echo 7; exit 0;;
  esac
  exec "$REAL" "$@"
fi
if [ "$1" = "-m" ]; then
  mod="$2"; shift 2
  case "$mod" in
    fluxrt_lora.pick_best_save|fluxrt_lora.lora_geometry|eval.garment_mix) exec "$REAL" -m "$mod" "$@";;   # the real tools
    fluxrt_lora.merge_lora|eval.generate_fluxrt|eval.garment_report|eval.step_a) exec "$REAL" "$STUB_DIR/tools.py" "$mod" "$@";;
  esac
  echo "unexpected module $mod" >&2; exit 3
fi
exec "$REAL" "$@"
'''

FAKE_MP = r'''#!/bin/bash
# stand-in for the MediaPipe venv's python
REAL="__REAL__"
if [ "$1" = "-c" ]; then exit 0; fi
if [ "$1" = "-m" ]; then mod="$2"; shift 2; exec "$REAL" "$STUB_DIR/tools.py" "$mod" "$@"; fi
exec "$REAL" "$@"
'''

# One file for every stand-in tool: `tools.py <module> <args...>`.
TOOLS = r'''
import json, os, sys, time
from pathlib import Path
mod, args = sys.argv[1], sys.argv[2:]
LOG = Path(os.environ["STUB_LOG"])
def opt(name, default=None, many=False):
    if name not in args:
        return default
    i = args.index(name) + 1
    if not many:
        return args[i]
    out = []
    while i < len(args) and not args[i].startswith("--"):
        out.append(args[i]); i += 1
    return out
def log(name, rec):
    with open(LOG / name, "a") as f:
        f.write(json.dumps(rec) + "\n")
def colour(name):
    """The made-up garment colour score of each model (favourite #2 = 12.0, lower is better)."""
    if name.startswith("fav2seed"): return 12.1
    if name.startswith("fx6s5800s65"): return 11.0
    if name.startswith("fx6s5800s80"): return 11.6
    if name.startswith("fx6blend"): return 12.5
    if name.startswith("fx6s5800"): return 13.0
    if name.startswith("fav2rt"): return 11.8 if "rt50" in name else 11.9
    if name.startswith("fx9a"): return 10.6 if "2000x" in name else 10.9
    if name.startswith("fx9b"): return 9.9 if "2000x" in name else 11.2
    if name.startswith("fx8"):
        for step, c in (("750x", 10.9), ("1000x", 11.1), ("500x", 11.4), ("250x", 11.7)):
            if name[3:].startswith(step): return c
    for step, c in (("2000x", 10.6), ("1500x", 10.2), ("1000x", 10.8), ("500x", 11.5)):
        if step in name: return c
    return 12.0
def lora(path):
    try:
        from safetensors.numpy import load_file
        return load_file(str(path))
    except Exception:
        return None
if mod == "fluxrt_lora.merge_lora":
    out = Path(opt("--out")); out.mkdir(parents=True, exist_ok=True)
    rec = {"inputs": opt("--inputs", many=True), "weights": opt("--weights", many=True), "scale": opt("--scale", "1.0"), "out": str(out)}
    sds = [lora(p) for p in rec["inputs"]]
    if all(sd is not None for sd in sds):
        # real LoRA files: the same sums as merge_lora (A stacked along rank; B scaled, stacked along rank)
        import numpy as np
        from safetensors.numpy import save_file
        scale = float(rec["scale"])
        if os.environ.get("STUB_BAD_UNSCALE") and scale > 1:
            scale = 1.2                                    # a made-up fault in the un-scaling
        merged = {}
        for k in sds[0]:
            if ".lora_A." in k:
                merged[k] = np.concatenate([sd[k] for sd in sds], 0)
            else:
                merged[k] = np.concatenate([sd[k] * np.float32(float(w) * scale) for sd, w in zip(sds, rec["weights"])], 1)
        save_file(merged, str(out / "pytorch_lora_weights.safetensors"))
    else:
        (out / "pytorch_lora_weights.safetensors").write_text("merged " + json.dumps(rec, sort_keys=True))
    (out / "selection.json").write_text(json.dumps({"merged_from": rec["inputs"], "strength_baked_in": float(rec["scale"])}))
    log("merge.log", rec)
elif mod == "eval.generate_fluxrt":
    out, name = Path(opt("--out")), opt("--name")
    log("render_try.log", {"name": name, "set": out.name})
    if os.environ.get("STUB_FAIL_RENDER") and name.startswith(os.environ["STUB_FAIL_RENDER"]):
        sys.exit("made-up render failure")
    assert Path(opt("--lora")).exists(), "the renderer was given a LoRA file that does not exist"
    new = 0
    for clip in ("c1", "c2"):
        d = out / clip; d.mkdir(parents=True, exist_ok=True)
        if not (d / f"out_{name}.mp4").exists():
            (d / f"out_{name}.mp4").write_text("video"); new += 1; print(json.dumps({"model": name, "clip": clip}))
    log("render.log", {"name": name, "set": out.name, "seed": opt("--seed"), "steady": opt("--steady-blend"), "new": new,
                       "t": time.time()})
    time.sleep(0.05)
elif mod == "eval.garment_report":
    if args[0] == "score":
        models = opt("--models", many=True); base = opt("--baseline"); name = [m for m in models if m != base][0]
        if os.environ.get("STUB_FAIL_SCORE") and name.startswith(os.environ["STUB_FAIL_SCORE"]):
            sys.exit("made-up scoring failure")
        rows = [{"priority": "COLOUR", "metric": "colour_de", "model": name, "baseline": base, "mean": colour(name), "baseline_mean": 12.0}]
        # per clip: the first clip of every set a little better than the set's mean, the second a little worse
        items = {c: {name: {"colour_de": colour(name) + d, "fabric_ratio": round(colour(name) / 12.0, 3)},
                     base: {"colour_de": 12.0, "fabric_ratio": 0.9}} for c, d in (("c1", -0.4), ("c2", 0.4))}
        Path(opt("--json")).write_text(json.dumps({"scorecard": rows, "items": items}))
        Path(opt("--out")).write_text("| garment |\n")
elif mod == "eval.face_skin_report":
    root = Path(args[0]); models = opt("--models", many=True); base = opt("--baseline"); name = [m for m in models if m != base][0]
    log("score.log", {"ev": "start", "name": name, "set": root.name, "t": time.time()})
    time.sleep(0.15)
    face = {"face_match": 0.93, "skin_de": 3.0, "face_drift": 1.0, "texture": 1.0}
    (root / "face_skin.json").write_text(json.dumps({"average": {name: face, base: face}, "verdict": {name: {"pass": True}}}))
    Path(opt("--out")).write_text("| face |\n")
    log("score.log", {"ev": "end", "name": name, "set": root.name, "t": time.time()})
elif mod == "eval.preserve_check":
    root = Path(args[0]); name = opt("--candidates", many=True)[0]
    # favourite #2 with another seed fails its own GATE on one set: chance, not a fault
    ok = not (name.startswith("fav2seed") and root.name == "fx6test")
    (root / "preserve_check.json").write_text(json.dumps({"verdict": {name: {"pass": ok, "fails": [], "clips": 2}}, "per_clip": {}}))
    Path(opt("--out")).write_text("| gate |\n")
elif mod == "eval.stability_report":
    out = Path(opt("--out")); out.write_text("| steady |\n"); out.with_suffix(".json").write_text("{}")
elif mod == "eval.step_a":
    root, s = Path(args[1]), opt("--set")
    (root / "eval/fxa" / f"{s}_labelled.mp4").write_text("video")
    log("video.log", {"set": s, "panels": opt("--panels", many=True), "cols": opt("--cols"), "legend": "--no-legend" not in args})
elif mod == "fake.addon":
    import numpy as np
    from safetensors.numpy import load_file, save_file
    frozen = load_file(args[1])
    rng = np.random.default_rng(int(args[2]))
    sd = {}
    for k, v in frozen.items():
        if ".lora_A." in k:
            sd[k] = rng.normal(0, 0.1, (2, v.shape[1])).astype(np.float32)
        else:
            sd[k] = rng.normal(0, 0.1, (v.shape[0], 2)).astype(np.float32)
    save_file(sd, args[0])
else:
    sys.exit(f"unexpected tool {mod}")
'''

FAKE_TRAIN = r'''#!/bin/bash
# stand-in for run_train.sh: records what it was asked to do and makes a save at every interval
[ "${STUB_TRAIN_FAIL:-0}" = "1" ] && { echo "Traceback (most recent call last): made-up trainer crash" >&2; exit 9; }
echo "AZ_EXP=$AZ_EXP AZ_MAX_STEPS=$AZ_MAX_STEPS AZ_LR=$AZ_LR AZ_LORA_RANK=$AZ_LORA_RANK AZ_INIT_LORA=$AZ_INIT_LORA AZ_ANCHOR_COLUMN=$AZ_ANCHOR_COLUMN AZ_ANCHOR_PROB=$AZ_ANCHOR_PROB AZ_CKPT_INTERVAL=$AZ_CKPT_INTERVAL KLEIN_REPO=$KLEIN_REPO AZ_TEXT_CACHE_DIR=$AZ_TEXT_CACHE_DIR AZ_FROZEN_LORA=$AZ_FROZEN_LORA AZ_STEADY_RATIO=$AZ_STEADY_RATIO AZ_REF_WEIGHT=$AZ_REF_WEIGHT AZ_SAFETY_STEPS=$AZ_SAFETY_STEPS AZ_PAIR_FLOOR=${AZ_PAIR_FLOOR:-} AZ_DETAIL_FLOOR=${AZ_DETAIL_FLOOR:-} AZ_TEXTURE_FLOOR=${AZ_TEXTURE_FLOOR:-} AZ_KEEP_SPREAD=${AZ_KEEP_SPREAD:-} end" >> "$STUB_LOG/train_env.log"
OUT="$AZ_ROOT/outputs/fluxrt_lora/$AZ_EXP"; mkdir -p "$OUT" "$AZ_ROOT/reports"
LOG="$AZ_ROOT/reports/fluxlora_${AZ_EXP}_test.log"
if [ -n "${AZ_FROZEN_LORA:-}" ]; then
  # fxreal8: no starting LoRA; favourite #2 un-scaled merged into the frozen model, a small add-on on top
  [ -z "${AZ_INIT_LORA:-}" ] || { echo "both a starting LoRA and a frozen LoRA" >&2; exit 9; }
  [ -f "$AZ_FROZEN_LORA/pytorch_lora_weights.safetensors" ] || { echo "no frozen LoRA in $AZ_FROZEN_LORA" >&2; exit 9; }
  echo "FROZEN_LORA merged 3 layers of $AZ_FROZEN_LORA at x1 into the frozen weights (rounding to torch.bfloat16: 6.1% of the change)" >> "$LOG"
  echo "FX8_ADDON rank $AZ_LORA_RANK on the same 3 layers as the frozen LoRA" >> "$LOG"
  if [ "${STUB_FX8_FAIL:-0}" = "1" ]; then
    echo "FX8_SAFETY_FAILED: the add-on does not start at the frozen model: reference term 0.4 at the first step (data term 0.9)" >> "$LOG"
    exit 3
  fi
  if [ -f "$OUT/fx8_steady_weight.json" ]; then echo "STEADY_WEIGHT 2.5 (from the first start of this run)" >> "$LOG"
  else echo "STEADY_WEIGHT 2.5" >> "$LOG"; echo '{"steady_weight": 2.5}' > "$OUT/fx8_steady_weight.json"; fi
  [ -n "${AZ_PAIR_FLOOR:-}${AZ_KEEP_SPREAD:-}" ] && echo "FX9_TERMS same-clip colour floor ${AZ_PAIR_FLOOR:-0}, detail floor ${AZ_DETAIL_FLOOR:-0}, fabric-contrast floor ${AZ_TEXTURE_FLOOR:-0} (1 = counted fully at every noise level), keep-spread weight ${AZ_KEEP_SPREAD:-0}" >> "$LOG"
  echo "FX8_SAFETY_OK: start = the frozen model (reference term 0.0e+00 at the first step); steadiness term on in 20/20 batches; steadiness weight 2.5; every term finite over the first 20 steps" >> "$LOG"
else
  [ -f "$AZ_INIT_LORA/pytorch_lora_weights.safetensors" ] || { echo "no start weights in $AZ_INIT_LORA" >&2; exit 9; }
  echo "LoRA initialised from $AZ_INIT_LORA (336 tensors)" >> "$LOG"
fi
[ "$AZ_ANCHOR_PROB" = "0" ] && echo "ANCHOR_BATCHES 0 with the earlier frame, 2000 without" >> "$LOG" || echo "ANCHOR_BATCHES 1100 with the earlier frame, 900 without" >> "$LOG"
last=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1); last=${last:-0}
n=$(( last + AZ_CKPT_INTERVAL ))
while [ "$n" -le "$AZ_MAX_STEPS" ]; do
  sleep 1
  mkdir -p "$OUT/checkpoint-$n"
  if [ -n "${AZ_FROZEN_LORA:-}" ]; then
    "$STUB_PY" "$STUB_DIR/tools.py" fake.addon "$OUT/checkpoint-$n/pytorch_lora_weights.safetensors" "$AZ_FROZEN_LORA/pytorch_lora_weights.safetensors" "$n"
  else
    echo "weights of $AZ_EXP at step $n" > "$OUT/checkpoint-$n/pytorch_lora_weights.safetensors"
  fi
  echo "optimiser state at step $n of $AZ_EXP" > "$OUT/checkpoint-$n/optimizer.bin"
  touch -d '3 minutes ago' "$OUT/checkpoint-$n/pytorch_lora_weights.safetensors"
  echo "Steps: loss=0.1 colour=0.01${AZ_FROZEN_LORA:+ steady=0.02 ref=0.003}${AZ_KEEP_SPREAD:+ keep=0.004}" >> "$LOG"
  n=$(( n + AZ_CKPT_INTERVAL ))
done
echo "final weights" > "$OUT/pytorch_lora_weights.safetensors"
'''


def _exe(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _sha8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def _lora(seed: int, rank: int = 4, layers: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    sd = {}
    for i in range(layers):
        sd[f"transformer.blocks.{i}.attn.to_q.lora_A.weight"] = rng.normal(0, 0.1, (rank, 24)).astype(np.float32)
        sd[f"transformer.blocks.{i}.attn.to_q.lora_B.weight"] = rng.normal(0, 0.1, (16, rank)).astype(np.float32)
    return sd


def _trained_from(fav: dict, strength: float, new: float, seed: int) -> dict:
    """A made-up trained file: favourite #2's change at `strength`, plus something new, as one LoRA."""
    extra = _lora(seed)
    out = {}
    for k in fav:
        if ".lora_A." in k:
            out[k] = np.concatenate([fav[k], extra[k]], 0)
        else:
            out[k] = np.concatenate([fav[k] * np.float32(strength), extra[k] * np.float32(new)], 1)
    return out


class Volume:
    """A fake training volume, the stand-in tools and the environment of one Pod."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.root = tmp / "azyah-vton"
        self.log = tmp / "log"
        self.log.mkdir()
        stubs, bins = tmp / "stubs", tmp / "bin"
        stubs.mkdir()
        (stubs / "tools.py").write_text(TOOLS)
        (stubs / "fake_train.sh").write_text(FAKE_TRAIN)
        real_sleep = shutil.which("sleep")
        _exe(bins / "sleep", f"#!/bin/bash\nexec {real_sleep} 0.2\n")           # a minute of waiting = 0.2 s
        _exe(bins / "nvidia-smi", '#!/bin/bash\ncase "$*" in *memory.free*) echo 80000;; *) echo "RTX PRO 6000, 97887 MiB";; esac\n')
        _exe(bins / "runpodctl", '#!/bin/bash\necho "$@" >> "$STUB_LOG/runpodctl.log"\n')
        _exe(bins / "curl", '#!/bin/bash\nfor a in "$@"; do case "$a" in */run_train.sh) cat "$STUB_DIR/fake_train.sh"; exit 0;; esac; done\nexit 0\n')
        _exe(bins / "git", "#!/bin/bash\nexit 0\n")
        _exe(bins / "ffmpeg", "#!/bin/bash\nexit 0\n")
        r = self.root
        (r / "azyah-code").mkdir(parents=True)
        (r / "azyah-code/training").symlink_to(ROOT / "training")
        _exe(r / "venv-fluxlora/bin/python", FAKE_PY.replace("__REAL__", sys.executable))
        _exe(r / "venv-mp/bin/python", FAKE_MP.replace("__REAL__", sys.executable))
        lora = r / "outputs/fluxrt_lora"
        fav = _lora(1)
        for name, sd in (("blend05s065", fav), ("fxreal6", _trained_from(fav, 1.45, 0.5, 2))):
            (lora / name).mkdir(parents=True)
            save_file(sd, str(lora / name / "pytorch_lora_weights.safetensors"))
            (lora / name / "selection.json").write_text("{}")
        data = r / "data/fluxrt_lora/fxreal6"
        (data / "data").mkdir(parents=True)
        (data / "dataset_summary.json").write_text("{}")
        (data / "region_weight.json").write_text(json.dumps({"rows_written": 100}))
        (data / "selection.json").write_text(json.dumps({"commercial_use_cleared": False}))
        fxa = r / "eval/fxa"
        for s in SETS:
            (fxa / f"{s}.json").parent.mkdir(parents=True, exist_ok=True)
            (fxa / f"{s}.json").write_text(json.dumps([{"key": c, "category": t, "case": s}
                                                       for c, t in zip(("c1", "c2"), CLIP_TYPES[s])]))
            for clip in ("c1", "c2"):
                d = fxa / "results" / s / clip
                d.mkdir(parents=True)
                (d / "input.mp4").write_text("camera")
                (d / f"out_fav2_{TAG}.mp4").write_text("favourite 2")
        (r / "reports").mkdir()
        (tmp / "textcache").mkdir()
        self.env = {
            "PATH": f"{bins}:{os.environ['PATH']}", "HOME": str(tmp), "STUB_LOG": str(self.log), "STUB_DIR": str(stubs),
            "STUB_PY": sys.executable,
            "AZ_ROOT": str(r), "AZ_REF": "testref", "AZ_SPENT": "53.17", "AZ_CAP": "65", "AZ_PRICE": "2.80",
            "AZ_BALANCE": "13.00", "AZ_TEXT_CACHE_DIR": str(tmp / "textcache"), "RUNPOD_POD_ID": "testpod",
            "AZ_FAV2_SHA": _sha8(lora / "blend05s065/pytorch_lora_weights.safetensors"),
            "AZ_START_SHA": _sha8(lora / "fxreal6/pytorch_lora_weights.safetensors"),
        }
        self.ev = fxa / "fx6b"

    def add_unscored_blend(self):
        """What the 2026-11-17 Pod left: a 50% blend, rendered, never scored, marked failed by hand."""
        d = self.ev / "b50x4e508f08"
        d.mkdir(parents=True)
        w = self.root / "outputs/fluxrt_lora/fxreal6c/blends/b50/pytorch_lora_weights.safetensors"
        w.parent.mkdir(parents=True)
        w.write_text("blend 50")
        name = f"fx6blend50x4e508f08_{TAG}"
        (d / "meta.json").write_text(json.dumps({"step": -50, "name": name, "weights": str(w), "sha256": "x", "kind": "blend",
                                                 "label": "blend: step 5800 at 50% + favourite #2"}))
        (d / ".rendered").touch()
        (d / ".failed").touch()
        for s in SETS:
            for clip in ("c1", "c2"):
                (self.root / "eval/fxa/results" / s / clip / f"out_{name}.mp4").write_text("video")
        # and an earlier run's videos and review file, which must be kept
        (self.ev / "fx6c_cases5_best.mp4").write_text("old video")
        (self.root / "eval/fx6c_review.tar").write_text("old review")

    def run(self, timeout: int = 240, **extra) -> tuple[str, float]:
        t0 = time.time()
        p = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env={**self.env, **extra}, timeout=timeout)
        out = p.stdout + p.stderr
        assert p.returncode == 0, out[-3000:]
        return out, time.time() - t0

    def lines(self, name: str) -> list:
        f = self.log / name
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() and name.endswith(".log") and name != "train_env.log" \
            and name != "runpodctl.log" else (f.read_text().splitlines() if f.exists() else [])

    def scored(self) -> dict:
        return {json.loads((d / "meta.json").read_text())["name"]: d for d in self.ev.iterdir()
                if (d / "meta.json").exists() and (d / ".scored").exists()}


def _no_two_scorers_at_once(events: list):
    """The scorers write one shared file per set: two models must never be scored on one set at the same time."""
    busy = {}
    for e in sorted(events, key=lambda e: e["t"]):
        if e["ev"] == "start":
            assert busy.get(e["set"]) is None, f"{e['name']} and {busy[e['set']]} were scored on {e['set']} at once"
            busy[e["set"]] = e["name"]
        else:
            assert busy.get(e["set"]) == e["name"]
            busy[e["set"]] = None


@pytest.fixture
def vol(tmp_path):
    return Volume(tmp_path)


def test_fxreal7_practice_run_from_favourite_2(vol):
    vol.add_unscored_blend()
    out, secs = vol.run(AZ_TRAIN_FROM="fav2")
    root = vol.root
    # it finished, in good time, and stopped its Pod exactly once
    assert "FX6B_DONE" in out and secs < 200, out[-2000:]
    assert vol.lines("runpodctl.log") == ["stop pod testpod"]

    # order: favourite #2 confirmed -> strength check -> the checks without training -> training -> pick
    order = ["1. favourite #2 confirmed", "GEOMETRY fxreal6 step 5800: strength x1.4",
             "GEOMETRY fxreal7 start (favourite #2 un-scaled): strength x1.54 of favourite #2, new part 0.00, cosine 1.00",
             "2. the training start is favourite #2 at full strength (x 1.54) and nothing else: confirmed",
             "2. before the training: rendering",
             f"render dress10/fx6s5800s80_{TAG}", "2. renders before the training done", "4. training fxreal7 from step 0 to 2000",
             "5. finishing the checks", "CHANCE favourite #2 with another random seed", "BEST_SAVE", "VERDICT", "FX6B_DONE"]
    at = [out.find(s) for s in order]
    assert all(i >= 0 for i in at) and at == sorted(at), list(zip(order, at))
    renders = vol.lines("render.log")
    first_save_render = min(r["t"] for r in renders if r["name"].startswith("fx7"))
    for name in (f"fav2seed53_{TAG}", f"fx6s5800s65_{TAG}", f"fx6s5800s80_{TAG}"):
        assert max(r["t"] for r in renders if r["name"] == name) < first_save_render
    # chance = favourite #2's own file with another seed; every other render uses the usual seed
    assert {r["seed"] for r in renders if r["name"].startswith("fav2seed53")} == {"53"}
    assert {r["seed"] for r in renders if not r["name"].startswith("fav2seed53")} == {None}

    # the trainer: favourite #2 un-scaled as the start, the base model, no earlier-frame input, the fxreal6 settings
    env = vol.lines("train_env.log")
    assert len(env) == 1
    for s in ("AZ_EXP=fxreal7 ", "AZ_MAX_STEPS=2000 ", "AZ_LR=5e-5 ", "AZ_LORA_RANK=64 ", "AZ_ANCHOR_PROB=0 ", "AZ_CKPT_INTERVAL=500 ",
              f"AZ_INIT_LORA={root}/outputs/fluxrt_lora/blend05_unscaled ", "KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B "):
        assert s in env[0], (s, env[0])
    assert "FX6B trainer: ANCHOR_BATCHES 0 with the earlier frame" in out and "earlier-frame input off" in out

    # the files: favourite #2 x 1/0.65 once; step 5800 at 0.65 and 0.80; every save turned down to 0.65
    merges = vol.lines("merge.log")
    fav = str(root / "outputs/fluxrt_lora/blend05s065/pytorch_lora_weights.safetensors")
    un = [m for m in merges if m["inputs"] == [fav]]
    assert len(un) == 1 and un[0]["scale"] == "1.538462" and un[0]["weights"] == ["1.0"]
    # turned back down to 0.65, the training start is favourite #2 again (to rounding)
    from safetensors.numpy import load_file
    a, b = load_file(fav), load_file(str(root / "outputs/fluxrt_lora/blend05_unscaled/pytorch_lora_weights.safetensors"))
    assert all(np.allclose(a[k], b[k] * (0.65 if ".lora_B." in k else 1.0), rtol=1e-5, atol=0) for k in a)
    s5800 = str(root / "outputs/fluxrt_lora/fxreal6_s5800/pytorch_lora_weights.safetensors")
    assert sorted(m["scale"] for m in merges if m["inputs"] == [s5800]) == ["0.65", "0.80"]
    saves = [m for m in merges if "/fxreal7/saves/" in m["inputs"][0]]
    assert len(saves) == 4 and {m["scale"] for m in saves} == {"0.65"}
    assert out.count("registered for checking") == 4           # each save once, not again at every look

    # everything was scored, the hand-marked blend included; never two scorers on one set at once
    scored = vol.scored()
    names = set(scored)
    assert {f"fav2seed53_{TAG}", f"fx6s5800s65_{TAG}", f"fx6s5800s80_{TAG}", f"fx6s5800_{TAG}", f"fx6blend50x4e508f08_{TAG}"} <= names
    fx7 = sorted(n for n in names if n.startswith("fx7"))
    assert len(fx7) == 4 and all(n.endswith(f"s65_{TAG}") for n in fx7)
    assert not (vol.ev / "b50x4e508f08/.failed").exists() and "was rendered but never scored" in out
    _no_two_scorers_at_once(vol.lines("score.log"))

    # the pick: the save at step 1500 (the best made-up colour), not favourite #2's twin; kept turned down to 0.65
    best = json.loads((vol.ev / "best.json").read_text())
    assert best["best"]["label"] == "fxreal7 step 1500 at 0.65" and best["chance"]["gate_sets"] == 2
    assert best["weights"] == {"fx6test": 1.0, "dress10": 1.0, "cases5": 0.5}
    kept = (root / "outputs/fluxrt_lora/fxreal7/best/pytorch_lora_weights.safetensors").read_text()
    assert kept.startswith("merged ") and '"scale": "0.65"' in kept and "checkpoint" not in kept
    assert json.loads((root / "outputs/fluxrt_lora/fxreal7/best/selection.json").read_text())["strength_baked_in"] == 0.65
    assert "VERDICT fxreal7 step 1500 at 0.65 beats favourite #2 beyond chance" in out
    assert "FX6B AREAS fxreal7 step 1500 at 0.65: colour +15.0%" in out

    # the videos: favourite #2, step 5800 at its best strength, this run's best save; the old ones kept
    videos = {v["set"]: v["panels"] for v in vol.lines("video.log")}
    assert set(videos) == {"dress10", "cases5", "fx6test", "g_dresses", "g_tops", "g_bottoms"}   # per set, per garment type
    assert videos["dress10"][0] == f"fav2_{TAG}=favourite #2 (today)"
    assert f"fx6s5800s65_{TAG}=fxreal6 step 5800 at 0.65" in videos["dress10"]
    assert any(p.startswith("fx7") and p.endswith("=fxreal7 step 1500 at 0.65") for p in videos["dress10"])
    assert not any(p.startswith("fav2seed53") for p in videos["dress10"])
    for s in SETS:
        assert (vol.ev / f"fx7_{s}_best.mp4").exists()
    backup = vol.ev / "backup_before_fxreal7"
    assert (backup / "fx6c_cases5_best.mp4").read_text() == "old video" and (backup / "fx6c_review.tar").exists()
    assert (root / "eval/fx7_review.tar").exists()


def test_fxreal7_restart_counts_the_money_and_carries_on_from_its_best_save(vol):
    out1, _ = vol.run(AZ_TRAIN_FROM="fav2", AZ_MAX_STEPS="1000")
    assert "FX6B_DONE" in out1 and "4. training fxreal7 from step 0 to 1000" in out1
    (vol.ev / "spend").mkdir(exist_ok=True)
    (vol.ev / "spend" / "1.run").write_text("53.17 13.00 2.10\n")   # as if that start had cost $2.10
    out2, _ = vol.run(AZ_TRAIN_FROM="fav2")
    assert "earlier starts of this same line spent $2." in out2 and "already spent $55." in out2
    # nothing made before is made again; the training carries on from the best of its own saves
    assert "3. carrying on from the best save, step 1000 (the newest)" in out2
    assert "4. training fxreal7 from step 1000 to 2000" in out2 and "2. before the training: rendering" not in out2
    merges = vol.lines("merge.log")
    assert len([m for m in merges if m["scale"] == "1.538462"]) == 1
    assert len([m for m in merges if "/fxreal7/saves/" in m["inputs"][0]]) == 4
    assert len({r["name"] for r in vol.lines("render.log") if r["name"].startswith("fx7") and r["new"]}) == 4
    assert json.loads((vol.ev / "best.json").read_text())["best"]["label"] == "fxreal7 step 1500 at 0.65"
    assert vol.lines("runpodctl.log") == ["stop pod testpod", "stop pod testpod"]
    _no_two_scorers_at_once(vol.lines("score.log"))


def test_a_wrong_favourite_2_file_stops_before_anything_is_spent(vol):
    p = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=120,
                       env={**vol.env, "AZ_TRAIN_FROM": "fav2", "AZ_FAV2_SHA": "8a2f5635"})
    assert p.returncode != 0 and "WRONG_FAVOURITE2" in p.stdout and "expected 8a2f5635" in p.stdout
    assert not vol.lines("render.log") and not vol.lines("train_env.log")
    assert vol.lines("runpodctl.log") == ["stop pod testpod"]              # it still stops its Pod


def test_the_older_modes_still_run_to_the_end(vol):
    """best_blend scores its blends in the main script: the path that hung for 40 minutes on 2026-11-17."""
    out, secs = vol.run(AZ_TRAIN_FROM="best_blend", AZ_BLENDS="75", AZ_MAX_STEPS="500")
    assert "FX6B_DONE" in out and secs < 200
    assert "2. scoring the blends before the training" in out and "2. training starts from the best blend" in out
    env = vol.lines("train_env.log")[0]
    # as trained before: from the blend, with the earlier-frame input, at half the fxreal6 rate, judged as trained
    assert "AZ_EXP=fxreal6c " in env and "AZ_ANCHOR_PROB=1.0 " in env and "AZ_LR=2.5e-5 " in env and "/blends/b75 " in env
    assert not [m for m in vol.lines("merge.log") if "/saves/" in m["inputs"][0]]
    assert any(n.startswith("fx6c500x") and not n.endswith(f"s65_{TAG}") for n in vol.scored())
    assert "CHANCE" not in out and not any(r["name"].startswith("fav2seed") for r in vol.lines("render.log"))
    _no_two_scorers_at_once(vol.lines("score.log"))


def test_a_trainer_that_crashes_still_ends_with_the_checks_and_a_stopped_pod(vol):
    out, secs = vol.run(AZ_TRAIN_FROM="fav2", STUB_TRAIN_FAIL="1")
    assert "FX6B_DONE" in out and secs < 120
    assert "training ended (exit 9), newest save: step none" in out and "made-up trainer crash" in out
    # the checks that need no training are all there, and the pick is made from them
    names = set(vol.scored())
    assert {f"fav2seed53_{TAG}", f"fx6s5800s65_{TAG}", f"fx6s5800s80_{TAG}", f"fx6s5800_{TAG}"} == names
    assert json.loads((vol.ev / "best.json").read_text())["best"]["label"] == "fxreal6 step 5800 at 0.65"
    assert vol.lines("runpodctl.log") == ["stop pod testpod"]


def test_a_model_that_cannot_be_scored_is_left_out_after_two_tries(vol):
    """It used to be tried again until the money ran out."""
    out, secs = vol.run(AZ_TRAIN_FROM="fav2", AZ_MAX_STEPS="1000", STUB_FAIL_SCORE="fx6s5800s80")
    assert "FX6B_DONE" in out and secs < 150
    assert "SCORING_FAILED for s5800s80" in out and "NO_SCORES for s5800s80 after 2 tries: left out" in out
    assert (vol.ev / "s5800s80/.failed").exists() and "NOT CHECKED" not in out
    names = set(vol.scored())
    assert f"fx6s5800s80_{TAG}" not in names and f"fx6s5800s65_{TAG}" in names
    assert len([n for n in names if n.startswith("fx7")]) == 2
    assert vol.lines("runpodctl.log") == ["stop pod testpod"]


def test_a_model_that_cannot_be_rendered_is_left_out_after_three_tries(vol):
    out, secs = vol.run(AZ_TRAIN_FROM="fav2", AZ_MAX_STEPS="500", STUB_FAIL_RENDER="fx6s5800s65")
    assert "FX6B_DONE" in out and secs < 150
    assert "NO_RENDERS for s5800s65 after 3 tries: left out" in out and "NOT CHECKED" not in out
    tries = [r for r in vol.lines("render_try.log") if r["name"].startswith("fx6s5800s65")]
    assert len(tries) == 3 * len(SETS)
    names = set(vol.scored())
    assert f"fx6s5800s65_{TAG}" not in names and f"fx6s5800s80_{TAG}" in names and any(n.startswith("fx7500x") for n in names)
    assert vol.lines("runpodctl.log") == ["stop pod testpod"]


def test_too_little_money_stops_before_anything_starts(vol):
    p = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=120,
                       env={**vol.env, "AZ_TRAIN_FROM": "fav2", "AZ_BALANCE": "4.00"})
    assert p.returncode != 0 and "STOPPED: only 53 minutes of money left" in p.stdout
    assert not vol.lines("render_try.log") and not vol.lines("train_env.log")
    assert vol.lines("runpodctl.log") == ["stop pod testpod"]


def test_a_wrong_training_start_is_caught_before_any_gpu_time(vol):
    """The run measures its own start: anything but favourite #2 x 1/0.65 and it does the checks only."""
    out, secs = vol.run(AZ_TRAIN_FROM="fav2", STUB_BAD_UNSCALE="1")
    assert "FX6B_DONE" in out and secs < 120
    assert "GEOMETRY fxreal7 start (favourite #2 un-scaled): strength x1.20" in out
    assert "UNSCALE_WRONG: the training start is not favourite #2 x 1/0.65" in out
    assert out.index("UNSCALE_WRONG") < out.index("2. before the training: rendering")
    assert "4. no training (see above): only the checks and the final pick" in out and not vol.lines("train_env.log")
    assert f"fx6s5800s65_{TAG}" in vol.scored() and vol.lines("runpodctl.log") == ["stop pod testpod"]


# ---------------------------------------------------------------- fxreal8 (owner, 2026-11-18)
def _fx8_env(vol):
    return {"AZ_TRAIN_FROM": "fav2_addon", "AZ_SPENT": "61.42", "AZ_CAP": "75", "AZ_BALANCE": "18.70"}


def test_fxreal8_practice_run(vol):
    out, secs = vol.run(**_fx8_env(vol))
    root = vol.root
    assert "FX6B_DONE" in out and secs < 200, out[-2000:]
    assert vol.lines("runpodctl.log") == ["stop pod testpod"]
    # order: favourite #2 checked, its strength measured, the run-time test rendered, then the training
    order = ["1. favourite #2 confirmed", "2. the training start is favourite #2 at full strength (x 1.54)",
             "2. before the training: rendering rt30 rt50 aa53", f"render dress10/fav2rt50_{TAG}",
             "4. training fxreal8 from step 0 to 1000 at learning rate 1e-4, rank 16, as an add-on over favourite #2",
             "FX6B trainer: FX8_SAFETY_OK", "5. finishing the checks", "RUNTIME favourite #2 + run-time steadiness 0.30",
             "BEST_SAVE", "VERDICT", "FX6B_DONE"]
    at = [out.find(x) for x in order]
    assert all(i >= 0 for i in at) and at == sorted(at), list(zip(order, at))
    # the trainer: no starting LoRA, favourite #2 un-scaled frozen underneath, the new terms on, the fxreal8 settings
    env = vol.lines("train_env.log")
    assert len(env) == 1
    unscaled = f"{root}/outputs/fluxrt_lora/blend05_unscaled"
    for x in ("AZ_EXP=fxreal8 ", "AZ_MAX_STEPS=1000 ", "AZ_LR=1e-4 ", "AZ_LORA_RANK=16 ", "AZ_INIT_LORA= ", "AZ_ANCHOR_PROB=0 ",
              "AZ_CKPT_INTERVAL=250 ", f"AZ_FROZEN_LORA={unscaled} ", "AZ_STEADY_RATIO=0.25 ", "AZ_REF_WEIGHT=1.0 ",
              "AZ_SAFETY_STEPS=20 ", "KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B "):
        assert x in env[0], (x, env[0])
    assert "FX6B trainer: FROZEN_LORA merged" in out and "FX6B trainer: STEADY_WEIGHT 2.5" in out and "steady=0.02" in out
    # the run-time test: favourite #2's own file, with the filter setting, never the pick
    renders = vol.lines("render.log")
    rt = [r for r in renders if r["name"].startswith("fav2rt")]
    assert {(r["name"], r["steady"]) for r in rt} == {(f"fav2rt30_{TAG}", "0.30"), (f"fav2rt50_{TAG}", "0.50")}
    assert {r["steady"] for r in renders if not r["name"].startswith("fav2rt")} == {None}
    first_save = min(r["t"] for r in renders if r["name"].startswith("fx8"))
    assert max(r["t"] for r in rt) < first_save
    assert "RUNTIME favourite #2 + run-time steadiness 0.50: score +" in out and "worth a live test" in out
    # every save is ONE file: favourite #2 un-scaled + the add-on, turned down to 0.65
    from safetensors.numpy import load_file
    merges = [m for m in vol.lines("merge.log") if "/fxreal8/saves/" in m["inputs"][-1]]
    assert len(merges) == 4 and all(m["inputs"][0] == f"{unscaled}/pytorch_lora_weights.safetensors" and
                                    m["weights"] == ["1.0", "1.0"] and m["scale"] == "0.65" for m in merges)
    fav = load_file(str(root / "outputs/fluxrt_lora/blend05s065/pytorch_lora_weights.safetensors"))
    best = json.loads((vol.ev / "best.json").read_text())
    assert best["best"]["label"] == "fxreal8 step 750 at 0.65"
    kept = load_file(str(root / "outputs/fluxrt_lora/fxreal8/best/pytorch_lora_weights.safetensors"))
    addon = load_file(next(str(p) for p in (root / "outputs/fluxrt_lora/fxreal8/saves").glob("step750_*.safetensors")))
    for k in fav:
        # rank = favourite #2's 4 + the add-on's 2; favourite #2's part is favourite #2 exactly (1/0.65 x 0.65)
        r = fav[k].shape[0] if ".lora_A." in k else fav[k].shape[1]
        if ".lora_A." in k:
            assert kept[k].shape[0] == r + 2 and np.allclose(kept[k][:r], fav[k]) and np.allclose(kept[k][r:], addon[k])
        else:
            assert kept[k].shape[1] == r + 2 and np.allclose(kept[k][:, :r], fav[k], rtol=1e-5, atol=1e-7)
            assert np.allclose(kept[k][:, r:], addon[k] * 0.65, rtol=1e-5, atol=1e-7)
    assert json.loads((root / "outputs/fluxrt_lora/fxreal8/best/selection.json").read_text())["strength_baked_in"] == 0.65
    # the videos: favourite #2, the best run-time test, this run's best save
    videos = {v["set"]: v["panels"] for v in vol.lines("video.log")}
    assert videos["dress10"][0] == f"fav2_{TAG}=favourite #2 (today)"
    assert f"fav2rt50_{TAG}=favourite #2 + run-time steadiness 0.50" in videos["dress10"]
    assert any(p.endswith("=fxreal8 step 750 at 0.65") for p in videos["dress10"]) and len(videos["dress10"]) == 3
    for x in SETS:
        assert (vol.ev / f"fx8_{x}_best.mp4").exists()
    _no_two_scorers_at_once(vol.lines("score.log"))


def test_fxreal8_safety_check_failure_stops_the_training_but_not_the_checks(vol):
    out, secs = vol.run(**_fx8_env(vol), STUB_FX8_FAIL="1")
    assert "FX6B_DONE" in out and secs < 150
    assert "FX6B trainer: FX8_SAFETY_FAILED: the add-on does not start at the frozen model" in out
    assert "4. the safety check stopped the training before it could spend more" in out
    assert not list((vol.root / "outputs/fluxrt_lora/fxreal8").glob("checkpoint-*"))
    # the run-time test and the chance run were still checked, and the Pod stopped once
    assert {f"fav2rt30_{TAG}", f"fav2rt50_{TAG}", f"fav2seed53_{TAG}"} <= set(vol.scored())
    assert vol.lines("runpodctl.log") == ["stop pod testpod"]


def test_fxreal8_restart_carries_on_from_its_best_save(vol):
    out1, _ = vol.run(**_fx8_env(vol), AZ_MAX_STEPS="500")
    assert "4. training fxreal8 from step 0 to 500" in out1 and "FX6B_DONE" in out1
    out2, _ = vol.run(**_fx8_env(vol))
    assert "3. carrying on from the best save, step 500 (the newest)" in out2
    assert "4. training fxreal8 from step 500 to 1000" in out2 and "2. before the training: rendering" not in out2
    # nothing made twice
    assert len({r["name"] for r in vol.lines("render.log") if r["name"].startswith("fav2rt") and r["new"]}) == 2
    merges = [m for m in vol.lines("merge.log") if "/fxreal8/saves/" in m["inputs"][-1]]
    assert len(merges) == 4 and len({m["out"] for m in merges}) == 4
    assert json.loads((vol.ev / "best.json").read_text())["best"]["label"] == "fxreal8 step 750 at 0.65"
    _no_two_scorers_at_once(vol.lines("score.log"))


# ---------------------------------------------------------------- by garment type, more steps (owner, 2026-11-21)
def test_fxreal8_carries_on_from_its_newest_save_and_reports_each_garment_type(vol):
    out1, _ = vol.run(**_fx8_env(vol), AZ_MAX_STEPS="500")
    assert "4. training fxreal8 from step 0 to 500" in out1 and "FX6B_DONE" in out1
    # the second start: more money, more steps, from the NEWEST save, two more chance seeds
    out2, secs = vol.run(**{**_fx8_env(vol), "AZ_SPENT": "68.34", "AZ_CAP": "100", "AZ_BALANCE": "24.29"},
                         AZ_MAX_STEPS="1500", AZ_SAVE_EVERY="500", AZ_RESUME_FROM="newest", AZ_AA_SEEDS="53 54 55")
    assert "FX6B_DONE" in out2 and secs < 200, out2[-2000:]
    assert vol.lines("runpodctl.log") == ["stop pod testpod", "stop pod testpod"]
    order = ["1b. by garment type, from the checks already on the volume", "GARMENT fxreal8 step 500 at 0.65: dresses",
             "2. before the training: rendering aa55 aa54", "3. carrying on from the newest save",
             "4. training fxreal8 from step 500 to 1500", "5. finishing the checks", "BEST_SAVE",
             "GARMENT_VERDICT", "video: ", "FX6B_DONE"]
    at = [out2.find(x) for x in order]
    assert all(i >= 0 for i in at) and at == sorted(at), list(zip(order, at))
    assert out2.count("FX6B GARMENT_CHANCE differences up to these sizes mean nothing: dresses") == 2   # at the start and the end
    env = vol.lines("train_env.log")
    assert len(env) == 2 and "AZ_MAX_STEPS=1500 " in env[1] and "AZ_CKPT_INTERVAL=500 " in env[1]
    # nothing set aside: the newest save is where the training carried on
    assert not (vol.root / "outputs/fluxrt_lora/fxreal8/set_aside").exists()
    assert {p.name for p in (vol.root / "outputs/fluxrt_lora/fxreal8").glob("checkpoint-*")} == \
        {"checkpoint-250", "checkpoint-500", "checkpoint-1000", "checkpoint-1500"}
    # three chance runs: the first is reused, the two new ones are rendered with their seeds and scored
    seeds = {r["name"]: r["seed"] for r in vol.lines("render.log") if r["name"].startswith("fav2seed") and r["new"]}
    assert seeds == {f"fav2seed53_{TAG}": "53", f"fav2seed54_{TAG}": "54", f"fav2seed55_{TAG}": "55"}
    assert {f"fav2seed{n}_{TAG}" for n in (53, 54, 55)} <= set(vol.scored())
    assert "CHANCE favourite #2 with 3 other random seeds: scores" in out2 and "CHANCE_AREAS" in out2
    best = json.loads((vol.ev / "best.json").read_text())
    assert best["chance"]["seeds"] == 3 and set(best["chance"]["garments"]) == {"dresses", "tops", "bottoms"}
    # per garment type: tops are the first clip of fx6test and the second of cases5; bottoms the second of fx6test
    g = {s["label"]: s["garments"] for s in best["saves"]}["fxreal8 step 500 at 0.65"]
    assert g["dresses"]["clips"] == 3 and g["tops"]["clips"] == 2 and g["bottoms"]["clips"] == 1
    # colour is the stand-in's only measure: (12 - x) / 12 per set, sets weighted 1, 1 and 0.5 (the owner's cases)
    assert g["dresses"]["score"] == pytest.approx((0.6 / 12 + 0.5 * 1.0 / 12) / 1.5, abs=1e-4)
    assert g["tops"]["score"] == pytest.approx((1.0 / 12 + 0.5 * 0.2 / 12) / 1.5, abs=1e-4)
    assert g["bottoms"]["score"] == pytest.approx(0.2 / 12, abs=1e-4)
    assert "GARMENT_AREAS " + best["best"]["label"] + ", dresses: colour" in out2
    # one video per garment type, cut from the videos that are there, with the type on screen
    videos = {v["set"]: v for v in vol.lines("video.log")[-6:]}
    for t, n in (("dresses", 3), ("tops", 2), ("bottoms", 1)):
        assert videos[f"g_{t}"]["legend"] is True and videos[f"g_{t}"]["panels"] == videos["dress10"]["panels"]
        assert int(videos[f"g_{t}"]["cols"]) == int(videos["dress10"]["cols"]) + 1
        assert (vol.ev / f"fx8_{t}_best.mp4").exists()
        items = json.loads((vol.root / f"eval/fxa/g_{t}.json").read_text())
        assert len(items) == n and all((vol.root / "eval/fxa/results" / f"g_{t}" / it["key"] / "input.mp4").exists() for it in items)
    assert videos["dress10"]["legend"] is False
    assert "GARMENT_SETS dresses: 3 clips from cases5, dress10 | tops: 2 clips from cases5, fx6test | bottoms: 1 clips from fx6test" in out2
    _no_two_scorers_at_once(vol.lines("score.log"))


# ---------------------------------------------------------------- fxreal9: two arms from fxreal8's step 1000
ARMS = ROOT / "training/fluxrt_lora/fx9_arms.sh"


def _arms(vol, timeout=400, **extra):
    env = {**vol.env, "AZ_SPENT": "79.50", "AZ_CAP": "100", "AZ_BALANCE": "43.00", "AZ_CONTINUE_SCRIPT": str(SCRIPT), **extra}
    t0 = time.time()
    p = subprocess.run(["bash", str(ARMS)], capture_output=True, text=True, env=env, timeout=timeout)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    return p.stdout + p.stderr, time.time() - t0


def test_fxreal9_two_arms_start_from_fxreal8_step_1000_and_are_compared_at_equal_steps(vol):
    root = vol.root
    # stage 1 as it will be on the volume: fxreal8 to step 2000, saves at 250..1000, 1500, 2000
    vol.run(**_fx8_env(vol))
    vol.run(**{**_fx8_env(vol), "AZ_SPENT": "68.34", "AZ_CAP": "100", "AZ_BALANCE": "24.29"}, AZ_MAX_STEPS="2000",
            AZ_SAVE_EVERY="500", AZ_RESUME_FROM="newest", AZ_AA_SEEDS="53 54 55")
    fx8 = root / "outputs/fluxrt_lora/fxreal8"
    before = {p.name: p.read_bytes() for p in (fx8 / "checkpoint-1000").iterdir()}
    n_env = len(vol.lines("train_env.log"))
    out, secs = _arms(vol)
    assert "FX9_DONE" in out and secs < 300, out[-2500:]
    # one Pod: the first arm leaves it running, the second stops it (and stage 1 stopped its own twice)
    assert vol.lines("runpodctl.log") == ["stop pod testpod"] * 3
    assert out.count("== AZ_NO_STOP=1: the Pod keeps running: the next arm starts now") == 1 and "stop it by hand" not in out
    order = ["FX9 ", "arm 1 of 2: fxreal9a (protect) from fxreal8:1000 to step 2000",
             "1. fxreal9a starts from fxreal8 step 1000 (its save and optimiser state copied; fxreal8 itself is not changed)",
             "1b. by garment type", "GARMENT fxreal8 step 1000 at 0.65",
             "4. training fxreal9a from step 1000 to 2000",
             "favourite #2's fabric contrast kept (x1.0)", "FX6B_DONE", "arm fxreal9a ended (exit 0)",
             "arm 2 of 2: fxreal9b (reach) from fxreal8:1000 to step 2000", "1. fxreal9b starts from fxreal8 step 1000",
             "4. training fxreal9b from step 1000 to 2000", "arm fxreal9b ended (exit 0)", "FX9_DONE"]
    at = []
    pos = 0
    for x in order:
        pos = out.find(x, pos)
        at.append(pos)
        assert pos >= 0, (x, out[-3000:])
    # the trainer: each arm resumes fxreal8's step 1000 in its own folder, with its own terms and the same recipe
    env = vol.lines("train_env.log")[n_env:]
    assert len(env) == 2
    for e, exp, extra in ((env[0], "fxreal9a", "AZ_PAIR_FLOOR=1.0 AZ_DETAIL_FLOOR= AZ_TEXTURE_FLOOR= AZ_KEEP_SPREAD=1.0 "),
                          (env[1], "fxreal9b", "AZ_PAIR_FLOOR=1.0 AZ_DETAIL_FLOOR=0.5 AZ_TEXTURE_FLOOR=0.5 AZ_KEEP_SPREAD=1.0 ")):
        for x in (f"AZ_EXP={exp} ", "AZ_MAX_STEPS=2000 ", "AZ_CKPT_INTERVAL=500 ", "AZ_LR=1e-4 ", "AZ_LORA_RANK=16 ", "AZ_INIT_LORA= ",
                  "AZ_STEADY_RATIO=0.25 ", "AZ_REF_WEIGHT=1.0 ", extra):
            assert x in e, (x, e)
    assert "FX6B trainer: FX9_TERMS same-clip colour floor 1.0, detail floor 0.5, fabric-contrast floor 0.5" in out
    assert "FX6B trainer: STEADY_WEIGHT 2.5 (from the first start of this run)" in out and "keep=0.004" in out
    # fxreal8 is untouched; each arm holds a copy of its step 1000 and its own later saves
    assert {p.name: p.read_bytes() for p in (fx8 / "checkpoint-1000").iterdir()} == before
    assert {p.name for p in fx8.glob("checkpoint-*")} == {f"checkpoint-{n}" for n in (250, 500, 750, 1000, 1500, 2000)}
    for exp in ("fxreal9a", "fxreal9b"):
        d = root / "outputs/fluxrt_lora" / exp
        assert {p.name for p in d.glob("checkpoint-*")} == {"checkpoint-1000", "checkpoint-1500", "checkpoint-2000"}
        assert (d / "checkpoint-1000/optimizer.bin").read_text() == "optimiser state at step 1000 of fxreal8\n"
        assert (d / "branch_from.txt").read_text().strip() == "fxreal8:1000" and (d / "fx8_steady_weight.json").exists()
        assert not list(d.glob(".branch_tmp*")) and not (d / "set_aside").exists()
    # the branch point is not checked again: two new saves per arm, nothing else rendered
    scored = set(vol.scored())
    for pfx in ("fx9a", "fx9b"):
        mine = sorted(n for n in scored if n.startswith(pfx))
        assert len(mine) == 2 and mine[0].startswith(f"{pfx}1500x") and mine[1].startswith(f"{pfx}2000x"), mine
    new = [r["name"] for r in vol.lines("render.log") if r["new"] and r["name"].startswith("fx9")]
    assert len(new) == 4 * len(SETS) and not any(n.startswith(("fx9a1000", "fx9b1000")) for n in new)
    # each save is favourite #2 un-scaled + that arm's add-on, turned down to 0.65
    merges = [m for m in vol.lines("merge.log") if "/fxreal9" in m["inputs"][-1]]
    assert len(merges) == 4 and all(m["weights"] == ["1.0", "1.0"] and m["scale"] == "0.65" and
                                    m["inputs"][0].endswith("blend05_unscaled/pytorch_lora_weights.safetensors") for m in merges)
    # side by side at equal steps, and the fabric direction of every newly scored save
    for step in (1500, 2000):
        for run in ("fxreal8", "fxreal9a", "fxreal9b"):
            assert f"FX6B COMPARE step {step}, {run}: score " in out, (step, run)
    assert "FX6B FABRIC fxreal9b step 2000 at 0.65: the garment's light-dark spread is x0.82 of the photo's (favourite #2 x0.90" in out
    best = json.loads((vol.ev / "best.json").read_text())
    assert best["best"]["label"] == "fxreal9b step 2000 at 0.65"          # the made-up best colour
    assert (root / "outputs/fluxrt_lora/fxreal9b/best/pytorch_lora_weights.safetensors").exists()
    for pfx in ("fx9a", "fx9b"):
        for v in ("dress10", "dresses", "tops", "bottoms"):
            assert (vol.ev / f"{pfx}_{v}_best.mp4").exists(), (pfx, v)
    # the money: both arms are recorded under the same AZ_SPENT and AZ_BALANCE, so the second one starts with what
    # the first one spent already counted (the adding-up itself: the restart test of fxreal7)
    runs = [x.read_text().split() for x in (vol.ev / "spend").glob("*.run")]
    assert len([r for r in runs if r[:2] == ["79.50", "43.00"]]) == 2
    _no_two_scorers_at_once(vol.lines("score.log"))


def test_fxreal9_started_again_makes_nothing_twice(vol):
    vol.run(**_fx8_env(vol))
    out1, _ = _arms(vol, AZ_ARMS="a", AZ_MAX_STEPS="1500")
    assert "4. training fxreal9a from step 1000 to 1500" in out1 and "FX9_DONE" in out1
    out2, _ = _arms(vol, AZ_ARMS="a", AZ_MAX_STEPS="2000")
    assert "1. fxreal9a carries on (it started from fxreal8 step 1000)" in out2
    assert "4. training fxreal9a from step 1500 to 2000" in out2 and "its save and optimiser state copied" not in out2
    names = [r["name"] for r in vol.lines("render.log") if r["new"] and r["name"].startswith("fx9a")]
    assert len(names) == 2 * len(SETS) and len(set(names)) == 2
    d = vol.root / "outputs/fluxrt_lora/fxreal9a"
    assert {p.name for p in d.glob("checkpoint-*")} == {"checkpoint-1000", "checkpoint-1500", "checkpoint-2000"}


def test_a_branch_from_a_save_that_is_not_there_stops_before_anything_is_spent(vol):
    p = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=120,
                       env={**vol.env, **_fx8_env(vol), "AZ_EXP": "fxreal9a", "AZ_BRANCH_FROM": "fxreal8:1000"})
    out = p.stdout + p.stderr
    assert p.returncode != 0 and "STOPPED: NO_BRANCH_POINT" in out and not vol.lines("train_env.log")
    assert vol.lines("runpodctl.log") == ["stop pod testpod"]
    for bad in ("fxreal8", "fxreal8:x", "fxreal9a:1000"):
        p = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=120,
                           env={**vol.env, **_fx8_env(vol), "AZ_EXP": "fxreal9a", "AZ_BRANCH_FROM": bad})
        assert p.returncode != 0 and "BAD_AZ_BRANCH_FROM" in p.stdout + p.stderr, bad

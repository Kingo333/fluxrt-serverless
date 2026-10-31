"""Check JoyAI WebDataset shards before training.

  python3 -m data_tools.validate shard.tar [...] [--catalog train.json --catalog test.json]

Per sample: files match the task, JSON has a caption and metadata.ref_image when a
garment photo is present, images decode and are big enough, the garment photo is not
identical to the target, videos are non-empty (and decodable when ffprobe exists),
consent is present for own recordings. Across catalogs: no garment or person
appears in both train and test (leakage).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

from . import sources
from .shards import TASK_FILES

MIN_SIDE = 256


def _group_members(tar: tarfile.TarFile) -> dict[str, dict[str, bytes]]:
    samples: dict[str, dict[str, bytes]] = defaultdict(dict)
    for m in tar.getmembers():
        if not m.isfile():
            continue
        key, _, ext = m.name.partition(".")
        samples[key][ext] = tar.extractfile(m).read()
    return samples


def _check_image(name: str, data: bytes) -> str | None:
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if min(im.size) < MIN_SIDE:
                return f"{name} too small {im.size}"
    except Exception as e:  # noqa: BLE001
        return f"{name} does not decode ({type(e).__name__})"
    return None


def _check_video(name: str, data: bytes) -> str | None:
    if len(data) < 1024:
        return f"{name} is empty or tiny ({len(data)} bytes)"
    if shutil.which("ffprobe") is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(data)
        f.flush()
        r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", f.name],
                           capture_output=True, text=True)
    if r.returncode != 0:
        return f"{name} does not decode"
    try:
        frames = int(r.stdout.strip().split(",")[0])
    except ValueError:
        return f"{name} frame count unknown"
    return None if frames >= 49 else f"{name} has {frames} frames (< 49, the training clip length)"


def validate_shard(path: str) -> tuple[int, list[str]]:
    problems: list[str] = []
    with tarfile.open(path) as tar:
        samples = _group_members(tar)
    for key, files in samples.items():
        where = f"{Path(path).name}:{key}"
        raw = files.pop("json", None)
        if raw is None:
            problems.append(f"{where}: no json")
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            problems.append(f"{where}: bad json")
            continue
        meta = rec.get("metadata") or {}
        task = meta.get("task")
        if not str(rec.get("caption", "")).strip():
            problems.append(f"{where}: empty caption")
        if task not in TASK_FILES or set(files) not in TASK_FILES[task]:
            problems.append(f"{where}: files {sorted(files)} do not match task {task}")
            continue
        ref = meta.get("ref_image")
        if task in ("r2i", "iv2v"):
            if ref not in files:
                problems.append(f"{where}: metadata.ref_image missing or wrong ({ref!r}); the loader would drop the garment photo")
        try:
            src = sources.get(meta.get("source", ""))
            if src.needs_consent and not meta.get("consent_id"):
                problems.append(f"{where}: own recording without consent_id")
        except KeyError as e:
            problems.append(f"{where}: {e}")
        for name, data in files.items():
            err = _check_video(name, data) if name.endswith(".mp4") else _check_image(name, data)
            if err:
                problems.append(f"{where}: {err}")
        target = "2.png" if "2.png" in files else "2.jpg" if "2.jpg" in files else "1.mp4" if task == "iv2v" else None
        if ref and target and ref in files and target in files and not target.endswith(".mp4"):
            if hashlib.sha256(files[ref]).digest() == hashlib.sha256(files[target]).digest():
                problems.append(f"{where}: garment photo identical to target")
    return len(samples), problems


def check_leakage(catalog_paths: list[str]) -> list[str]:
    by_split: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"garment": set(), "person": set()})
    for p in catalog_paths:
        for e in json.loads(Path(p).read_text()):
            by_split[e["split"]]["garment"].update(e.get("garment_ids", []))
            by_split[e["split"]]["person"].update(e.get("person_ids", []))
    problems = []
    for kind in ("garment", "person"):
        overlap = by_split["train"][kind] & by_split["test"][kind]
        if overlap:
            problems.append(f"{kind} leakage between train and test: {sorted(overlap)[:10]} ({len(overlap)} total)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="*")
    ap.add_argument("--catalog", action="append", default=[])
    a = ap.parse_args()
    total, problems = 0, []
    for s in a.shards:
        n, p = validate_shard(s)
        total += n
        problems += p
    problems += check_leakage(a.catalog)
    for p in problems[:200]:
        print("PROBLEM", p)
    print(f"checked {total} samples in {len(a.shards)} shards: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

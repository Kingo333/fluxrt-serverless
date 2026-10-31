"""Write samples as WebDataset TAR shards in the layout JoyAI's loader expects
(training/src/dataset/webdataset.py, ToyData README):

  r2i  (reference image edit): 0.png garment, 1.png source, 2.png target
  iv2v (reference video edit): 0.jpg garment, 0.mp4 source, 1.mp4 target
  i2i  / v2v (no reference):   0.* source, 1.* target

The JSON must carry metadata.ref_image when a garment photo is included, or the
loader drops the reference for video samples. We always write:
  {"caption": ..., "metadata": {"ref_image": "0.jpg"|"0.png"|None, "source": ..., ...}}
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import sources

TASK_FILES = {
    "r2i": ({"0.png", "1.png", "2.png"}, {"0.jpg", "1.jpg", "2.jpg"}),
    "i2i": ({"0.png", "1.png"}, {"0.jpg", "1.jpg"}),
    "iv2v": ({"0.jpg", "0.mp4", "1.mp4"},),
    "v2v": ({"0.mp4", "1.mp4"},),
}
REF_KEY = {"r2i": ("0.png", "0.jpg"), "iv2v": ("0.jpg",)}
MANIFEST_GROUP = {"r2i": "i2i", "i2i": "i2i", "iv2v": "v2v", "v2v": "v2v"}


@dataclass
class Sample:
    key: str                      # unique, no dots or slashes
    task: str                     # r2i | i2i | iv2v | v2v
    files: dict[str, bytes]       # e.g. {"0.png": ..., "1.png": ..., "2.png": ...}
    caption: str
    source: str                   # key in sources.SOURCES
    split: str = "train"          # train | val | test
    garment_id: str | None = None
    person_id: str | None = None
    consent_id: str | None = None
    extra: dict = field(default_factory=dict)
    # Extra caption fields, e.g. {"caption_detailed": ...}; the training config samples
    # between "caption" and these (missing ones are skipped by the upstream loader).
    captions: dict = field(default_factory=dict)

    def check(self) -> None:
        if "." in self.key or "/" in self.key or not self.key:
            raise ValueError(f"bad key {self.key!r}")
        if self.task not in TASK_FILES:
            raise ValueError(f"unknown task {self.task!r}")
        if set(self.files) not in TASK_FILES[self.task]:
            raise ValueError(f"{self.key}: task {self.task} needs files {TASK_FILES[self.task]}, got {sorted(self.files)}")
        if not self.caption.strip():
            raise ValueError(f"{self.key}: empty caption")
        src = sources.get(self.source)
        if src.needs_consent and not self.consent_id:
            raise ValueError(f"{self.key}: source {self.source} requires a consent_id")
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"{self.key}: bad split {self.split!r}")

    def ref_name(self) -> str | None:
        for name in REF_KEY.get(self.task, ()):
            if name in self.files:
                return name
        return None

    def json_record(self) -> dict:
        meta = {
            "ref_image": self.ref_name(),
            "task": self.task,
            "source": self.source,
            "split": self.split,
            "garment_id": self.garment_id,
            "person_id": self.person_id,
            "consent_id": self.consent_id,
            **self.extra,
        }
        return {"caption": self.caption, **{k: v for k, v in self.captions.items() if v}, "metadata": meta}


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


def write_shards(samples: Iterable[Sample], out_dir: str | Path, prefix: str, per_shard: int = 500) -> list[dict]:
    """Write samples into prefix-000000.tar, ... Returns one catalog entry per shard."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    tar = None
    current: dict | None = None
    seen: set[str] = set()

    def close() -> None:
        nonlocal tar, current
        if tar is not None:
            tar.close()
            entries.append(current)
        tar, current = None, None

    for s in samples:
        s.check()
        if s.key in seen:
            raise ValueError(f"duplicate key {s.key}")
        seen.add(s.key)
        if tar is None or current["num_samples"] >= per_shard:
            close()
            path = out / f"{prefix}-{len(entries):06d}.tar"
            tar = tarfile.open(path, "w")
            current = {"path": str(path), "task": s.task, "group": MANIFEST_GROUP[s.task],
                       "source": s.source, "split": s.split, "num_samples": 0,
                       "garment_ids": [], "person_ids": []}
        if (s.task, s.source, s.split) != (current["task"], current["source"], current["split"]):
            raise ValueError("one shard per task/source/split: write them with separate write_shards calls")
        for name, data in sorted(s.files.items()):
            _add(tar, f"{s.key}.{name}", data)
        _add(tar, f"{s.key}.json", json.dumps(s.json_record(), ensure_ascii=False).encode())
        current["num_samples"] += 1
        if s.garment_id:
            current["garment_ids"].append(s.garment_id)
        if s.person_id:
            current["person_ids"].append(s.person_id)
    close()
    return entries

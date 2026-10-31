"""Staging area between raw datasets and shards.

A staged pair is one folder: ref.(png|jpg) garment photo, target.(png|jpg|mp4) the
real person wearing it, optional mask.png / densepose.png (offline only), meta.json,
and, once generated, source.(png|mp4): the same person in a *different* garment.

Why pseudo-sources (plan G1): JoyAI edits an unmasked input frame. Training on
"generated other-garment source -> real target" keeps training input identical to
live input (a plain camera frame), so the worker needs no human parser at runtime,
and the model always learns to produce a *real* target.
"""

from __future__ import annotations

import json
from pathlib import Path

CATEGORY_INSTRUCTION = {
    "upper": "Substitute the current top with the garment in the reference image.",
    "lower": "Substitute the current pants or skirt with the garment in the reference image.",
    "dress": "Substitute the current outfit with the dress in the reference image.",
}
KEEP = "Keep the person's face, body, pose and the background unchanged."


def instruction(category: str) -> str:
    """Same wording the Azyah app sends at inference (buildTryOnPrompt.ts), so training matches live use."""
    return f"{CATEGORY_INSTRUCTION[category]} {KEEP}"


def stage_pair(root: Path, key: str, files: dict[str, bytes], meta: dict) -> Path:
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (d / name).write_bytes(data)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def find(d: Path, stem: str) -> Path | None:
    for ext in ("png", "jpg", "mp4"):
        p = d / f"{stem}.{ext}"
        if p.exists():
            return p
    return None


def iter_pairs(root: Path):
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "meta.json").exists():
            yield d, json.loads((d / "meta.json").read_text())

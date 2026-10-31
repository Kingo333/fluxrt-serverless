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


FIDELITY = "Match its exact color, pattern, print, logo, fabric and construction details."


def clean_product_text(raw: str | None, max_chars: int = 320) -> str:
    """Python twin of cleanProductText() in the app's buildTryOnPrompt.ts."""
    import re
    if not raw:
        return ""
    s = re.sub(r"<[^>]*>", " ", raw)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        cut = s[:max_chars]
        last_space = cut.rfind(" ")
        s = cut[:last_space] if last_space > max_chars * 0.6 else cut
        s = re.sub(r"[\s,;:.-]+$", "", s)
    return s


def instruction(category: str, name: str | None = None, description: str | None = None) -> str:
    """Exact twin of buildTryOnInstruction() in the app (azyah-shopping
    src/components/ai-studio/live-cam/buildTryOnPrompt.ts), so training captions
    match what the live app sends (review R2). Without name/description it is the
    short form; with them, the detailed form."""
    import re
    parts = [CATEGORY_INSTRUCTION[category]]
    n, d = clean_product_text(name, 120), clean_product_text(description)
    if n or d:
        label = f"The reference garment is the {n}" if n else "The reference garment"
        parts.append(f"{label}: {re.sub(r'[.!]?$', '.', d, count=1)}" if d else f"{label}.")
        parts.append(FIDELITY)
    parts.append(KEEP)
    return " ".join(parts)


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

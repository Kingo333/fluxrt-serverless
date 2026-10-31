"""Every dataset we may train on, with its licence terms. The manifest builder
reads this, so a research-only source can never end up in a commercial run."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    licence: str
    commercial_ok: bool
    attribution: str | None = None
    needs_consent: bool = False
    note: str = ""


SOURCES: dict[str, Source] = {
    "glance": Source(
        name="glance",
        licence="CC-BY-4.0",
        commercial_ok=True,
        attribution=(
            "Glance VTON Synthetic Dataset (2026), "
            "https://huggingface.co/datasets/glanceai/VTON-Synthetic-Pairs, CC BY 4.0"
        ),
        note="Synthetic stills, front-facing, grey background: garment fidelity, not motion.",
    ),
    "vivid": Source(
        name="vivid",
        licence="Apache-2.0 tag on Hugging Face; paper says 'available for research purposes' (Net-A-Porter footage)",
        commercial_ok=False,
        note="R&D only until legal sign-off. Never in a commercial checkpoint.",
    ),
    "own": Source(
        name="own",
        licence="Azyah-owned recordings under signed consent",
        commercial_ok=True,
        needs_consent=True,
        note="Phone/webcam half-body clips. Each sample must carry a consent_id with commercial use agreed.",
    ),
    "own_research": Source(
        name="own_research",
        licence="Azyah-owned recordings; consent excludes commercial use",
        commercial_ok=False,
        needs_consent=True,
        note="Own clips whose consent said commercial=no: R&D runs only.",
    ),
    "toy": Source(
        name="toy",
        licence="JoyAI ToyData (smoke tests only)",
        commercial_ok=False,
        note="Data-loading checks only.",
    ),
}


def get(name: str) -> Source:
    if name not in SOURCES:
        raise KeyError(f"unknown data source {name!r}; add it to training/data_tools/sources.py with its licence")
    return SOURCES[name]

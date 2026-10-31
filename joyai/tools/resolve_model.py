#!/usr/bin/env python3
"""Pick the DiT checkpoint for a named model version.

  python3 resolve_model.py <version> <registry.json> <default_dit>
  -> prints the checkpoint path, or exits non-zero with a reason

Versions keep experiments away from the live model:
  joyai-baseline           the deployed upstream model (default; needs no registry)
  azyah-vton-experimental  any trained checkpoint under test
  azyah-vton-candidate     one that passed evaluation, waiting for production approval

registry.json (on the volume, e.g. /runpod-volume/joyai/models/registry.json):
  {"azyah-vton-experimental": {"dit": "models/exp1/dit_merged.pth", "sha256": "...", "notes": "..."}}
Relative paths are resolved against the registry's folder. Fails closed: an unknown
version or a missing file never silently falls back to another model.

Pre-launch gate (Phase K): an entry with "commercial_use_cleared": false (e.g. trained
with ViViD) is refused unless JOYAI_ALLOW_RESEARCH_MODEL=1 is set on the endpoint.
Set it only on staging; production then cannot serve such a model by accident.
"""
import os
import json
import sys
from pathlib import Path

BASELINE = "joyai-baseline"


def resolve(version: str, registry: str, default_dit: str) -> str:
    if version in ("", BASELINE):
        return default_dit
    reg = Path(registry)
    if not reg.is_file():
        raise SystemExit(f"model version {version!r} requested but no registry at {reg}")
    entries = json.loads(reg.read_text())
    if version not in entries:
        raise SystemExit(f"model version {version!r} not in {reg} (has: {', '.join(sorted(entries)) or 'none'})")
    entry = entries[version]
    if entry.get("commercial_use_cleared") is not True and os.environ.get("JOYAI_ALLOW_RESEARCH_MODEL") != "1":
        raise SystemExit(f"model version {version!r} is not cleared for commercial use "
                         f"({entry.get('notice') or 'commercial_use_cleared is not true'}); "
                         "set JOYAI_ALLOW_RESEARCH_MODEL=1 only on a staging endpoint")
    dit = Path(entry["dit"])
    if not dit.is_absolute():
        dit = reg.parent / dit
    if not dit.is_file():
        raise SystemExit(f"model version {version!r}: checkpoint missing at {dit}")
    return str(dit)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    print(resolve(*sys.argv[1:]))

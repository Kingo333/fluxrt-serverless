#!/usr/bin/env python3
"""Write MODEL_CARD.md and a registry entry for a trained checkpoint, from its run's
selection.json (data provenance) and checksums. Every card states the training data;
research-only data (ViViD) is printed at the top and blocks production serving via
"commercial_use_cleared": false in the registry entry.

  python3 model_card.py --selection manifests/research-train/selection.json \
      --checkpoint models/exp1/dit_merged.pth --version azyah-vton-experimental \
      --base stage1:fashion-exp1/step_2000 --notes "Stage1 2000 + Stage2 1000" --out-dir models/exp1
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def build(selection: dict, checkpoint: str, version: str, base: str, notes: str, digest: str) -> tuple[str, dict]:
    cleared = bool(selection.get("commercial_use_cleared"))
    lines = [f"# {version}", ""]
    if not cleared:
        lines += [f"> **Not cleared for commercial use.** {selection.get('notice', '')}",
                  "> Production serving is blocked unless legal sign-off is recorded, or use the commercial-mode (ViViD-free) retrain.", ""]
    lines += [
        f"- Checkpoint: `{checkpoint}` (sha256 `{digest}`)",
        f"- Built from: {base}",
        f"- Data mode: `{selection.get('mode')}`; sources: {', '.join(selection.get('used_sources', [])) or 'none'}",
        f"- Trained with ViViD: **{'yes' if selection.get('trained_with_vivid') else 'no'}**",
        f"- Date: {datetime.now(timezone.utc).date().isoformat()}",
        f"- Notes: {notes}",
        "",
        "## Required notices",
        "- JoyAI-Video-Edit (Apache-2.0), MiMo-VL-7B-RL-2508 (MIT).",
        *[f"- {a}" for a in selection.get("attribution", [])],
    ]
    entry = {"dit": Path(checkpoint).name, "sha256": digest, "notes": notes, "data_mode": selection.get("mode"),
             "used_sources": selection.get("used_sources", []), "trained_with_vivid": bool(selection.get("trained_with_vivid")),
             "commercial_use_cleared": cleared, "notice": selection.get("notice", "")}
    return "\n".join(lines) + "\n", entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    sel = json.loads(Path(a.selection).read_text())
    card, entry = build(sel, a.checkpoint, a.version, a.base, a.notes, sha256(a.checkpoint))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MODEL_CARD.md").write_text(card)
    (out / "registry_entry.json").write_text(json.dumps({a.version: entry}, indent=2))
    print(card)
    print(f"Add {out / 'registry_entry.json'} to models/registry.json (paths relative to the registry folder).")


if __name__ == "__main__":
    main()

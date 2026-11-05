"""Build the FluxRT shoe regression set (owner requirement 2026-11-05: the FluxRT LoRA must not
lose live FluxRT's ability to put shoes on a person).

  python -m eval.make_shoe_set OUT_DIR [--persons DIR] [--vivid-staging DIR] [--set eval/shoe_set.json]

Each item OUT_DIR/<shoe_key>/ gets person.png (a full-body photo with feet visible), ref.<ext>
(the shoe product photo from the app catalogue, downloaded here) and meta.json
{category: "shoes", name, description}; generate_fluxrt --pairs then runs it exactly like the
app (buildTryOnPrompt twin with the shoes clause, item name and description; 288x512; ref padded
to 384x512; 2 steps).

People: --persons DIR (the owner's own consented full-body photos, preferred) or else the middle
frame of held-out ViViD test clips (full-body studio footage; eval only, never training).
Resumable: existing items are kept.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def person_sources(persons: Path | None, vivid: Path | None, n: int) -> list[Path]:
    """Owner photos first (sorted), then ViViD test clips of full-body categories (dress, lower)."""
    out: list[Path] = []
    if persons and persons.is_dir():
        out += sorted(p for p in persons.iterdir() if p.suffix.lower() in IMG_EXT)
    if vivid and vivid.is_dir() and len(out) < n:
        for d in sorted(p for p in vivid.iterdir() if p.is_dir()):
            mp = d / "meta.json"
            if not (mp.exists() and (d / "target.mp4").exists()):
                continue
            meta = json.loads(mp.read_text())
            if meta.get("split") == "test" and meta.get("category") in ("dress", "lower"):
                out.append(d / "target.mp4")
            if len(out) >= n:
                break
    return out[:n]


def middle_frame(mp4: Path):
    import av
    with av.open(str(mp4)) as c:
        frames = [f.to_image().convert("RGB") for _, f in zip(range(97), c.decode(c.streams.video[0]))]
    return frames[(len(frames) - 1) // 2]


def build(out_dir: str, set_json: str, persons: str | None = None, vivid: str | None = None) -> dict:
    from PIL import Image
    items = json.loads(Path(set_json).read_text())["items"]
    srcs = person_sources(Path(persons) if persons else None, Path(vivid) if vivid else None, len(items))
    if not srcs:
        raise SystemExit("no person photos: pass --persons DIR or --vivid-staging DIR")
    out = Path(out_dir)
    stats = {"written": 0, "kept": 0, "failed": []}
    for k, it in enumerate(items):
        d = out / it["key"]
        if (d / "meta.json").exists():
            stats["kept"] += 1
            continue
        d.mkdir(parents=True, exist_ok=True)
        src = srcs[k % len(srcs)]
        person = middle_frame(src) if src.suffix == ".mp4" else Image.open(src).convert("RGB")
        person.save(d / "person.png")
        try:
            req = urllib.request.Request(it["image_url"], headers={"User-Agent": "azyah-eval/1.0"})
            data = urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:  # noqa: BLE001
            stats["failed"].append(f"{it['key']}: {type(e).__name__}: {e}")
            continue
        (d / "ref.jpg").write_bytes(data)
        Image.open(d / "ref.jpg").verify()
        (d / "meta.json").write_text(json.dumps({"category": "shoes", "name": it["name"],
                                                 "description": it["description"], "person_from": str(src)}))
        stats["written"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--set", default=str(Path(__file__).with_name("shoe_set.json")))
    ap.add_argument("--persons")
    ap.add_argument("--vivid-staging")
    a = ap.parse_args()
    print(json.dumps(build(a.out_dir, a.set, a.persons, a.vivid_staging), indent=2))


if __name__ == "__main__":
    main()

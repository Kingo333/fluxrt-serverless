"""Read ViViD straight from ViViD.zip: list it, check the video-to-garment pairing,
and extract only a sampled subset (cost cut C5: about 1-2K clips instead of 9.7K,
and no 35 GB fully extracted copy on the volume).

  python3 -m data_tools.vivid_zip inspect ViViD.zip
  python3 -m data_tools.vivid_zip extract ViViD.zip OUT_DIR --sample 2000 [--seed 7] [--pairing stem]

extract writes OUT_DIR/videos/... and OUT_DIR/cloth/... for the sampled pairs only;
then run `convert_vivid convert OUT_DIR STAGING --pairing stem`.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter
from pathlib import PurePosixPath

from .convert_vivid import CLOTH_DIRS, _pattern, pair_id

IMG = (".jpg", ".jpeg", ".png")


def _index(names: list[str]):
    videos = [n for n in names if n.lower().endswith(".mp4") and "/videos/" in f"/{n}"]
    cloths = [n for n in names if n.lower().endswith(IMG) and any(f"/{d}/" in f"/{n}" for d in CLOTH_DIRS)]
    return videos, cloths


def pairs_by_stem(names: list[str]) -> list[tuple[str, str]]:
    videos, cloths = _index(names)
    # pair within the same parent folder of videos/ and cloth/ (e.g. upper_body/videos, upper_body/cloth)
    by_key = {}
    # ViViD.zip names them N_detail.mp4 and N_in_xl.jpg, so pair on the product number.
    for c in cloths:
        p = PurePosixPath(c)
        by_key[(str(p.parent.parent), pair_id(p.stem))] = c
    out = []
    for v in videos:
        p = PurePosixPath(v)
        c = by_key.get((str(p.parent.parent), pair_id(p.stem)))
        if c:
            out.append((v, c))
    return sorted(out)


def inspect(zip_path: str) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
    videos, cloths = _index(names)
    pairs = pairs_by_stem(names)
    return {
        "files": len(names), "videos": len(videos), "cloth_images": len(cloths),
        "pairs_by_stem": len(pairs),
        "top_folders": Counter("/".join(PurePosixPath(n).parts[:2]) for n in names).most_common(15),
        "video_name_patterns": Counter(_pattern(PurePosixPath(v).name) for v in videos).most_common(5),
        "cloth_name_patterns": Counter(_pattern(PurePosixPath(c).name) for c in cloths).most_common(5),
        "examples": {"videos": videos[:5], "cloth": cloths[:5], "pairs": pairs[:3]},
    }


def extract(zip_path: str, out_dir: str, sample: int, seed: int = 7) -> dict:
    from pathlib import Path
    with zipfile.ZipFile(zip_path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        pairs = pairs_by_stem(names)
        if not pairs:
            raise SystemExit("no video/cloth pairs by stem: run inspect and send the report to Claude Code")
        rng = random.Random(seed)
        chosen = rng.sample(pairs, min(sample, len(pairs)))
        out = Path(out_dir)
        for v, c in chosen:
            for member, kind in ((v, "videos"), (c, "cloth")):
                p = PurePosixPath(member)
                group = p.parent.parent.name or "all"
                dest = out / group / kind / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(member))
    return {"pairs_available": len(pairs), "extracted": len(chosen)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("zip")
    e = sub.add_parser("extract")
    e.add_argument("zip")
    e.add_argument("out_dir")
    e.add_argument("--sample", type=int, required=True)
    e.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(json.dumps(inspect(a.zip) if a.cmd == "inspect" else extract(a.zip, a.out_dir, a.sample, a.seed), indent=2))


if __name__ == "__main__":
    main()

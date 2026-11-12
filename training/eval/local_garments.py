"""Fill OUT/garments/ with local photos for every test garment the shop server would not give us.

  python3 local_garments.py [/workspace/azyah-vton]

select_test_clips downloads each catalogue photo only if OUT/garments/<key>.jpg is missing, so after
this runs it makes no network calls for garments. Stand-ins (same category, never trained on):
  tops / dresses / bottoms -> the garment photo of a held-out ViViD TEST clip (upper / dress / lower)
  shoes                    -> the same shoe from the shoe set if it downloaded, else another shoe-set shoe
Also removes a half-downloaded MediaPipe pose model (under 4 MB) so it is fetched again.
Standard library only.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

CLIP_CATEGORY = {"tops": "upper", "dresses": "dress", "bottoms": "lower"}


def main(root: str = "/workspace/azyah-vton") -> dict:
    r = Path(root)
    e = r / "eval/fx2"
    g_dir = e / "cases/garments"
    g_dir.mkdir(parents=True, exist_ok=True)
    report = {"have": [], "local": {}, "none": []}
    task = e / "cases/pose_landmarker_lite.task"
    if task.exists() and task.stat().st_size < 4_000_000:
        task.unlink()
        report["removed_partial_pose_model"] = True
    pool: dict[str, list[Path]] = {}
    for d in sorted(p for p in (r / "data/staging/vivid").iterdir() if p.is_dir()):
        m, ref = d / "meta.json", d / "ref.jpg"
        if m.exists() and ref.exists():
            meta = json.loads(m.read_text())
            if meta.get("split") == "test":
                pool.setdefault(meta.get("category"), []).append(ref)
    shoes = sorted(e.glob("shoe_set/*/ref.jpg"))
    used: dict[str, int] = {}
    items = json.loads((r / "azyah-code/training/eval/test_garments.json").read_text())["items"]
    for g in items:
        key, cat = g["key"], g["category"]
        dst = g_dir / f"{key}.jpg"
        if dst.exists() and dst.stat().st_size > 0:
            report["have"].append(key)
            continue
        if cat == "shoes":
            same = e / "shoe_set" / key.split("_", 1)[-1] / "ref.jpg"
            src = [same] if same.exists() else shoes
        else:
            src = pool.get(CLIP_CATEGORY.get(cat), [])
        if not src:
            report["none"].append(key)
            continue
        i = used.get(cat, 0)
        used[cat] = i + 1
        s = src[-1 - i % len(src)]  # from the end of the list, away from the clips picked first
        shutil.copy(s, dst)
        report["local"][key] = str(s)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(*sys.argv[1:2])

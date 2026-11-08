"""Pick held-out real clips for the owner's five test cases and pair them with app-catalogue garments.

  python -m eval.select_test_clips --vivid S/vivid --garments eval/test_garments.json --out EVAL/cases \
      [--shoe-set EVAL/shoe_set] [--per-case 2]

Cases (owner, 2026-11-08), from ViViD TEST clips only (never trained on):
  standing  : least motion            turning : most motion
  arms      : most frames with a wrist above its shoulder (MediaPipe pose, if installed)
  lighting  : darkest and brightest clip
  feet      : feet visible (ankles found by MediaPipe pose), tried on with a catalogue SHOE

Writes OUT/clip_list.json for eval.generate_fluxrt --clip-list, OUT/garments/<key>.jpg (downloaded
catalogue photos), OUT/cases.json (scores and what could not be covered), and, with --shoe-set,
OUT/noface/<item>/ "no face" variants of the shoe items (only the lower body is in view) for
generate_fluxrt --pairs.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

CATEGORY_OF_CLIP = {"upper": "tops", "lower": "bottoms", "dress": "dresses"}


def clip_stats(path: Path, n: int = 49, size: int = 128) -> dict:
    """Mean frame-to-frame change (0-1) and mean brightness (0-1) on small grey frames."""
    import av
    import numpy as np
    prev, diffs, lum = None, [], []
    with av.open(str(path)) as c:
        for k, f in enumerate(c.decode(video=0)):
            g = np.asarray(f.to_image().convert("L").resize((size, size)), dtype=np.float32) / 255.0
            lum.append(float(g.mean()))
            if prev is not None:
                diffs.append(float(np.abs(g - prev).mean()))
            prev = g
            if k + 1 >= n:
                break
    return {"motion": sum(diffs) / len(diffs) if diffs else 0.0, "brightness": sum(lum) / len(lum) if lum else 0.0}


class Pose:
    """MediaPipe pose landmarker (Apache-2.0); None-safe if MediaPipe is not installed."""

    URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        if not model_path.exists():
            urllib.request.urlretrieve(self.URL, model_path)
        self.mp = mp
        self.det = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(model_path))))

    def landmarks(self, img):
        import numpy as np
        res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.asarray(img.convert("RGB"))))
        return res.pose_landmarks[0] if res.pose_landmarks else None


def arms_up_fraction(pose, path: Path, n: int = 49, every: int = 6) -> float:
    import av
    hits = total = 0
    with av.open(str(path)) as c:
        for k, f in enumerate(c.decode(video=0)):
            if k >= n:
                break
            if k % every:
                continue
            lm = pose.landmarks(f.to_image())
            total += 1
            # MediaPipe: 11/12 shoulders, 15/16 wrists; y grows downwards
            if lm and (lm[15].y < lm[11].y or lm[16].y < lm[12].y):
                hits += 1
    return hits / total if total else 0.0


def feet_visible(pose, path: Path) -> bool:
    import av
    with av.open(str(path)) as c:
        frames = [f.to_image() for k, f in zip(range(25), c.decode(video=0))]
    if not frames:
        return False
    lm = pose.landmarks(frames[len(frames) // 2])
    return bool(lm) and min(lm[27].visibility, lm[28].visibility) > 0.5 and max(lm[27].y, lm[28].y) < 0.98


def test_clips(vivid: Path) -> list[tuple[Path, dict]]:
    out = []
    for d in sorted(p for p in vivid.iterdir() if p.is_dir()):
        mp = d / "meta.json"
        if mp.exists() and (d / "target.mp4").exists():
            meta = json.loads(mp.read_text())
            if meta.get("split") == "test":
                out.append((d, meta))
    return out


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (azyah eval)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def choose(scored: list[dict], key: str, k: int, reverse: bool, used: set) -> list[dict]:
    out = []
    for s in sorted(scored, key=lambda x: x[key], reverse=reverse):
        if s["clip"] not in used:
            out.append(s)
            used.add(s["clip"])
        if len(out) >= k:
            break
    return out


def make_noface(shoe_set: Path, out: Path, keep_from: float = 0.5) -> int:
    """Lower body only (no face in view) copies of the shoe items."""
    from PIL import Image
    n = 0
    for d in sorted(p for p in shoe_set.iterdir() if p.is_dir()):
        person = next((p for p in d.iterdir() if p.stem == "person"), None)
        ref = next((p for p in d.iterdir() if p.stem == "ref"), None)
        if not (person and ref and (d / "meta.json").exists()):
            continue
        dst = out / f"noface_{d.name}"
        dst.mkdir(parents=True, exist_ok=True)
        img = Image.open(person).convert("RGB")
        img.crop((0, int(img.height * keep_from), img.width, img.height)).save(dst / "person.png")
        (dst / f"ref{ref.suffix}").write_bytes(ref.read_bytes())
        meta = json.loads((d / "meta.json").read_text())
        meta["case"] = "no_face"
        (dst / "meta.json").write_text(json.dumps(meta))
        n += 1
    return n


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vivid", required=True)
    ap.add_argument("--garments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shoe-set")
    ap.add_argument("--per-case", type=int, default=2)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "garments").mkdir(parents=True, exist_ok=True)

    garments = json.loads(Path(a.garments).read_text())["items"]
    for g in garments:
        g["path"] = str(download(g["image_url"], out / "garments" / f"{g['key']}.jpg"))
    by_cat: dict[str, list[dict]] = {}
    for g in garments:
        by_cat.setdefault(g["category"], []).append(g)

    clips = test_clips(Path(a.vivid))
    scored = []
    for d, meta in clips:
        s = clip_stats(d / "target.mp4")
        scored.append({"clip": d.name, "path": str(d / "target.mp4"), "category": meta.get("category", "upper"), **s})

    pose, notes = None, []
    try:
        pose = Pose(out / "pose_landmarker_lite.task")
    except Exception as e:  # noqa: BLE001 - arms/feet cases are then reported as not covered
        notes.append(f"MediaPipe pose unavailable ({type(e).__name__}): arms and feet cases not scored")

    used: set = set()
    picked: dict[str, list[dict]] = {
        "standing": choose(scored, "motion", a.per_case, False, used),
        "turning": choose(scored, "motion", a.per_case, True, used),
    }
    dark = choose(scored, "brightness", 1, False, used)
    bright = choose(scored, "brightness", 1, True, used)
    picked["lighting"] = dark + bright
    if pose:
        # Only a manageable number of clips get the slower pose pass.
        pool = [s for s in scored if s["clip"] not in used][:120]
        for s in pool:
            s["arms_up"] = arms_up_fraction(pose, Path(s["path"]))
        arms = [s for s in sorted(pool, key=lambda x: x["arms_up"], reverse=True) if s["arms_up"] > 0][: a.per_case]
        for s in arms:
            used.add(s["clip"])
        picked["arms"] = arms
        if not arms:
            notes.append("arms: no held-out clip shows a wrist above a shoulder (catwalk footage)")
        feet = []
        for s in [s for s in scored if s["clip"] not in used][:120]:
            if feet_visible(pose, Path(s["path"])):
                feet.append(s)
                used.add(s["clip"])
            if len(feet) >= a.per_case:
                break
        picked["feet"] = feet
        if not feet:
            notes.append("feet: no held-out clip with both ankles clearly in view")

    shoes = by_cat.get("shoes", [])
    items, rot = [], {}
    for case, chosen in picked.items():
        for s in chosen:
            if case == "feet":
                pool = shoes
                cat = "shoes"
            else:
                cat = CATEGORY_OF_CLIP.get(s["category"], "tops")
                pool = by_cat.get(cat) or by_cat.get("tops", [])
            if not pool:
                notes.append(f"{case}: no catalogue garment for category {cat}")
                continue
            k = rot.get(cat, 0)
            g = pool[k % len(pool)]
            rot[cat] = k + 1
            items.append({"key": f"{case}_{s['clip']}", "case": case, "video": s["path"], "ref": g["path"],
                          "category": g["category"], "name": g.get("name"), "description": g.get("description"),
                          "garment_key": g["key"]})
    (out / "clip_list.json").write_text(json.dumps(items, indent=2))
    n_noface = make_noface(Path(a.shoe_set), out / "noface") if a.shoe_set and Path(a.shoe_set).is_dir() else 0
    if not n_noface:
        notes.append("no-face: no shoe set found, no no-face items made")
    report = {"clips_scored": len(scored), "items": len(items), "noface_items": n_noface, "notes": notes,
              "cases": {c: [s["clip"] for s in v] for c, v in picked.items()}}
    (out / "cases.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()

"""The logo test set (fxreal6): does a logo on the shopper's own garment show through a plain try-on?

  python -m eval.logo_set ROOT --from eval/fxa/fx6test.json --out eval/fxa/fx6logo.json [--n 6]

Picks the clips (tops and dresses) whose garment photo is plainest, and writes each one twice into
the new clip list: <key>_clean (the clip as filmed) and <key>_logo (the same clip with a sharp mark
stamped on the garment the person wears, every frame, following the garment). Both entries carry
"plain": true, so the renderer uses the app's plain wording. eval.face_skin_report --leak-pairs then
measures how much of the mark survives (0 = gone, 1 = fully kept). Runs in the MediaPipe venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_tools.logo_stamp import KINDS, stamp  # noqa: E402
from eval.garment_report import border_background, srgb_to_lab  # noqa: E402

CLOTHES = 4
PLAIN_CHROMA = 6.0   # spread of colour (Lab a, b) inside the garment
PLAIN_LIGHT = 35.0   # spread of lightness (folds and shading are allowed)
TOP_SHARE = 0.55     # for a top, the mark goes on the upper part of the clothes area (not the trousers)


def plain_score(rgb: np.ndarray) -> dict | None:
    """How plain a garment photo is: colour spread and lightness spread inside the garment (the photo
    minus its backdrop). None when the garment cannot be told from the backdrop."""
    garment = ~border_background(rgb)
    share = float(garment.mean())
    if share < 0.05 or share > 0.95:
        return None
    lab = srgb_to_lab(rgb[garment])
    chroma = float(np.hypot(lab[:, 1].std(), lab[:, 2].std()))
    light = float(np.percentile(lab[:, 0], 95) - np.percentile(lab[:, 0], 5))
    return {"chroma": round(chroma, 2), "light": round(light, 2), "plain": chroma < PLAIN_CHROMA and light < PLAIN_LIGHT,
            "rank": round(chroma + light / 5, 2)}


def garment_area(cat: np.ndarray, category: str | None) -> np.ndarray:
    """Where the mark may go: the clothes; for a top, only the upper part of them."""
    m = cat == CLOTHES
    if category in ("tops", "upper", "outerwear") and m.any():
        rows = np.nonzero(m.any(axis=1))[0]
        cut = int(rows.min() + TOP_SHARE * (rows.max() - rows.min() + 1))
        m = m.copy()
        m[cut:, :] = False
    return m


def stamp_frames(frames: list[np.ndarray], cats: list[np.ndarray], category: str | None, seed: int) -> tuple[list[np.ndarray], str]:
    """The same mark on every frame, placed relative to the garment so it moves with it."""
    kind = KINDS[seed % len(KINDS)]
    out = []
    for f, c in zip(frames, cats):
        img, _ = stamp(f, garment_area(c, category), np.random.default_rng(seed), kind=kind)
        out.append(img)
    return out, kind


def choose(items: list[dict], scores: dict[str, dict | None], n: int) -> list[dict]:
    """The n plainest garments among tops and dresses; truly plain ones first, then the nearest."""
    ok = [it for it in items if it.get("category") in ("tops", "dresses") and scores.get(it["key"])]
    ok.sort(key=lambda it: (not scores[it["key"]]["plain"], scores[it["key"]]["rank"]))
    return ok[:n]


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="/workspace/azyah-vton")
    ap.add_argument("--from", dest="src", required=True, help="clip list to choose from (relative to ROOT or absolute)")
    ap.add_argument("--out", required=True, help="new clip list (relative to ROOT or absolute)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    a = ap.parse_args(argv)
    from PIL import Image
    from data_tools.clothes_mask import Segmenter
    from eval.generate_fluxrt import _read_frames, _write_mp4
    root = Path(a.root)
    at = lambda p: Path(p) if Path(p).is_absolute() else root / p  # noqa: E731
    items = json.loads(at(a.src).read_text())
    scores = {}
    for it in items:
        try:
            im = Image.open(it["ref"]).convert("RGB")
            im.thumbnail((320, 320))
            scores[it["key"]] = plain_score(np.asarray(im))
        except OSError:
            scores[it["key"]] = None
    picked = choose(items, scores, a.n)
    seg = Segmenter(Path(a.model))
    clips = at(a.out).parent / "logo_clips"
    clips.mkdir(parents=True, exist_ok=True)
    out = []
    for k, it in enumerate(picked):
        frames = _read_frames(Path(it["video"]), a.frames)
        if not frames:
            continue
        cats = [np.squeeze(seg(f)) for f in frames]
        stamped, kind = stamp_frames([np.asarray(f) for f in frames], cats, it.get("category"), seed=101 + k)
        video = clips / f"{it['key']}.mp4"
        _write_mp4(video, [Image.fromarray(s) for s in stamped], fps=25)
        base = {**it, "plain": True, "case": "fx6logo"}
        out.append({**base, "key": f"{it['key']}_clean"})
        out.append({**base, "key": f"{it['key']}_logo", "video": str(video), "mark": kind})
        print(f"LOGO_CLIP {it['key']}: {kind}; garment photo {scores[it['key']]}", flush=True)
    at(a.out).write_text(json.dumps(out, indent=2))
    print(f"LOGO_SET {len(out) // 2} clips ({sum(1 for it in picked if scores[it['key']]['plain'])} with a truly plain garment photo) -> {at(a.out)}")
    return out


if __name__ == "__main__":
    main()

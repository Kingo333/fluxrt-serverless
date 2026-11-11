"""The owner's five priorities, measured per clip (fxreal5, owner 2026-11-11): colour, detail, fabric, movement,
consistency of the try-on garment.

  python -m eval.garment_report seg   RESULTS_DIR --models base fav2 fx5 [--model SEG.tflite]      (venv-mp)
  python -m eval.garment_report score RESULTS_DIR --models fav2 fx5 [--baseline fav2] [--out report.md] [--no-dino]

RESULTS_DIR/<item>/ holds input.mp4 (camera), out_<model>.mp4, ref.* (the garment photo) and item.json
(category). `seg` (MediaPipe multiclass selfie segmenter) saves seg_camera.npz, seg_<model>.npz and
seg_ref.npz; `score` (numpy, PIL, av; DINOv2-small for detail when torch is there) reads them. The garment is
the "clothes" class in the model's output, limited to the upper 55% of the person for tops and the lower 55%
for bottoms (so the trousers do not count for a top); on the photo it is the clothes class when a model wears
it, else everything but the photo's backdrop. Frames where the camera shows almost no person are left out.

  COLOUR       colour_de    CIE76 distance between the median Lab colour of the garment and of the photo (lower)
  DETAIL       likeness     DINOv2 similarity of the garment cut-out and the photo's garment (motifs, trims,
                            cape, piping, shape; higher)
  FABRIC       fabric       |log| of the ratio of the garment's light-dark spread (95th - 5th percentile
                            lightness) to the photo's: shine, sheen, print contrast (lower; 0 = same)
  MOVEMENT     follow_px    how far the garment's centre moves differently from the person's centre between
                            frames, in pixels (lower: the dress moves WITH the body); on_body = share of the
                            garment on the person (higher)
  CONSISTENCY  jitter       garment shape change between frames minus the person's own (lower: no jumps)
               flicker_de   colour change of the garment between frames (lower: no flicker)

Then a scorecard: per priority, on how many clips each model beats --baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BACKGROUND, HAIR, BODY_SKIN, FACE, CLOTHES, OTHERS = 0, 1, 2, 3, 4, 5
MIN_PERSON = 0.08
EPS = {"colour_de": 0.5, "likeness": 0.005, "fabric": 0.02, "follow_px": 0.2, "jitter": 0.002, "flicker_de": 0.1}
LOWER_BETTER = {"colour_de": True, "likeness": False, "fabric": True, "follow_px": True, "jitter": True, "flicker_de": True}
PRIORITIES = (("COLOUR", "colour_de"), ("DETAIL", "likeness"), ("FABRIC", "fabric"), ("MOVEMENT", "follow_px"),
              ("CONSISTENCY", "jitter"), ("CONSISTENCY", "flicker_de"))


# ---------------------------------------------------------------- pure helpers (unit-tested)
def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) sRGB 0-255 -> CIE Lab (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def median_lab(rgb: np.ndarray, mask: np.ndarray, max_px: int = 20000, seed: int = 0):
    px = rgb[mask]
    if len(px) == 0:
        return None
    if len(px) > max_px:
        px = px[np.random.default_rng(seed).choice(len(px), max_px, replace=False)]
    return np.median(srgb_to_lab(px), axis=0)


def delta_e(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def spread(rgb: np.ndarray, mask: np.ndarray):
    px = rgb[mask]
    if len(px) < 50:
        return None
    lum = srgb_to_lab(px)[..., 0]
    return float(np.percentile(lum, 95) - np.percentile(lum, 5))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def centroid(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def band_mask(clothes: np.ndarray, person: np.ndarray, category) -> np.ndarray:
    """The try-on garment's part of the clothes: upper 55% of the person for tops, lower 55% for bottoms."""
    c = str(category or "").lower()
    rows = np.nonzero(person.any(axis=1))[0]
    if not len(rows) or c not in ("tops", "bottoms", "upper", "lower"):
        return clothes
    top, bottom = int(rows.min()), int(rows.max())
    h = bottom - top + 1
    out = np.zeros_like(clothes)
    if c in ("tops", "upper"):
        out[top: top + int(0.55 * h)] = clothes[top: top + int(0.55 * h)]
    else:
        out[top + int(0.45 * h): bottom + 1] = clothes[top + int(0.45 * h): bottom + 1]
    return out


def border_background(rgb: np.ndarray, tol: int = 40, step: int = 6) -> np.ndarray:
    """The photo's backdrop: up to two backdrop colours are read off the picture's edge (an off-white or grey
    studio backdrop, a letterbox bar), and the backdrop grows inwards from the edge through small colour steps
    (at most `step` per pixel) while staying within `tol` of a backdrop colour. A light trim, piping or lace INSIDE
    the garment is not reached through small steps, so it stays garment."""
    from collections import deque
    px = rgb.astype(np.int16)
    h, w = px.shape[:2]
    ring = np.concatenate([px[0], px[-1], px[:, 0], px[:, -1]])
    cols, rest = [], ring
    for _ in range(2):
        if len(rest) < 0.15 * len(ring):
            break
        c = np.median(rest, axis=0)
        near = np.abs(rest - c).max(axis=1) <= 12
        if near.sum() < 0.15 * len(ring):
            break
        cols.append(c)
        rest = rest[~near]
    bg = np.zeros((h, w), bool)
    if not cols:
        return bg
    ok, seed = np.zeros((h, w), bool), np.zeros((h, w), bool)
    for c in cols:
        d = np.abs(px - c).max(axis=-1)
        ok |= d <= tol
        seed |= d <= 12
    right = np.abs(px[:, 1:] - px[:, :-1]).max(axis=-1) <= step   # x -> x+1 is a small step
    down = np.abs(px[1:] - px[:-1]).max(axis=-1) <= step           # y -> y+1 is a small step
    q = deque()
    for y, x in [(y, x) for y in range(h) for x in (0, w - 1)] + [(y, x) for x in range(w) for y in (0, h - 1)]:
        if seed[y, x] and not bg[y, x]:
            bg[y, x] = True
            q.append((y, x))
    while q:
        y, x = q.popleft()
        for yy, xx, small in ((y, x + 1, x + 1 < w and right[y, x]), (y, x - 1, x > 0 and right[y, x - 1]),
                              (y + 1, x, y + 1 < h and down[y, x]), (y - 1, x, y > 0 and down[y - 1, x])):
            if small and ok[yy, xx] and not bg[yy, xx]:
                bg[yy, xx] = True
                q.append((yy, xx))
    return bg


def ref_garment_mask(rgb: np.ndarray, cat) -> np.ndarray:
    """Clothes when a model wears the garment on the photo, else everything but the photo's backdrop."""
    if cat is not None and cat.shape == rgb.shape[:2] and float(np.isin(cat, (FACE, BODY_SKIN, HAIR)).mean()) >= 0.005:
        return cat == CLOTHES
    from PIL import Image
    small = rgb
    if max(rgb.shape[:2]) > 320:                       # work on a smaller copy, then back to full size
        k = 320 / max(rgb.shape[:2])
        small = np.asarray(Image.fromarray(rgb).resize((max(1, round(rgb.shape[1] * k)), max(1, round(rgb.shape[0] * k)))))
    bg = border_background(small)
    if (~bg).mean() < 0.03:                            # the backdrop leaked into a garment of almost its colour:
        bg = border_background(small, tol=12, step=255)  # only the backdrop colour itself
    if bg.shape != rgb.shape[:2]:
        bg = np.asarray(Image.fromarray(bg.astype(np.uint8) * 255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
    return ~bg


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for _ in range(r):
        p = np.pad(out, 1)
        out = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return out


def clip_metrics(cam_frames, out_frames, cam_cats, out_cats, ref_rgb, ref_mask, category, dino=None, n_dino: int = 6) -> dict:
    """All metrics of one model on one clip (frames: lists of HxWx3 uint8; cats: lists of HxW uint8)."""
    n = min(len(cam_frames), len(out_frames), len(cam_cats), len(out_cats))
    valid = [t for t in range(n) if float((cam_cats[t] != BACKGROUND).mean()) >= MIN_PERSON]
    res = {"frames": n, "valid_frames": len(valid)}
    if len(valid) < 2 or ref_mask.sum() < 50:
        return res
    person = {t: cam_cats[t] != BACKGROUND for t in valid}
    garm = {t: band_mask(out_cats[t] == CLOTHES, person[t], category) for t in valid}
    pooled = np.concatenate([out_frames[t][garm[t]] for t in valid if garm[t].any()] or [np.zeros((0, 3), np.uint8)])
    ref_lab = median_lab(ref_rgb, ref_mask)
    if len(pooled) and ref_lab is not None:
        res["colour_de"] = round(delta_e(median_lab(pooled.reshape(-1, 1, 3), np.ones((len(pooled), 1), bool)), ref_lab), 3)
        s_ref, s_out = spread(ref_rgb, ref_mask), spread(pooled.reshape(-1, 1, 3), np.ones((len(pooled), 1), bool))
        if s_ref and s_out:
            res["fabric"] = round(abs(float(np.log(s_out / s_ref))), 4)
    on = [np.logical_and(garm[t], _dilate(person[t], 6)).sum() / max(1, garm[t].sum()) for t in valid if garm[t].any()]
    if on:
        res["on_body"] = round(float(np.mean(on)), 4)
    follow, jit, pj, flick = [], [], [], []
    for t0, t1 in zip(valid, valid[1:]):
        if t1 != t0 + 1:
            continue
        cg0, cg1, cp0, cp1 = centroid(garm[t0]), centroid(garm[t1]), centroid(person[t0]), centroid(person[t1])
        if None not in (cg0, cg1, cp0, cp1):
            dg = np.subtract(cg1, cg0)
            dp = np.subtract(cp1, cp0)
            follow.append(float(np.linalg.norm(dg - dp)))
        jit.append(1 - iou(garm[t0], garm[t1]))
        pj.append(1 - iou(person[t0], person[t1]))
        l0, l1 = median_lab(out_frames[t0], garm[t0]), median_lab(out_frames[t1], garm[t1])
        if l0 is not None and l1 is not None:
            flick.append(delta_e(l0, l1))
    if follow:
        res["follow_px"] = round(float(np.mean(follow)), 3)
    if jit:
        res["jitter"] = round(float(np.mean(jit) - np.mean(pj)), 4)
    if flick:
        res["flicker_de"] = round(float(np.mean(flick)), 3)
    if dino is not None:
        picks = [valid[int(i)] for i in np.linspace(0, len(valid) - 1, min(n_dino, len(valid)))]
        ref_cut = cutout(ref_rgb, ref_mask)
        sims = [dino(cutout(out_frames[t], garm[t]), ref_cut) for t in picks if garm[t].sum() >= 50]
        if sims and ref_cut is not None:
            res["likeness"] = round(float(np.mean(sims)), 4)
    return res


def cutout(rgb: np.ndarray, mask: np.ndarray, pad: int = 4):
    """The garment on white, cropped to its box (PIL image), or None."""
    from PIL import Image
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    img = np.where(mask[..., None], rgb, 255).astype(np.uint8)[y0:y1, x0:x1]
    return Image.fromarray(img)


def scorecard(per_item: dict, models: list[str], baseline: str) -> list[dict]:
    """Per priority metric and candidate: clips where it beats / loses to the baseline, and the means."""
    rows = []
    for prio, key in PRIORITIES:
        for m in models:
            if m == baseline:
                continue
            better = worse = n = 0
            vals_m, vals_b = [], []
            for item in per_item.values():
                a, b = item.get(m, {}).get(key), item.get(baseline, {}).get(key)
                if a is None or b is None:
                    continue
                n += 1
                vals_m.append(a)
                vals_b.append(b)
                d = (b - a) if LOWER_BETTER[key] else (a - b)
                better += d > EPS[key]
                worse += d < -EPS[key]
            if n:
                rows.append({"priority": prio, "metric": key, "model": m, "baseline": baseline, "clips": n,
                             "better": better, "worse": worse, "mean": round(float(np.mean(vals_m)), 4),
                             "baseline_mean": round(float(np.mean(vals_b)), 4)})
    return rows


# ---------------------------------------------------------------- I/O
def _video(p: Path) -> list[np.ndarray]:
    import av
    with av.open(str(p)) as c:
        return [np.asarray(f.to_image().convert("RGB")) for f in c.decode(video=0)]


def _ref(d: Path):
    from PIL import Image
    p = next((q for q in sorted(d.iterdir()) if q.stem == "ref"), None)
    return np.asarray(Image.open(p).convert("RGB")) if p else None


def _items(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "input.mp4").exists())


def seg_stage(root: Path, models: list[str], seg_model: str) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_tools.clothes_mask import Segmenter
    from PIL import Image
    seg0 = Segmenter(Path(seg_model))
    seg = lambda arr: np.squeeze(seg0(Image.fromarray(arr))).astype(np.uint8)  # noqa: E731
    done = 0
    for d in _items(root):
        ref = _ref(d)
        if ref is not None and not (d / "seg_ref.npz").exists():
            np.savez_compressed(d / "seg_ref.npz", cat=seg(ref))
        for name, vid in [("camera", d / "input.mp4")] + [(m, d / f"out_{m}.mp4") for m in models]:
            if vid.exists() and not (d / f"seg_{name}.npz").exists():
                np.savez_compressed(d / f"seg_{name}.npz", cat=np.stack([seg(f) for f in _video(vid)]))
        done += 1
    print(json.dumps({"segmented_items": done}))
    return {"segmented_items": done}


class Dino:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(self.dev).eval()

    def __call__(self, a, b) -> float:
        if a is None or b is None:
            return float("nan")
        with self.torch.no_grad():
            x = self.proc(images=[a, b], return_tensors="pt").to(self.dev)
            f = self.torch.nn.functional.normalize(self.model(**x).pooler_output, dim=-1)
            return float((f[0] * f[1]).sum())


def score_stage(root: Path, models: list[str], baseline: str, out: Path | None, use_dino: bool,
                json_out: Path | None = None) -> dict:
    dino = None
    if use_dino:
        try:
            dino = Dino()
        except Exception as e:  # noqa: BLE001 - detail is skipped, the rest still runs
            print(f"(DINO unavailable, detail skipped: {type(e).__name__}: {e})", flush=True)
    per_item = {}
    for d in _items(root):
        if not (d / "seg_camera.npz").exists():
            continue
        meta = json.loads((d / "item.json").read_text()) if (d / "item.json").exists() else {}
        ref = _ref(d)
        if ref is None:
            continue
        ref_cat = np.load(d / "seg_ref.npz")["cat"] if (d / "seg_ref.npz").exists() else None
        ref_mask = ref_garment_mask(ref, ref_cat)
        cam = _video(d / "input.mp4")
        cam_cats = list(np.load(d / "seg_camera.npz")["cat"])
        per_item[d.name] = {}
        for m in models:
            if not (d / f"out_{m}.mp4").exists() or not (d / f"seg_{m}.npz").exists():
                continue
            outs = _video(d / f"out_{m}.mp4")
            cats = list(np.load(d / f"seg_{m}.npz")["cat"])
            r = clip_metrics(cam, outs, cam_cats, cats, ref, ref_mask, meta.get("category"), dino)
            if r.get("likeness") is not None and np.isnan(r["likeness"]):
                r.pop("likeness")
            per_item[d.name][m] = r
    card = scorecard(per_item, models, baseline)
    for r in card:
        arrow = "lower is better" if LOWER_BETTER[r["metric"]] else "higher is better"
        print(f"PRIORITY {r['priority']} ({r['metric']}, {arrow}): {r['model']} better on {r['better']}/{r['clips']} clips, "
              f"worse on {r['worse']} (mean {r['mean']} vs {r['baseline']} {r['baseline_mean']})")
    result = {"baseline": baseline, "models": models, "scorecard": card, "items": per_item}
    (json_out or root / "garment_report.json").write_text(json.dumps(result, indent=1))
    if out:
        keys = ("colour_de", "likeness", "fabric", "follow_px", "on_body", "jitter", "flicker_de")
        lines = ["| clip | model | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
        for item, ms in per_item.items():
            for m, r in ms.items():
                lines.append(f"| {item} | {m} | " + " | ".join(str(r.get(k, "-")) for k in keys) + " |")
        lines += ["", "| priority | metric | model | better | worse | clips | mean | baseline mean |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['priority']} | {r['metric']} | {r['model']} | {r['better']} | {r['worse']} | {r['clips']} | "
                  f"{r['mean']} | {r['baseline_mean']} |" for r in card]
        out.write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["seg", "score"])
    ap.add_argument("results")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", help="model to compare against (score); default: the first of --models")
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite", help="segmenter (seg)")
    ap.add_argument("--out", help="markdown report (score)")
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--json", help="report json (score); default RESULTS/garment_report.json")
    a = ap.parse_args(argv)
    root = Path(a.results)
    if a.stage == "seg":
        return seg_stage(root, a.models, a.model)
    return score_stage(root, a.models, a.baseline or a.models[0], Path(a.out) if a.out else None, not a.no_dino,
                       Path(a.json) if a.json else None)


if __name__ == "__main__":
    main()

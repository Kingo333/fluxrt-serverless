"""Where and when each model jitters, frame by frame (owner's 5-item re-test, 2026-11-08).

  python -m eval.flicker_report RESULTS_DIR OUT_DIR [--models base l2000 l1500] [--model SEG.tflite] [--no-seg]

Each RESULTS_DIR/<item>/ has input.mp4 and out_<model>.mp4 (same size, same frames). For every frame
pair (t-1, t) and region, "excess jitter" = how much the output changed minus how much the camera
changed (mean |difference|, 0-255). 0 = the output moves exactly as much as the person; higher =
it flickers or shifts by itself. Regions come from MediaPipe's multiclass selfie segmenter:
  garment     clothes area of that model's own output (the new garment)
  accessory   bags, belts and other accessories in the camera frame (class "others")
  skin        face + body skin in the camera frame
  background  background in the camera frame and in the output
"still" = the frames where the camera moves least (lowest 40%): does the output settle when the
person stops? "drift" = the same over 8 frames instead of 1: slow change of the material over time.
Writes OUT_DIR/flicker.md (table + a per-item timeline), flicker.json and <item>_heat.png: where the
excess jitter is, for each model, over a dimmed middle frame (labels name the models).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BACKGROUND, HAIR, BODY_SKIN, FACE, CLOTHES, OTHERS = 0, 1, 2, 3, 4, 5
BARS = "▁▂▃▄▅▆▇█"


def _frames(path: Path, max_frames: int = 49) -> list[np.ndarray]:
    import av
    out = []
    with av.open(str(path)) as c:
        for k, f in enumerate(c.decode(video=0)):
            if k >= max_frames:
                break
            out.append(np.asarray(f.to_image().convert("RGB"), dtype=np.float32))
    return out


def _d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).mean(axis=2)


def _m(x: np.ndarray, mask: np.ndarray | None, min_px: int = 150) -> float | None:
    if mask is None:
        return float(x.mean())
    return float(x[mask].mean()) if mask.sum() >= min_px else None


def _mean(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return round(float(np.mean(v)), 2) if v else None


def spark(v: list, top: float) -> str:
    return "".join(" " if x is None else BARS[min(len(BARS) - 1, int(max(0.0, x) / top * (len(BARS) - 1) + 0.5))]
                   for x in v) if top > 0 else ""


def item_report(d: Path, models: list[str], seg, lag: int = 8) -> dict:
    cam = _frames(d / "input.mp4")
    outs = {m: _frames(d / f"out_{m}.mp4") for m in models}
    n = min([len(cam)] + [len(v) for v in outs.values()])
    cam = [c if c.shape == outs[models[0]][0].shape else c for c in cam[:n]]
    cat_in = [seg(c) for c in cam] if seg else None
    cat_out = {m: [seg(o) for o in outs[m][:n]] for m in models} if seg else None
    motion = [float(_d(cam[t], cam[t - 1]).mean()) for t in range(1, n)]
    still = set(np.argsort(motion)[: max(1, int(len(motion) * 0.4))] + 1) if motion else set()
    res = {"frames": n, "camera_motion": round(float(np.mean(motion)), 2) if motion else None, "models": {}}
    heat = {}
    for m in models:
        o = outs[m]
        per = {k: [] for k in ("garment", "accessory", "skin", "background", "whole")}
        drift, garment_t = [], []
        hm = np.zeros(o[0].shape[:2], dtype=np.float32)
        for t in range(1, n):
            dc, do = _d(cam[t], cam[t - 1]), _d(o[t], o[t - 1])
            ex = do - dc
            hm += np.maximum(ex, 0)
            masks = {"whole": None}
            if seg:
                masks["garment"] = (cat_out[m][t] == CLOTHES) | (cat_out[m][t - 1] == CLOTHES)
                masks["accessory"] = (cat_in[t] == OTHERS) & (cat_in[t - 1] == OTHERS)
                masks["skin"] = np.isin(cat_in[t], (FACE, BODY_SKIN)) & np.isin(cat_in[t - 1], (FACE, BODY_SKIN))
                masks["background"] = ((cat_in[t] == BACKGROUND) & (cat_in[t - 1] == BACKGROUND)
                                       & (cat_out[m][t] == BACKGROUND) & (cat_out[m][t - 1] == BACKGROUND))
            for k, mk in masks.items():
                per[k].append(_m(ex, mk))
            garment_t.append(per["garment"][-1] if seg else per["whole"][-1])
            if t >= lag:
                g = masks.get("garment")
                g8 = g if g is None else g & ((cat_out[m][t - lag] == CLOTHES) if seg else g)
                drift.append(_m(_d(o[t], o[t - lag]) - _d(cam[t], cam[t - lag]), g8))
        key = "garment" if seg else "whole"
        res["models"][m] = {
            **{k: _mean(v) for k, v in per.items() if any(x is not None for x in v)},
            f"{key}_still": _mean([per[key][t - 1] for t in range(1, n) if t in still]),
            f"{key}_moving": _mean([per[key][t - 1] for t in range(1, n) if t not in still]),
            f"{key}_drift8": _mean(drift),
            "timeline": [None if x is None else round(x, 2) for x in garment_t],
        }
        heat[m] = hm / max(1, n - 1)
    res["timeline_camera"] = [round(x, 2) for x in motion]
    return res, heat, cam[n // 2]


def heat_image(mid: np.ndarray, heat: dict, path: Path, scale: float = 12.0) -> None:
    from PIL import Image, ImageDraw
    h, w = mid.shape[:2]
    panels = [Image.fromarray(mid.astype(np.uint8))]
    labels = ["camera"]
    for m, hm in heat.items():
        a = np.clip(hm / scale, 0, 1)[..., None]
        base = mid * 0.35
        red = np.zeros_like(mid)
        red[..., 0], red[..., 1] = 255, 60
        panels.append(Image.fromarray((base * (1 - a) + red * a).astype(np.uint8)))
        labels.append(f"{m} (mean {hm.mean():.1f})")
    canvas = Image.new("RGB", (w * len(panels), h + 24), (0, 0, 0))
    dr = ImageDraw.Draw(canvas)
    for i, (p, lb) in enumerate(zip(panels, labels)):
        canvas.paste(p, (i * w, 24))
        dr.text((i * w + 6, 5), lb, fill=(255, 255, 255))
    canvas.save(path)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--models", nargs="+", default=["base", "l2000", "l1500"])
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    ap.add_argument("--no-seg", action="store_true")
    a = ap.parse_args(argv)
    seg = None
    if not a.no_seg:
        from data_tools.clothes_mask import Segmenter
        from PIL import Image
        s = Segmenter(Path(a.model))
        seg = lambda arr: np.squeeze(s(Image.fromarray(arr.astype(np.uint8))))  # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report, lines = {}, []
    key = "garment" if seg else "whole"
    for d in sorted(p for p in Path(a.results).iterdir() if p.is_dir()):
        models = [m for m in a.models if (d / f"out_{m}.mp4").exists()]
        if not (d / "input.mp4").exists() or not models:
            continue
        r, heat, mid = item_report(d, models, seg)
        report[d.name] = r
        heat_image(mid, heat, out / f"{d.name}_heat.png")
        cols = [c for c in ("garment", "accessory", "skin", "background", "whole") if any(c in v for v in r["models"].values())]
        lines += [f"## {d.name} (camera motion {r['camera_motion']})", "",
                  "| model | " + " | ".join(cols) + f" | {key} when still | {key} when moving | {key} drift (8 frames) |",
                  "|---" * (len(cols) + 4) + "|"]
        for m, v in r["models"].items():
            lines.append(f"| {m} | " + " | ".join(str(v.get(c, "-")) for c in cols)
                         + f" | {v[key + '_still']} | {v[key + '_moving']} | {v[key + '_drift8']} |")
        top = max([x for v in r["models"].values() for x in v["timeline"] if x is not None] + r["timeline_camera"] + [1e-6])
        lines += ["", "Timeline, frame by frame (taller = more change):", "```",
                  f"camera motion   {spark(r['timeline_camera'], top)}"]
        lines += [f"{m:<15} {spark(v['timeline'], top)}" for m, v in r["models"].items()]
        lines += ["```", ""]
    text = ("# Excess jitter: output change minus camera change (0-255; 0 = moves only as the person moves)\n\n"
            + "\n".join(lines))
    (out / "flicker.md").write_text(text)
    (out / "flicker.json").write_text(json.dumps(report, indent=1))
    print(text)
    return report


if __name__ == "__main__":
    main()

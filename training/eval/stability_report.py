"""Steadiness, sharpness and colour constancy of each model on the 5-item clips (owner, 2026-11-08).

  python -m eval.stability_report RESULTS_DIR [--models base l2000 f3_2000] [--model SEG.tflite] [--out report.md]

Each RESULTS_DIR/<item>/ has input.mp4 and out_<model>.mp4. Frames where the camera shows almost no
person (walk-in, under 8% of the picture) are left out. Per model, on the garment area (MediaPipe
"clothes" in that model's output or any other model's output):
  wobble     mean |frame t - average of frames t-1 and t+1| (0-255): flicker and jumps, not smooth motion
  sharpness  mean |Laplacian| of the brightness inside the garment: fine detail (higher = crisper)
  colour     mean distance of the garment's average colour from its own average over the clip (0-255)
             (lower = the material keeps its colour)
plus wobble on the whole frame. Prints a markdown table; the camera row is the real video for reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BACKGROUND, CLOTHES = 0, 4


def _frames(p: Path) -> list[np.ndarray]:
    import av
    with av.open(str(p)) as c:
        return [np.asarray(f.to_image().convert("RGB"), dtype=np.float32) for f in c.decode(video=0)]


def _lap(g: np.ndarray) -> np.ndarray:
    return np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])


def item_stats(vids: dict[str, list[np.ndarray]], cats: dict[str, list[np.ndarray]]) -> dict:
    n = min(len(v) for v in vids.values())
    walk = {t for t in range(n) if float((cats["camera"][t] != BACKGROUND).mean()) < 0.08}
    outs = [k for k in vids if k != "camera"]
    res = {"frames": n, "walk_in_frames": len(walk)}
    for k, fr in vids.items():
        wob_g, wob_w, sharp, cols = [], [], [], []
        for t in range(n):
            if t in walk:
                continue
            g = np.zeros(fr[t].shape[:2], dtype=bool)
            for o in outs:
                g |= cats[o][t] == CLOTHES
            if g.sum() < 300:
                continue
            grey = fr[t].mean(axis=2)
            sharp.append(float(_lap(grey)[g[1:-1, 1:-1]].mean()))
            cols.append(fr[t][g].mean(0))
            if 0 < t < n - 1 and t - 1 not in walk and t + 1 not in walk:
                w = np.abs(fr[t] - (fr[t - 1] + fr[t + 1]) / 2).mean(axis=2)
                wob_g.append(float(w[g].mean()))
                wob_w.append(float(w.mean()))
        c = np.array(cols) if cols else np.zeros((1, 3))
        res[k] = {"wobble_garment": round(float(np.mean(wob_g)), 2) if wob_g else None,
                  "wobble_frame": round(float(np.mean(wob_w)), 2) if wob_w else None,
                  "sharpness": round(float(np.mean(sharp)), 2) if sharp else None,
                  "colour_wander": round(float(np.linalg.norm(c - c.mean(0), axis=1).mean()), 2)}
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--models", nargs="+", default=["base", "l2000"])
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from PIL import Image
    from data_tools.clothes_mask import Segmenter
    s = Segmenter(Path(a.model))
    seg = lambda arr: np.squeeze(s(Image.fromarray(arr.astype(np.uint8))))  # noqa: E731
    report, lines = {}, []
    cols = ("wobble_garment", "wobble_frame", "sharpness", "colour_wander")
    for d in sorted(p for p in Path(a.results).iterdir() if p.is_dir()):
        models = [m for m in a.models if (d / f"out_{m}.mp4").exists()]
        if not (d / "input.mp4").exists() or not models:
            continue
        vids = {"camera": _frames(d / "input.mp4"), **{m: _frames(d / f"out_{m}.mp4") for m in models}}
        cats = {k: [seg(f) for f in v] for k, v in vids.items()}
        r = item_stats(vids, cats)
        report[d.name] = r
        lines += [f"## {d.name} ({r['frames']} frames, {r['walk_in_frames']} walk-in frames left out)", "",
                  "| | garment wobble (lower = steadier) | frame wobble | garment sharpness (higher = crisper) "
                  "| garment colour wander (lower = constant) |", "|---|---|---|---|---|"]
        for k in ["camera"] + models:
            lines.append(f"| {k} | " + " | ".join(str(r[k][c]) for c in cols) + " |")
        lines.append("")
    text = "# Steadiness, sharpness, colour constancy\n\n" + "\n".join(lines)
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(text)
    return report


if __name__ == "__main__":
    main()

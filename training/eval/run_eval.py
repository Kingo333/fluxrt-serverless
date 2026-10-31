"""Score every model's outputs on the same held-out clips.

  python3 -m eval.run_eval RESULTS_DIR [--out report.md]

RESULTS_DIR/<clip_id>/
  input.mp4 | input.png       what the camera saw (no edit)
  ref.jpg | ref.png           garment photo
  mask.png                    garment area (optional; whole frame if missing)
  out_<model>.mp4 | .png      one per model, e.g. out_joyai-baseline.mp4, out_azyah-vton-exp1.mp4
  perf_<model>.json           optional: {"fps": .., "latency_ms_p50": .., "gpu_mem_gb": ..}

Writes a per-model table (mean over clips) and per-clip JSON next to the report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .metrics import score_clip

MAX_FRAMES = 96


def _frames(path: Path):
    from PIL import Image
    if path.suffix in (".png", ".jpg", ".jpeg"):
        return [Image.open(path).convert("RGB")]
    import av
    out = []
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            out.append(f.to_image())
            if len(out) >= MAX_FRAMES:
                break
    return out


def _one(d: Path, stem: str) -> Path | None:
    for ext in ("mp4", "png", "jpg", "jpeg"):
        p = d / f"{stem}.{ext}"
        if p.exists():
            return p
    return None


def evaluate(results_dir: str) -> dict:
    from PIL import Image
    per_clip: dict[str, dict] = {}
    per_model: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        inp, ref = _one(d, "input"), _one(d, "ref")
        if not (inp and ref):
            continue
        mask_p = d / "mask.png"
        mask = Image.open(mask_p) if mask_p.exists() else None
        in_frames, ref_im = _frames(inp), Image.open(ref)
        per_clip[d.name] = {}
        for out in sorted(d.glob("out_*")):
            model = out.stem[len("out_"):]
            out_frames = _frames(out)
            n = min(len(in_frames), len(out_frames))
            s = score_clip(in_frames[:n], out_frames[:n], ref_im, mask)
            perf = d / f"perf_{model}.json"
            if perf.exists():
                s.update(json.loads(perf.read_text()))
            per_clip[d.name][model] = s
            per_model[model].append(s)
    summary = {}
    for model, rows in per_model.items():
        keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float))})
        summary[model] = {"clips": len(rows), **{k: round(sum(r[k] for r in rows if k in r) / sum(1 for r in rows if k in r), 4) for k in keys}}
    return {"summary": summary, "per_clip": per_clip}


def to_markdown(result: dict) -> str:
    s = result["summary"]
    if not s:
        return "No results found.\n"
    cols = ["clips", "garment_colour", "garment_detail", "preservation", "flicker", "fps", "latency_ms_p50", "gpu_mem_gb"]
    cols = [c for c in cols if any(c in v for v in s.values())]
    lines = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for model, v in sorted(s.items()):
        lines.append(f"| {model} | " + " | ".join(str(v.get(c, "—")) for c in cols) + " |")
    lines += ["", "Higher is better for garment_colour, preservation, fps; garment_detail best near 1.0; "
              "lower is better for flicker, latency, memory. Metrics are proxies: confirm with the blinded review."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default="eval_report.md")
    a = ap.parse_args()
    r = evaluate(a.results_dir)
    Path(a.out).write_text(to_markdown(r))
    Path(a.out).with_suffix(".json").write_text(json.dumps(r, indent=2))
    print(to_markdown(r))


if __name__ == "__main__":
    main()

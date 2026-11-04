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
    """Yields frames one at a time (a CPU Pod has 512 MB; whole clips do not fit)."""
    from PIL import Image
    if path.suffix in (".png", ".jpg", ".jpeg"):
        yield Image.open(path).convert("RGB")
        return
    import av
    with av.open(str(path)) as c:
        for k, f in enumerate(c.decode(c.streams.video[0])):
            if k >= MAX_FRAMES:
                break
            yield f.to_image()


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
    masked: list[bool] = []
    for d in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        inp, ref = _one(d, "input"), _one(d, "ref")
        if not (inp and ref):
            continue
        mask_p = d / "mask.png"
        mask = Image.open(mask_p) if mask_p.exists() else None
        ref_im = Image.open(ref).convert("RGB")
        tgt = _one(d, "target")
        masked.append(mask is not None)
        per_clip[d.name] = {}
        for out in sorted(d.glob("out_*")):
            model = out.stem[len("out_"):]
            s = score_clip(_frames(inp), _frames(out), ref_im, mask, _frames(tgt) if tgt else None)
            print(f"{d.name} {model}: {s}", flush=True)
            perf = d / f"perf_{model}.json"
            if perf.exists():
                s.update(json.loads(perf.read_text()))
            per_clip[d.name][model] = s
            per_model[model].append(s)
    summary = {}
    for model, rows in per_model.items():
        keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float))})
        summary[model] = {"clips": len(rows), **{k: round(sum(r[k] for r in rows if k in r) / sum(1 for r in rows if k in r), 4) for k in keys}}
    return {"summary": summary, "per_clip": per_clip, "masked": all(masked) if masked else False}


def to_markdown(result: dict) -> str:
    s = result["summary"]
    if not s:
        return "No results found.\n"
    cols = ["clips", "psnr_target", "colour_target", "change", "garment_colour", "garment_detail", "preservation",
            "flicker", "fps", "latency_ms_p50", "gpu_mem_gb"]
    cols = [c for c in cols if any(c in v for v in s.values())]
    lines = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for model, v in sorted(s.items()):
        lines.append(f"| {model} | " + " | ".join(str(v.get(c, "—")) for c in cols) + " |")
    lines += ["", "Higher is better for garment_colour, preservation, fps; garment_detail best near 1.0; "
              "lower is better for flicker, latency, memory. psnr_target / colour_target compare with the real "
              "ground truth (higher is better); change near 0 means the output is the input unchanged. "
              "Metrics are proxies: confirm with the blinded review."]
    if not result.get("masked", True):
        lines += ["", "No mask.png in some clips: garment_colour / garment_detail use the whole frame there, "
                  "and preservation is left out for them."]
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

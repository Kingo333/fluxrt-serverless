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
        keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float)) and not k.endswith("_worst_frame")})
        summary[model] = {"clips": len(rows), **{k: round(sum(r[k] for r in rows if k in r) / sum(1 for r in rows if k in r), 4) for k in keys}}
    return {"summary": summary, "per_clip": per_clip, "masked": all(masked) if masked else False}


def to_markdown(result: dict) -> str:
    s = result["summary"]
    if not s:
        return "No results found.\n"
    cols = ["clips", "psnr_target", "colour_target", "change", "garment_colour", "garment_detail", "preservation",
            "face_keep", "border_keep", "flicker", "fps", "latency_ms_p50", "gpu_mem_gb"]
    cols = [c for c in cols if any(c in v for v in s.values())]
    lines = ["| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for model, v in sorted(s.items()):
        lines.append(f"| {model} | " + " | ".join(str(v.get(c, "—")) for c in cols) + " |")
    lines += ["", "Higher is better for garment_colour, preservation, fps; garment_detail best near 1.0; "
              "lower is better for flicker, latency, memory. psnr_target / colour_target compare with the real "
              "ground truth (higher is better); change near 0 means the output is the input unchanged. "
              "Metrics are proxies: confirm with the blinded review."]
    fcols = [c for c in ("face_keep_min", "border_keep_min", "preservation_min", "garment_colour_min",
                         "garment_detail_min", "face_keep_drift", "border_keep_drift", "garment_colour_drift")
             if any(c in v for v in s.values())]
    if fcols:
        lines += ["", "Every frame (mean over clips of each clip's worst frame; drift = last chunk minus first "
                  "chunk of 4 frames, negative = worse over time):", "",
                  "| model | " + " | ".join(fcols) + " |", "|---|" + "---|" * len(fcols)]
        for model, v in sorted(s.items()):
            lines.append(f"| {model} | " + " | ".join(str(v.get(c, "—")) for c in fcols) + " |")
    if not result.get("masked", True):
        lines += ["", "No mask.png in some clips: garment_colour / garment_detail use the whole frame there, "
                  "and preservation is left out for them."]
    return "\n".join(lines) + "\n"


# A model is flagged when it is worse than the reference model by more than these amounts
# (owner requirement 2026-11-05: training must not change the person or the background).
KEEP_DROP = {"preservation": 0.01, "face_keep": 0.01, "border_keep": 0.01}
QUALITY_DROP = {"garment_colour": 0.02, "psnr_target": 0.5, "colour_target": 0.02}
FLICKER_RISE = 0.005
WORST_FRAME_DROP = 0.03   # a single frame worse than the reference's worst frame by this much
DRIFT_DROP = 0.03         # last chunk vs first chunk gets worse by this much more than the reference


def flags(result: dict, ref: str) -> list[str]:
    """Plain-language list of where each model is worse than `ref` (summary and per clip)."""
    s = result["summary"]
    if ref not in s:
        return [f"reference model {ref!r} not in the results"]
    out = []
    for model, v in sorted(s.items()):
        if model == ref:
            continue
        for k, lim in {**KEEP_DROP, **QUALITY_DROP}.items():
            if k in v and k in s[ref] and s[ref][k] - v[k] > lim:
                out.append(f"FLAG {model}: {k} {v[k]} vs {ref} {s[ref][k]} (drop > {lim})")
        if "flicker" in v and v["flicker"] - s[ref].get("flicker", v["flicker"]) > FLICKER_RISE:
            out.append(f"FLAG {model}: flicker {v['flicker']} vs {ref} {s[ref]['flicker']} (rise > {FLICKER_RISE})")
        for clip, per in sorted(result["per_clip"].items()):
            a, b = per.get(model), per.get(ref)
            if not (a and b):
                continue
            for k in KEEP_DROP:
                if k in a and k in b and b[k] - a[k] > 3 * KEEP_DROP[k]:
                    out.append(f"FLAG {model} clip {clip}: {k} {a[k]} vs {ref} {b[k]}")
            # every frame: the worst frame, and drift over the clip (owner 2026-11-05)
            for k in ("face_keep", "border_keep", "preservation", "garment_colour"):
                km = f"{k}_min"
                if km in a and km in b and b[km] - a[km] > WORST_FRAME_DROP:
                    out.append(f"FLAG {model} clip {clip}: worst-frame {k} {a[km]} at frame "
                               f"{a.get(k + '_worst_frame')} vs {ref} {b[km]}")
                kd = f"{k}_drift"
                if kd in a and kd in b and b[kd] - a[kd] > DRIFT_DROP:
                    out.append(f"FLAG {model} clip {clip}: {k} drifts {a[kd]} over the clip vs {ref} {b[kd]}")
    return out or [f"no flags: every model is within the limits of {ref}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default="eval_report.md")
    ap.add_argument("--ref", help="reference model (e.g. base, live0811): list where others are worse")
    a = ap.parse_args()
    r = evaluate(a.results_dir)
    md = to_markdown(r)
    if a.ref:
        r["flags"] = flags(r, a.ref)
        md += "\n## Flags vs " + a.ref + "\n\n" + "\n".join(f"- {f}" for f in r["flags"]) + "\n"
    Path(a.out).write_text(md)
    Path(a.out).with_suffix(".json").write_text(json.dumps(r, indent=2))
    print(md)


if __name__ == "__main__":
    main()

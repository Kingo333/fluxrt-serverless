"""Automatic face / skin / background gate for a FluxRT LoRA (owner rule, 2026-11-08).

  python -m eval.preserve_check RESULTS_DIR --base base --candidates l2000 l2000s07 [--out report.md]

The LoRA may only improve the garment. It FAILS if, on ANY clip, it is worse than base FluxRT on:
  face      mean change of the face (vs the camera input) in the input's face-skin area
  skin      shift of the average skin colour (face + body skin) vs the input
  background mean change of the background vs the input (pixels that are background in the input
            and in both outputs, so a longer garment covering background is not counted)
  bg_flicker (videos) frame-to-frame jitter of the background, beyond the input's own
  head      mean change of the person's own hair or head covering (hijab, scarf, cap): hair / clothes / other
            pixels of the INPUT around and above the face, down to the chin (owner, 2026-11-11: a dress or
            abaya must never change the shopper's hijab)
  head_lost (people wearing a head covering) share of the input's covering that turned into hair, face or
            skin in the output (0-1): the hijab removed or opened up
"Worse" = candidate > base * 1.10 + a small margin (face 1.5, skin 2.0, background 1.0, flicker 0.5, head 2.0
on a 0-255 scale; head_lost 0.05), so noise does not fail a clip. Areas come from MediaPipe's multiclass selfie
segmenter; face, skin and background are compared only where they are that class in the input AND in
every output (so a garment covering or uncovering skin is not counted as a skin change). Needs MediaPipe.
Each RESULTS_DIR/<clip>/ has input.mp4|png and out_<model>.mp4|png (eval.generate_fluxrt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BACKGROUND, HAIR, BODY_SKIN, FACE, CLOTHES, OTHERS = 0, 1, 2, 3, 4, 5
MARGIN = {"face": 1.5, "skin": 2.0, "background": 1.0, "bg_flicker": 0.5, "head": 2.0, "head_lost": 0.05}
RATIO = 1.10
MAX_FRAMES = 49
STRIDE = 4


def frames(path: Path):
    from PIL import Image
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        return [Image.open(path).convert("RGB")]
    import av
    out = []
    with av.open(str(path)) as c:
        for k, f in enumerate(c.decode(video=0)):
            if k >= MAX_FRAMES:
                break
            if k % STRIDE == 0:
                out.append(f.to_image().convert("RGB"))
    return out


def common_masks(cat_in, cats_out: list) -> tuple:
    """(face, skin, background) areas that are that class in the input AND in every output (2026-11-09 fix:
    before, face and skin came from the input only, so a new garment covering an arm, or a shorter sleeve
    uncovering one, counted as a skin colour change. Now only pixels that stay skin are compared)."""
    import numpy as np
    face = cat_in == FACE
    skin = np.isin(cat_in, (FACE, BODY_SKIN))
    bg = cat_in == BACKGROUND
    for c in cats_out:
        face &= c == FACE
        skin &= np.isin(c, (FACE, BODY_SKIN))
        bg &= c == BACKGROUND
    return face, skin, bg


def region_stats(inp, out, cat_in, bg_both, face_mask=None, skin_mask=None) -> dict:
    """Per-frame numbers for one output frame against its input frame (numpy arrays HxWx3, 0-255).
    face_mask / skin_mask: the areas to compare (from common_masks); without them, the input's own areas."""
    import numpy as np
    d = np.abs(out.astype(np.float32) - inp.astype(np.float32)).mean(axis=2)
    face = (cat_in == FACE) if face_mask is None else face_mask
    skin = np.isin(cat_in, (FACE, BODY_SKIN)) if skin_mask is None else skin_mask
    r = {"face": float(d[face].mean()) if face.sum() > 50 else None,
         "background": float(d[bg_both].mean()) if bg_both.sum() > 200 else None, "skin": None}
    if skin.sum() > 50:
        r["skin"] = float(np.linalg.norm(out[skin].astype(np.float32).mean(0) - inp[skin].astype(np.float32).mean(0)))
    return r


def head_zone(cat):
    """(head, cover) masks of the INPUT frame, or (None, None) without a face: head = hair, clothes or other
    pixels around and above the face down to the chin (the person's own hair or head covering); cover = the
    clothes / other part of it when it outweighs the hair there, i.e. a hijab, scarf or cap, else None."""
    import numpy as np
    face = cat == FACE
    if face.sum() < max(80, 0.002 * face.size):
        return None, None
    ys, xs = np.nonzero(face)
    top, bot, left, right = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    fh, fw = bot - top + 1, right - left + 1
    above = np.nonzero((cat[:top, left:right + 1] != BACKGROUND).any(axis=1))[0]
    if above.size and (int(above.min()) == 0 or top - int(above.min()) > 1.5 * fh):
        # the person runs off the top of the frame, or goes on far above this "face": the head is not in view
        # (a hand, or chest skin in a coat's neckline, taken for a face), so there is no head to check
        return None, None
    zone = np.zeros_like(face)
    zone[max(0, int(top - 0.9 * fh)): bot + 1, max(0, int(left - 0.6 * fw)): min(cat.shape[1], int(right + 0.6 * fw) + 1)] = True
    head = zone & np.isin(cat, (HAIR, CLOTHES, OTHERS))
    cover = zone & np.isin(cat, (CLOTHES, OTHERS))
    if cover.sum() < 0.3 * face.sum() or cover.sum() <= (zone & (cat == HAIR)).sum():
        cover = None
    return (head if head.sum() > 50 else None), cover


def head_stats(inp, out, cat_in, cat_out) -> dict:
    """head (0-255 mean change on the input's hair / head covering), head_lost (0-1, covering pixels that became
    hair, face or skin in the output; only when the input person wears one) and head_covered (1 or 0)."""
    import numpy as np
    head, cover = head_zone(cat_in)
    r = {"head": None, "head_lost": None, "head_covered": None}
    if head is not None:
        d = np.abs(out.astype(np.float32) - inp.astype(np.float32)).mean(axis=2)
        r["head"] = float(d[head].mean())
        r["head_covered"] = 0.0
    if cover is not None:
        r["head_covered"] = 1.0
        r["head_lost"] = float(np.isin(cat_out[cover], (HAIR, FACE, BODY_SKIN)).mean())
    return r


def worse(metric: str, cand: float | None, base: float | None) -> bool:
    if cand is None or base is None:
        return False
    return cand > base * RATIO + MARGIN[metric]


def clip_scores(d: Path, models: list[str], seg) -> dict:
    import numpy as np
    inp_p = next((d / f"input{e}" for e in (".mp4", ".png") if (d / f"input{e}").exists()), None)
    if inp_p is None:
        return {}
    ins = frames(inp_p)
    outs = {}
    for m in models:
        p = next((d / f"out_{m}{e}" for e in (".mp4", ".png") if (d / f"out_{m}{e}").exists()), None)
        if p is None:
            return {}
        outs[m] = frames(p)
    n = min([len(ins)] + [len(v) for v in outs.values()])
    if n == 0:
        return {}
    acc = {m: {"face": [], "skin": [], "background": [], "bg_flicker": [], "head": [], "head_lost": [], "head_covered": []}
           for m in models}
    prev = {m: None for m in models}
    prev_in = None
    for k in range(n):
        size = outs[models[0]][k].size
        fi = np.asarray(ins[k].resize(size))
        cat_in = seg(ins[k].resize(size))
        cats_out = {m: seg(outs[m][k]) for m in models}
        face_both, skin_both, bg_both = common_masks(cat_in, list(cats_out.values()))
        for m in models:
            fo = np.asarray(outs[m][k])
            for key, v in region_stats(fi, fo, cat_in, bg_both, face_both, skin_both).items():
                if v is not None:
                    acc[m][key].append(v)
            for key, v in head_stats(fi, fo, cat_in, cats_out[m]).items():
                if v is not None:
                    acc[m][key].append(v)
            if prev[m] is not None and prev_in is not None and bg_both.sum() > 200:
                jit = np.abs(fo.astype(np.float32) - prev[m]).mean(axis=2)[bg_both].mean()
                jit_in = np.abs(fi.astype(np.float32) - prev_in).mean(axis=2)[bg_both].mean()
                acc[m]["bg_flicker"].append(float(max(0.0, jit - jit_in)))
            prev[m] = fo.astype(np.float32)
        prev_in = fi.astype(np.float32)
    return {m: {k: (sum(v) / len(v) if v else None) for k, v in acc[m].items()} for m in models}


def judge(scores: dict, base: str, cand: str) -> list[str]:
    """Metrics on which the candidate is worse than base on this clip."""
    return [k for k in MARGIN if worse(k, scores[cand].get(k), scores[base].get(k))]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--base", default="base")
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--model", default=str(Path.home() / "selfie_multiclass_256x256.tflite"))
    a = ap.parse_args(argv)
    from data_tools.clothes_mask import Segmenter
    seg = Segmenter(Path(a.model))
    root = Path(a.results)
    verdict = {c: {"fails": [], "clips": 0} for c in a.candidates}
    per_clip = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        present = [c for c in a.candidates if any((d / f"out_{c}{e}").exists() for e in (".mp4", ".png"))]
        if not present or not any((d / f"out_{a.base}{e}").exists() for e in (".mp4", ".png")):
            continue
        sc = clip_scores(d, [a.base] + present, seg)
        if not sc:
            continue
        per_clip[d.name] = sc
        for c in present:
            verdict[c]["clips"] += 1
            bad = judge(sc, a.base, c)
            if bad:
                verdict[c]["fails"].append({"clip": d.name, "worse_on": bad})
    result = {c: {"pass": not v["fails"] and v["clips"] > 0, **v} for c, v in verdict.items()}
    (root / "preserve_check.json").write_text(json.dumps({"verdict": result, "per_clip": per_clip}, indent=2))
    lines = [f"# Face / skin / background gate ({root.name})", "", "| candidate | clips | result | worse on |", "|---|---|---|---|"]
    for c, v in result.items():
        worst = "; ".join(f"{f['clip']}: {', '.join(f['worse_on'])}" for f in v["fails"][:6]) or "-"
        lines.append(f"| {c} | {v['clips']} | {'PASS' if v['pass'] else 'FAIL'} | {worst} |")
    lines += ["", "## Numbers per clip (base -> candidate; lower = closer to the camera input)", "",
              "| clip | candidate | face | skin | background | bg_flicker | head | head_lost |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    for clip, sc in per_clip.items():
        for c in a.candidates:
            if c not in sc:
                continue
            cells = [f"{fmt(sc[a.base].get(k))} -> {fmt(sc[c].get(k))}" + (" **worse**" if worse(k, sc[c].get(k), sc[a.base].get(k)) else "")
                     for k in ("face", "skin", "background", "bg_flicker", "head", "head_lost")]
            lines.append(f"| {clip} | {c} | " + " | ".join(cells) + " |")
    covered = {clip: sc for clip, sc in per_clip.items() if (sc[a.base].get("head_covered") or 0) >= 0.5}
    lines += ["", "## Head covering (clips where the camera person wears one: hijab, scarf, cap)", ""]
    if covered:
        lines += ["| clip | model | covering turned into hair/face/skin | head change (0-255) |", "|---|---|---|---|"]
        for clip, sc in covered.items():
            for m in [a.base] + [c for c in a.candidates if c in sc]:
                lost = sc[m].get("head_lost")
                lines.append(f"| {clip} | {m} | {'-' if lost is None else f'{lost * 100:.0f}%'} | {fmt(sc[m].get('head'))} |")
    else:
        lines.append("(no clip with a head covering)")
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return result


if __name__ == "__main__":
    main()

"""Face and skin numbers for rendered clips (fxreal6): favourite #2's baseline first, then the A/B.

  python -m eval.face_skin_report RESULTS --models fav2_r528 [fx6_r528 ...] --baseline fav2_r528 \\
      --model SEG.tflite --out report.md [--leak-pairs]

RESULTS/<clip>/ has input.mp4 and out_<model>.mp4 (eval.generate_fluxrt). Every number compares an
output frame with the camera frame it was made from, on the pixels that are face / skin in both:

  face_match   brightness correlation on the face (1 = the same face); the "90% true" number on close-ups
  face_px      mean face difference, 0-255
  face_drift   face change from one output frame to the next, beyond the camera's own
  skin_de      skin tone difference (CIEDE2000; about 1 = just noticeable)
  skin_dl      lightness shift (+ lighter, - darker)
  texture      skin fine detail, output / camera (under 1 = smoothed)

--leak-pairs: RESULTS holds <key>_clean and <key>_logo clips (eval.logo_set); prints the logo leak.
Prints FACESKIN lines, writes a markdown table and RESULTS/face_skin.json. Runs in the MediaPipe venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import face_skin as fs  # noqa: E402

FACE, BODY_SKIN, CLOTHES = 3, 2, 4
KEYS = ("face_match", "face_px", "face_drift", "skin_de", "skin_dl", "skin_dl_abs", "texture")


def frame_numbers(cam, out, cat_cam, cat_out, prev=None) -> tuple[dict, dict]:
    """Numbers for one frame, and what the next frame needs (the face pixels of this one)."""
    import numpy as np
    face = (cat_cam == FACE) & (cat_out == FACE)
    skin = np.isin(cat_cam, (FACE, BODY_SKIN)) & np.isin(cat_out, (FACE, BODY_SKIN))
    r: dict = {"closeup": fs.face_width(cat_cam == FACE) >= fs.CLOSEUP_FACE_PX * cam.shape[1] / 288}
    r["face_match"] = fs.face_ncc(cam, out, face)
    r["face_px"] = fs.face_pixel_error(cam, out, face)
    s = fs.skin_frame(cam, out, skin)
    if s:
        r.update(skin_de=s["de"], skin_dl=s["dl"], texture=s["texture"])
    if prev is not None:
        both = face & prev["face"]
        if both.sum() >= fs.MIN_PX:
            d_out = np.abs(out.astype(np.float32) - prev["out"]).mean(axis=2)[both].mean()
            d_cam = np.abs(cam.astype(np.float32) - prev["cam"]).mean(axis=2)[both].mean()
            r["face_drift"] = float(max(0.0, d_out - d_cam))
    return r, {"face": face, "out": out.astype(np.float32), "cam": cam.astype(np.float32)}


def clip_numbers(cams: list, outs: list, seg) -> dict:
    """cams / outs: PIL images of one clip (camera, one model). seg(PIL) -> category map."""
    import numpy as np
    rows, prev = [], None
    for cam_im, out_im in zip(cams, outs):
        cam_im = cam_im.resize(out_im.size)
        r, prev = frame_numbers(np.asarray(cam_im), np.asarray(out_im), np.squeeze(seg(cam_im)), np.squeeze(seg(out_im)), prev)
        rows.append(r)
    return fs.clip_summary(rows)


def leak_numbers(d_clean: Path, d_logo: Path, model: str, seg, frames) -> float | None:
    import numpy as np
    paths = [d_clean / "input.mp4", d_logo / "input.mp4", d_clean / f"out_{model}.mp4", d_logo / f"out_{model}.mp4"]
    if not all(p.exists() for p in paths):
        return None
    ic, il, oc, ol = (frames(p) for p in paths)
    vals = []
    for a, b, c, d in zip(ic, il, oc, ol):
        garment = np.squeeze(seg(a.resize(c.size))) == CLOTHES
        v = fs.leak_ratio(np.asarray(a.resize(c.size)), np.asarray(b.resize(c.size)), np.asarray(c), np.asarray(d), garment)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def table(per_clip: dict, models: list[str]) -> list[str]:
    lines = ["| clip | model | " + " | ".join(KEYS) + " | close-up |", "|---|---|" + "---|" * (len(KEYS) + 1)]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    for clip, sc in sorted(per_clip.items()):
        for m in models:
            if m in sc:
                lines.append(f"| {clip} | {m} | " + " | ".join(fmt(sc[m].get(k)) for k in KEYS) + f" | {'yes' if sc[m].get('closeup') else 'no'} |")
    return lines


def averages(per_clip: dict, model: str) -> dict:
    import numpy as np
    out = {}
    for k in KEYS:
        v = [sc[model][k] for sc in per_clip.values() if model in sc and sc[model].get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    close = [sc[model]["face_match"] for sc in per_clip.values()
             if model in sc and sc[model].get("closeup") and sc[model].get("face_match") is not None]
    out["face_match_closeup"] = float(np.mean(close)) if close else None
    out["clips"] = sum(1 for sc in per_clip.values() if model in sc)
    out["closeup_clips"] = len(close)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", help="the model to beat (default: the first of --models)")
    ap.add_argument("--model", default="/workspace/azyah-vton/selfie_multiclass_256x256.tflite")
    ap.add_argument("--out")
    ap.add_argument("--leak-pairs", action="store_true")
    a = ap.parse_args(argv)
    from data_tools.clothes_mask import Segmenter
    from eval.preserve_check import frames
    seg = Segmenter(Path(a.model))
    root, base = Path(a.results), a.baseline or a.models[0]
    per_clip: dict = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "input.mp4").exists():
            continue
        cams = frames(d / "input.mp4")
        sc = {m: clip_numbers(cams, frames(d / f"out_{m}.mp4"), seg) for m in a.models if (d / f"out_{m}.mp4").exists()}
        if sc:
            per_clip[d.name] = sc
    result: dict = {"per_clip": per_clip, "average": {m: averages(per_clip, m) for m in a.models}, "verdict": {}, "leak": {}}
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    for m in a.models:
        av = result["average"][m]
        print(f"FACESKIN {root.name} {m}: clips {av['clips']} (close-ups {av['closeup_clips']}); face match {fmt(av['face_match'])} "
              f"(close-ups {fmt(av['face_match_closeup'])}); face px {fmt(av['face_px'])}; drift {fmt(av['face_drift'])}; "
              f"skin tone {fmt(av['skin_de'])}; lightness {fmt(av['skin_dl'])} (abs {fmt(av['skin_dl_abs'])}); texture {fmt(av['texture'])}")
        if m != base:
            fails = {c: fs.judge(sc[m], sc[base]) for c, sc in per_clip.items() if m in sc and base in sc}
            fails = {c: f for c, f in fails.items() if f}
            result["verdict"][m] = {"pass": not fails, "fails": fails}
            print(f"FACESKIN_GATE {root.name} {m}: {'PASS' if not fails else 'FAIL'}" + ("" if not fails else f"; {json.dumps(fails)}"))
    if a.leak_pairs:
        import numpy as np
        for m in a.models:
            vals = {}
            for d in sorted(root.glob("*_clean")):
                v = leak_numbers(d, root / (d.name[:-6] + "_logo"), m, seg, frames)
                if v is not None:
                    vals[d.name[:-6]] = v
            result["leak"][m] = {"per_clip": vals, "mean": float(np.mean(list(vals.values()))) if vals else None}
            print(f"LOGO_LEAK {root.name} {m}: {fmt(result['leak'][m]['mean'])} over {len(vals)} clips (0 = logo gone, 1 = fully kept)")
    (root / "face_skin.json").write_text(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).write_text("\n".join([f"# Face and skin: {root.name}", ""] + table(per_clip, a.models)) + "\n")
    return result


if __name__ == "__main__":
    main()

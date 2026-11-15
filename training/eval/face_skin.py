"""Face and skin measures for fxreal6 (owner, 2026-11-15): the face redrawn by the model must stay the
person's own (about 90% true), and skin must keep its tone, texture and lighting.

Truth is the camera frame: outside the garment the model is asked to change nothing, so every number
compares an output frame with the camera frame it was made from. Pure numpy here (unit-tested); the
runner that segments frames and loads a face model sits in eval.preserve_check / the fx6 script.

  ciede2000(lab1, lab2)            colour difference the eye sees (about 1 = just noticeable, 2-3 = small)
  skin_frame(cam, out, mask)       tone difference, signed lightness shift, texture ratio for one area
  face_pixel_error(cam, out, m)    mean absolute difference on the face, 0-255
  face_match(sim, ceiling)         share of the camera's own score (1.0 = as alike as two camera frames)
  judge(cand, base)                the pass bar against favourite #2, per clip
"""

from __future__ import annotations

import numpy as np

from .garment_report import srgb_to_lab

# Pass bars (FXREAL6_PLAN.md). Absolute targets are confirmed once favourite #2's own numbers are in.
TARGET = {"skin_de": 3.0, "skin_dl": 2.0, "texture_lo": 0.85, "texture_hi": 1.15, "face_match_closeup": 0.90}
TOLERANCE = 1.02            # "no worse than favourite #2" allows 2%
CLOSEUP_FACE_PX = 80        # a face at least this wide counts as a close-up
MIN_PX = 50


def ciede2000(lab1, lab2) -> float:
    """CIEDE2000 colour difference between two Lab colours (Sharma, Wu, Dalal 2005)."""
    l1, a1, b1 = (float(x) for x in lab1)
    l2, a2, b2 = (float(x) for x in lab2)
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360 if c1p else 0.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360 if c2p else 0.0
    dl, dc = l2 - l1, c2p - c1p
    if c1p * c2p == 0:
        dh = 0.0
    else:
        dh = h2p - h1p
        dh = dh - 360 if dh > 180 else dh + 360 if dh < -180 else dh
    dhp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    l_bar, cp_bar = (l1 + l2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        h_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_bar = (h1p + h2p) / 2
    else:
        h_bar = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (1 - 0.17 * np.cos(np.radians(h_bar - 30)) + 0.24 * np.cos(np.radians(2 * h_bar))
         + 0.32 * np.cos(np.radians(3 * h_bar + 6)) - 0.20 * np.cos(np.radians(4 * h_bar - 63)))
    sl = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    sc = 1 + 0.045 * cp_bar
    sh = 1 + 0.015 * cp_bar * t
    rt = (-2 * np.sqrt(cp_bar ** 7 / (cp_bar ** 7 + 25.0 ** 7))
          * np.sin(np.radians(60 * np.exp(-(((h_bar - 275) / 25) ** 2)))))
    return float(np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dhp / sh) ** 2 + rt * (dc / sc) * (dhp / sh)))


def _median_lab(rgb: np.ndarray, mask: np.ndarray):
    px = rgb[mask]
    return np.median(srgb_to_lab(px), axis=0) if len(px) >= MIN_PX else None


def texture(rgb: np.ndarray, mask: np.ndarray) -> float | None:
    """Fine detail inside the area: spread of the brightness Laplacian (pores, hair, grain). Smoothing lowers it."""
    if mask.sum() < MIN_PX:
        return None
    g = rgb.astype(np.float64).mean(axis=2)
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    inner = mask.copy()
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    # only pixels whose four neighbours are in the area too, so the area's edge is not counted as detail
    inner[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    return float(lap[inner].std()) if inner.sum() >= MIN_PX else None


def skin_frame(cam: np.ndarray, out: np.ndarray, mask: np.ndarray) -> dict | None:
    """One skin area (face, or neck/arms/hands) in one frame. mask: skin in the camera frame AND the output.
    de: tone difference (CIEDE2000); dl: lightness shift, + = lighter, - = darker; texture: output / camera."""
    a, b = _median_lab(cam, mask), _median_lab(out, mask)
    if a is None or b is None:
        return None
    ta, tb = texture(cam, mask), texture(out, mask)
    return {"de": ciede2000(a, b), "dl": float(b[0] - a[0]),
            "texture": float(tb / ta) if ta and tb is not None and ta > 1e-6 else None}


def face_pixel_error(cam: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float | None:
    if mask.sum() < MIN_PX:
        return None
    return float(np.abs(out.astype(np.float64) - cam.astype(np.float64)).mean(axis=2)[mask].mean())


def face_width(mask: np.ndarray) -> int:
    cols = np.nonzero(mask.any(axis=0))[0]
    return int(cols.max() - cols.min() + 1) if len(cols) else 0


def face_match(similarity: float | None, ceiling: float | None) -> float | None:
    """Output-to-camera face similarity as a share of the camera's own (camera frame against the next camera
    frame of the same person). 0.90 = "90% true". Capped at 1."""
    if similarity is None or not ceiling or ceiling <= 0:
        return None
    return float(max(0.0, min(1.0, similarity / ceiling)))


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0


def clip_summary(frames: list[dict]) -> dict:
    """Mean of each number over a clip's frames (None values skipped). Keys: skin_de, skin_dl, texture,
    face_px, face_match, face_drift, closeup (bool per frame)."""
    out: dict = {}
    for k in ("skin_de", "skin_dl", "texture", "face_px", "face_match", "face_drift"):
        v = [f[k] for f in frames if f.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    dl = [abs(f["skin_dl"]) for f in frames if f.get("skin_dl") is not None]
    out["skin_dl_abs"] = float(np.mean(dl)) if dl else None
    out["closeup"] = bool(frames) and sum(1 for f in frames if f.get("closeup")) * 2 >= len(frames)
    return out


def judge(cand: dict, base: dict) -> list[str]:
    """Why a clip fails the bar against favourite #2 (empty list = pass). cand / base: clip_summary()."""
    fails = []

    def worse_low(key, label):  # lower is better
        c, b = cand.get(key), base.get(key)
        if c is not None and b is not None and c > b * TOLERANCE + 1e-9:
            fails.append(f"{label} worse than favourite #2 ({c:.2f} vs {b:.2f})")

    worse_low("skin_de", "skin tone")
    worse_low("face_px", "face")
    worse_low("face_drift", "face drift")
    c, b = cand.get("face_match"), base.get("face_match")
    if c is not None and b is not None and c < b / TOLERANCE - 1e-9:
        fails.append(f"face match worse than favourite #2 ({c:.2f} vs {b:.2f})")
    if cand.get("closeup") and c is not None and c < TARGET["face_match_closeup"]:
        fails.append(f"close-up face match {c:.2f} under {TARGET['face_match_closeup']:.2f}")
    t = cand.get("texture")
    if t is not None and not TARGET["texture_lo"] <= t <= TARGET["texture_hi"]:
        tb = base.get("texture")
        if tb is None or abs(t - 1) > abs(tb - 1) * TOLERANCE:  # outside the band AND no closer than favourite #2
            fails.append(f"skin texture ratio {t:.2f} outside {TARGET['texture_lo']}-{TARGET['texture_hi']}")
    return fails


def by_group(clips: dict[str, dict], groups: dict[str, str], key: str = "skin_de") -> dict:
    """Average of one number per skin-tone group, and which groups are worse than 1.5x the overall average."""
    vals = {c: s[key] for c, s in clips.items() if s.get(key) is not None}
    if not vals:
        return {"overall": None, "groups": {}, "unfair": []}
    overall = float(np.mean(list(vals.values())))
    per: dict[str, list] = {}
    for c, v in vals.items():
        per.setdefault(groups.get(c, "unknown"), []).append(v)
    means = {g: float(np.mean(v)) for g, v in per.items()}
    return {"overall": overall, "groups": means, "unfair": sorted(g for g, m in means.items() if m > 1.5 * overall + 1e-9)}


def face_ncc(cam: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float | None:
    """How alike the face is in structure: correlation of brightness inside the face area between the
    output and the camera frame it was made from (same pose, so same pixels). 1 = the same face,
    whatever the lighting level; 0 = unrelated. This is the "90% true" number (0.90) on close-ups."""
    if mask.sum() < MIN_PX:
        return None
    a = cam.astype(np.float64).mean(axis=2)[mask]
    b = out.astype(np.float64).mean(axis=2)[mask]
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-9 else None


def leak_ratio(in_clean: np.ndarray, in_marked: np.ndarray, out_clean: np.ndarray, out_marked: np.ndarray,
               garment: np.ndarray, min_contrast: float = 30.0) -> float | None:
    """How much of a logo on the shopper's own garment survives into the try-on (0 = gone, 1 = fully kept).
    The same clip is rendered twice, once with a mark stamped on the input garment (data_tools.logo_stamp)
    and once without. Where the two inputs differ is the mark; the two outputs should not differ there any
    more than they differ on the rest of the garment."""
    d_in = np.abs(in_marked.astype(np.float64) - in_clean.astype(np.float64)).mean(axis=2)
    mark = (d_in > min_contrast) & garment
    rest = garment & ~mark
    if mark.sum() < MIN_PX or rest.sum() < MIN_PX:
        return None
    d_out = np.abs(out_marked.astype(np.float64) - out_clean.astype(np.float64)).mean(axis=2)
    return float(max(0.0, d_out[mark].mean() - d_out[rest].mean()) / d_in[mark].mean())


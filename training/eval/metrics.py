"""Cheap, explainable try-on metrics (CPU, numpy). They are proxies; the blinded
human review (ab_sheet.py) is the final judge. Frames: lists of PIL images.
Mask: white = garment area (from data_tools.masks / dataset masks).

- garment_colour: how close the output garment's colours are to the garment photo
  (1 - histogram distance in a coarse RGB cube, 0..1, higher is better)
- garment_detail: how much fine detail (prints, logos, texture) the output garment keeps
  relative to the garment photo (edge-energy ratio, 1.0 = same, lower = blurred)
- preservation: how unchanged everything outside the garment is vs the input
  (1 - mean abs change outside the mask, higher is better)
- flicker: extra frame-to-frame change in the garment area beyond what the input
  itself moved (lower is better; 0 = as steady as the input)

With the real ground truth (target = the same person really wearing the garment):
- psnr_target: whole-frame PSNR of the output vs the target, in dB (higher is better)
- colour_target: colour-histogram match of the output vs the target (0..1, higher is better)
- change: mean abs change of the output vs the input (0..1). Near 0 = the model did nothing.

Person kept intact (owner requirement 2026-11-05: background, face, skin and body must not change):
- preservation: outside the garment mask (only when the clip has mask.png)
- face_keep: 1 - mean abs change inside the face box found on the first input frame (OpenCV's
  bundled frontal-face detector; left out when OpenCV is missing or no face is found)
- border_keep: 1 - mean abs change in the left and right 12% of the frame (outside the garment
  mask when there is one): a background check that also works without a mask

Every frame, not just the average (owner 2026-11-05): for face_keep, border_keep, preservation,
garment_colour and garment_detail the result also has <name>_min (worst frame) and
<name>_worst_frame (its index), and <name>_drift = mean of the last chunk minus mean of the first
chunk (CHUNK = 4 frames, JoyAI's streaming chunk; negative = gets worse over the clip).

Without a mask the "garment area" is the whole frame: garment_colour and garment_detail then
compare the full frame with the garment photo, and preservation has no outside area, so
score_clip leaves preservation out instead of reporting a meaningless 1.0.
"""

from __future__ import annotations

import numpy as np

BINS = 8


def _arr(im, size=None):
    im = im.convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size)
    return np.asarray(im, dtype=np.float32) / 255.0


def _mask(mask, size):
    if mask is None:
        return np.ones(size[::-1], dtype=bool)
    return np.asarray(mask.convert("L").resize(size), dtype=np.float32) / 255.0 > 0.5


def _hist(pixels: np.ndarray) -> np.ndarray:
    if len(pixels) == 0:
        return np.full(BINS ** 3, 1.0 / BINS ** 3)
    idx = np.clip((pixels * BINS).astype(int), 0, BINS - 1)
    flat = idx[:, 0] * BINS * BINS + idx[:, 1] * BINS + idx[:, 2]
    h = np.bincount(flat, minlength=BINS ** 3).astype(np.float64)
    return h / h.sum()


def _garment_pixels_of_photo(ref) -> np.ndarray:
    """Garment photos are on white/plain backgrounds: drop near-white pixels."""
    a = _arr(ref).reshape(-1, 3)
    keep = a.min(axis=1) < 0.92
    return a[keep] if keep.sum() > 100 else a


def _edge_energy(a: np.ndarray, m: np.ndarray | None = None) -> float:
    # Colour edges on all channels: a red/blue print can have no brightness edges at all.
    gx = np.abs(np.diff(a, axis=1)).mean(axis=2)[:-1, :]
    gy = np.abs(np.diff(a, axis=0)).mean(axis=2)[:, :-1]
    e = gx + gy
    if m is not None:
        mm = m[:-1, :-1]
        return float(e[mm].mean()) if mm.any() else 0.0
    return float(e.mean())


def garment_colour(output_frames, ref, mask) -> float:
    size = output_frames[0].size
    m = _mask(mask, size)
    ref_h = _hist(_garment_pixels_of_photo(ref))
    scores = []
    for f in output_frames:
        out_h = _hist(_arr(f, size)[m])
        scores.append(1.0 - 0.5 * float(np.abs(out_h - ref_h).sum()))
    return float(np.mean(scores))


def garment_detail(output_frames, ref, mask) -> float:
    size = output_frames[0].size
    m = _mask(mask, size)
    ref_a = _arr(ref)
    ref_m = ref_a.min(axis=2) < 0.92
    ref_e = _edge_energy(ref_a, ref_m if ref_m.sum() > 100 else None)
    if ref_e <= 1e-6:
        return 1.0
    out_e = np.mean([_edge_energy(_arr(f, size), m) for f in output_frames])
    return float(min(out_e / ref_e, 2.0))


def preservation(input_frames, output_frames, mask) -> float:
    size = output_frames[0].size
    outside = ~_mask(mask, size)
    if not outside.any():
        return 1.0
    diffs = [np.abs(_arr(i, size) - _arr(o, size)).mean(axis=2)[outside].mean()
             for i, o in zip(input_frames, output_frames)]
    return float(1.0 - np.mean(diffs))


def flicker(input_frames, output_frames, mask) -> float:
    if len(output_frames) < 2:
        return 0.0
    size = output_frames[0].size
    m = _mask(mask, size)
    def motion(frames):
        a = [_arr(f, size) for f in frames]
        return np.array([np.abs(a[k + 1] - a[k]).mean(axis=2)[m].mean() for k in range(len(a) - 1)])
    return float(max(0.0, (motion(output_frames) - motion(input_frames)).mean()))


def psnr_target(output_frames, target_frames) -> float:
    size = output_frames[0].size
    vals = []
    for o, t in zip(output_frames, target_frames):
        mse = float(np.mean((_arr(o, size) - _arr(t, size)) ** 2))
        vals.append(10.0 * np.log10(1.0 / max(mse, 1e-10)))
    return float(np.mean(vals))


def colour_target(output_frames, target_frames) -> float:
    size = output_frames[0].size
    scores = [1.0 - 0.5 * float(np.abs(_hist(_arr(o, size).reshape(-1, 3)) - _hist(_arr(t, size).reshape(-1, 3))).sum())
              for o, t in zip(output_frames, target_frames)]
    return float(np.mean(scores))


def change(input_frames, output_frames) -> float:
    size = output_frames[0].size
    return float(np.mean([np.abs(_arr(i, size) - _arr(o, size)).mean() for i, o in zip(input_frames, output_frames)]))


_CASCADE = None


def face_box(rgb_float):
    """Largest frontal face (x, y, w, h) in an HxWx3 0..1 image, or None (no OpenCV / no face)."""
    global _CASCADE
    try:
        import cv2
    except ImportError:
        return None
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor((rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    side = min(gray.shape)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(max(12, side // 20),) * 2)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def border_mask(shape_hw, garment=None, frac: float = 0.12) -> np.ndarray:
    h, w = shape_hw
    b = np.zeros((h, w), dtype=bool)
    k = max(1, int(round(w * frac)))
    b[:, :k] = True
    b[:, w - k:] = True
    if garment is not None:
        b &= ~garment
    return b


CHUNK = 4
PER_FRAME = ("face_keep", "border_keep", "preservation", "garment_colour", "garment_detail")


def frame_stats(values: list[float], name: str, chunk: int = CHUNK) -> dict:
    """Worst frame and first-to-last-chunk drift of one per-frame series. garment_detail's best
    value is 1.0, so its 'worst' is the frame farthest from 1."""
    if not values:
        return {}
    if name == "garment_detail":
        worst = max(range(len(values)), key=lambda k: abs(values[k] - 1.0))
    else:
        worst = min(range(len(values)), key=lambda k: values[k])
    out = {f"{name}_min": round(values[worst], 4), f"{name}_worst_frame": worst}
    if len(values) >= 2 * chunk:
        out[f"{name}_drift"] = round(float(np.mean(values[-chunk:]) - np.mean(values[:chunk])), 4)
    return out


class ClipScorer:
    """The same metrics as the functions above, added up one frame at a time, so a clip never has
    to be in memory (the CPU Pod has a 512 MB limit). Call add() per frame, then result()."""

    def __init__(self, ref, mask):
        self.ref_h = _hist(_garment_pixels_of_photo(ref))
        ref_a = _arr(ref)
        ref_m = ref_a.min(axis=2) < 0.92
        self.ref_e = _edge_energy(ref_a, ref_m if ref_m.sum() > 100 else None)
        self.mask_img, self.size, self.m = mask, None, None
        self.n = self.n_t = 0
        self.colour = self.edge = self.change = self.pres = self.flick = 0.0
        self.psnr = self.col_t = 0.0
        self.prev = None
        self.face, self.face_d, self.border_d, self.border = None, 0.0, 0.0, None
        self.series: dict[str, list[float]] = {k: [] for k in PER_FRAME}

    def add(self, inp, out, tgt=None) -> None:
        if self.size is None:
            self.size = out.size
            self.m = _mask(self.mask_img, self.size)
        i, o = _arr(inp, self.size), _arr(out, self.size)
        m = self.m
        if self.border is None:
            self.face = face_box(i) or False
            self.border = border_mask(i.shape[:2], m if self.mask_img is not None else None)
        diff = np.abs(i - o).mean(axis=2)
        if self.face:
            x, y, w, h = self.face
            fd = float(diff[y:y + h, x:x + w].mean())
            self.face_d += fd
            self.series["face_keep"].append(1.0 - fd)
        if self.border.any():
            bd = float(diff[self.border].mean())
            self.border_d += bd
            self.series["border_keep"].append(1.0 - bd)
        col = 1.0 - 0.5 * float(np.abs(_hist(o[m]) - self.ref_h).sum())
        self.colour += col
        self.series["garment_colour"].append(col)
        e = _edge_energy(o, m)
        self.edge += e
        self.series["garment_detail"].append(1.0 if self.ref_e <= 1e-6 else min(e / self.ref_e, 2.0))
        self.change += float(np.abs(i - o).mean())
        outside = ~m
        if self.mask_img is not None and outside.any():
            pd = float(diff[outside].mean())
            self.pres += pd
            self.series["preservation"].append(1.0 - pd)
        if self.prev is not None:
            pi, po = self.prev
            self.flick += float(np.abs(o - po).mean(axis=2)[m].mean() - np.abs(i - pi).mean(axis=2)[m].mean())
        self.prev = (i, o)
        if tgt is not None:
            t = _arr(tgt, self.size)
            self.psnr += 10.0 * np.log10(1.0 / max(float(np.mean((o - t) ** 2)), 1e-10))
            self.col_t += 1.0 - 0.5 * float(np.abs(_hist(o.reshape(-1, 3)) - _hist(t.reshape(-1, 3))).sum())
            self.n_t += 1
        self.n += 1

    def result(self) -> dict:
        if not self.n:
            return {}
        n = self.n
        detail = 1.0 if self.ref_e <= 1e-6 else min((self.edge / n) / self.ref_e, 2.0)
        out = {
            "garment_colour": round(self.colour / n, 4),
            "garment_detail": round(float(detail), 4),
            "flicker": round(max(0.0, self.flick / (n - 1)) if n > 1 else 0.0, 4),
            "change": round(self.change / n, 4),
        }
        if self.mask_img is not None:
            out["preservation"] = round(1.0 - self.pres / n, 4)
        if self.face:
            out["face_keep"] = round(1.0 - self.face_d / n, 4)
        if self.border is not None and self.border.any():
            out["border_keep"] = round(1.0 - self.border_d / n, 4)
        for k, vals in self.series.items():
            out.update(frame_stats(vals, k))
        if self.n_t:
            out["psnr_target"] = round(float(self.psnr / self.n_t), 3)
            out["colour_target"] = round(self.col_t / self.n_t, 4)
        return out


def score_clip(input_frames, output_frames, ref, mask, target_frames=None) -> dict:
    """Frames can be lists or iterators; the clip length is the shorter of input and output."""
    sc = ClipScorer(ref, mask)
    tg = iter(target_frames or [])
    for i, o in zip(input_frames, output_frames):
        sc.add(i, o, next(tg, None))
    return sc.result()

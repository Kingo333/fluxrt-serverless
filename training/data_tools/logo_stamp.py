"""Put a sharp logo, print or stripes on the garment the person wears in a training INPUT (fxreal6).

Why: live, the shopper's own shirt often carries a logo or print. Our training inputs were drawn by a
model and were soft, so the LoRA never practised erasing a crisp mark: a plain white or black tee came
out with the shopper's logo on it. With these inputs the answer is still the real picture (the real
garment, without the mark), so the model learns that everything on the old garment goes.

stamp(rgb, mask, rng) -> rgb with one mark inside the garment area only. The mark follows the
garment's own shading (folds stay visible through it), like a print on cloth. Kinds: a block logo, a
line of block "text", horizontal stripes, a big graphic patch. Pure numpy.
"""

from __future__ import annotations

import numpy as np

KINDS = ("logo", "text", "stripes", "graphic")


def _bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _contrast_colour(rgb: np.ndarray, mask: np.ndarray, rng) -> np.ndarray:
    """A colour that stands out from the garment: light on dark cloth, dark or saturated on light."""
    mean = rgb[mask].mean(axis=0)
    dark = mean.mean() < 128
    if rng.random() < 0.5:
        return np.array([240.0, 240.0, 240.0] if dark else [20.0, 20.0, 20.0])
    hue = np.array([[220, 40, 40], [40, 90, 220], [240, 200, 30], [30, 160, 90], [230, 110, 30]][int(rng.integers(5))], dtype=np.float64)
    return hue if abs(hue.mean() - mean.mean()) > 60 else (np.array([240.0, 240.0, 240.0]) if dark else np.array([20.0, 20.0, 20.0]))


def mark_mask(mask: np.ndarray, kind: str, rng) -> np.ndarray:
    """Where the mark goes (bool, same shape as mask, always inside it)."""
    h, w = mask.shape
    box = _bbox(mask)
    out = np.zeros((h, w), dtype=bool)
    if box is None:
        return out
    y0, y1, x0, x1 = box
    bh, bw = y1 - y0, x1 - x0
    if kind == "stripes":
        period = max(4, int(bh * rng.uniform(0.06, 0.14)))
        rows = (np.arange(h) - y0) % period < max(2, period // 2)
        out[rows, :] = True
    else:
        # chest area: upper middle of the garment
        cy = y0 + bh * rng.uniform(0.22, 0.45)
        cx = x0 + bw * rng.uniform(0.38, 0.62)
        if kind == "graphic":
            mh, mw = bh * rng.uniform(0.28, 0.42), bw * rng.uniform(0.38, 0.55)
        elif kind == "text":
            mh, mw = bh * rng.uniform(0.05, 0.09), bw * rng.uniform(0.35, 0.55)
        else:
            mh, mw = bh * rng.uniform(0.10, 0.18), bw * rng.uniform(0.14, 0.24)
        ya, yb = int(max(0, cy - mh / 2)), int(min(h, cy + mh / 2))
        xa, xb = int(max(0, cx - mw / 2)), int(min(w, cx + mw / 2))
        if kind == "text":  # separate "letters": blocks with gaps
            n = int(rng.integers(4, 9))
            edges = np.linspace(xa, xb, n + 1).astype(int)
            for a, b in zip(edges, edges[1:]):
                out[ya:yb, a:max(a + 1, b - max(1, (b - a) // 4))] = True
        elif kind == "graphic":  # an ellipse with a hole: a sharp, closed shape
            yy, xx = np.mgrid[0:h, 0:w]
            r = ((yy - cy) / max(1.0, mh / 2)) ** 2 + ((xx - cx) / max(1.0, mw / 2)) ** 2
            out[(r <= 1.0) & (r >= 0.25)] = True
        else:
            out[ya:yb, xa:xb] = True
    return out & mask


def stamp(rgb: np.ndarray, mask: np.ndarray, rng, kind: str | None = None, opacity: float = 0.92) -> tuple[np.ndarray, str]:
    """rgb: HxWx3 uint8; mask: HxW bool (the garment area). Returns (stamped rgb, kind). Nothing outside
    the garment changes; with an empty mask the picture comes back unchanged."""
    kind = kind or KINDS[int(rng.integers(len(KINDS)))]
    area = mark_mask(mask.astype(bool), kind, rng)
    out = rgb.astype(np.float64).copy()
    if not area.any():
        return rgb.copy(), kind
    colour = _contrast_colour(rgb, mask.astype(bool), rng)
    # Shading of the cloth under the mark (relative brightness), so folds show through the print.
    lum = rgb.astype(np.float64).mean(axis=2)
    shade = np.clip(lum / max(1.0, float(lum[mask.astype(bool)].mean())), 0.6, 1.3)[..., None]
    out[area] = (1 - opacity) * out[area] + opacity * np.clip(colour * shade, 0, 255)[area]
    return np.clip(out, 0, 255).astype(np.uint8), kind

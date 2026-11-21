"""Run-time steadiness for the live picture (owner, 2026-11-18: "run-time anti-flicker", fxreal8 test).

Every live frame is drawn on its own, so parts of the picture that do not move can still shimmer from one frame
to the next. This keeps part of the previous OUTPUT where the CAMERA picture did not change, and none of it where
the person moves:

  still  = 1 - (blurred |camera_t - camera_t-1|) / motion, clipped to 0..1   (1 = nothing moved there)
  out_t' = out_t + blend * still * (out_t-1' - out_t)

blend 0 = off. The risk is smearing at the edge of a moving arm; the eval's movement score shows it.
Pure numpy (the live worker can use the same two functions), about 2 ms per 288x512 frame.
"""

from __future__ import annotations

import numpy as np


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    """Mean over a k x k window (k odd), edges repeated; integral image, so the cost does not grow with k."""
    if k <= 1:
        return x
    r = k // 2
    p = np.pad(x, ((r + 1, r), (r + 1, r)), mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k)).astype(np.float32)


def still_weight(prev_cam, cam, motion: float = 0.04, blur: int = 9) -> np.ndarray:
    """H x W in 0..1: 1 where the camera picture did not change between the two frames, 0 where it moved by
    `motion` (0..1 of full brightness) or more."""
    a = np.asarray(prev_cam, dtype=np.float32) / 255.0
    b = np.asarray(cam, dtype=np.float32) / 255.0
    if a.shape != b.shape:
        raise ValueError(f"camera frames differ in size: {a.shape} vs {b.shape}")
    d = np.abs(b - a)
    d = d.mean(axis=2) if d.ndim == 3 else d
    d = _box_blur(d, blur)
    return np.clip(1.0 - d / max(motion, 1e-6), 0.0, 1.0)


def steady_blend(prev_out, out, still: np.ndarray, blend: float) -> np.ndarray:
    """The new output with `blend * still` of the previous (already steadied) output kept, as uint8 RGB."""
    o = np.asarray(out, dtype=np.float32)
    if blend <= 0:
        return o.round().clip(0, 255).astype(np.uint8)
    p = np.asarray(prev_out, dtype=np.float32)
    if p.shape != o.shape:
        raise ValueError(f"outputs differ in size: {p.shape} vs {o.shape}")
    w = (blend * still)[..., None] if o.ndim == 3 else blend * still
    return (o + w * (p - o)).round().clip(0, 255).astype(np.uint8)


class SteadyFilter:
    """Frame by frame: feed (camera frame, model output) pairs; returns the steadied output. reset() on a new
    clip or a garment change (the live worker would call it when the garment changes)."""

    def __init__(self, blend: float, motion: float = 0.04, blur: int = 9):
        if not 0.0 <= blend < 1.0:
            raise ValueError(f"blend must be in [0, 1), got {blend}")
        self.blend, self.motion, self.blur = blend, motion, blur
        self.reset()

    def reset(self) -> None:
        self._cam = None
        self._out = None

    def __call__(self, cam, out) -> np.ndarray:
        cam = np.asarray(cam)
        o = np.asarray(out)
        if self.blend <= 0 or self._out is None or self._cam is None or self._cam.shape != cam.shape:
            res = o.astype(np.uint8)
        else:
            res = steady_blend(self._out, o, still_weight(self._cam, cam, self.motion, self.blur), self.blend)
        self._cam, self._out = cam, res
        return res

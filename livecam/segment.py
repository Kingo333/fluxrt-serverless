"""Person-part segmentation for the FluxRT live path, and the pure rules built on it.

Used by server.py behind env switches (all default off):
- FLUXRT_COMPOSITE: keep the camera frame's face, hair and background, take the
  clothing from FluxRT's output (`keep_mask` + `composite`).
- FLUXRT_REF_CLEAN: cut the garment out of an on-model product photo (`clean_reference`).
- FLUXRT_CACHE_MASK=seg: drive FluxRT's spatial cache from the segmentation
  (`CacheMaskBuilder`).

Segmenter: MediaPipe's multiclass selfie segmenter (Apache-2.0): classes
background, hair, body skin, face skin, clothes, others. 256x256 input, a few ms.

Policy (owner, 2026-11-06): try-ons with no face in view must work fully; when
unsure, let the model's output through rather than keep camera pixels.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("fluxrt-lb.segment")

# MediaPipe multiclass selfie segmenter classes.
BACKGROUND, HAIR, BODY_SKIN, FACE, CLOTHES, OTHERS = 0, 1, 2, 3, 4, 5

# The app's categories (buildTryOnPrompt.ts) and the dataset names used offline.
_GARMENT = {"tops", "bottoms", "dresses", "outerwear", "upper", "lower", "dress", "top", "bottom"}
_SHOES = {"shoes", "footwear", "shoe"}
_BAGS = {"bags", "bag"}
_ACCESSORIES = {"accessories", "accessory"}

# A frame counts as "no person" when almost everything is background.
NO_PERSON_BACKGROUND_FRAC = 0.97
# A face counts as present when it covers at least this fraction of the frame.
FACE_MIN_FRAC = 0.005
# Reference cleaning: a person is "on the photo" above this fraction; the
# garment must cover at least this much or cleaning is skipped (never blank it).
REF_PERSON_MIN_FRAC = 0.005
REF_CLOTHES_MIN_FRAC = 0.05


def normalize_category(category) -> Optional[str]:
    """App category -> garment | shoes | bags | accessories | None (unknown = pass-through)."""
    c = str(category or "").strip().lower()
    if c in _GARMENT:
        return "garment"
    if c in _SHOES:
        return "shoes"
    if c in _BAGS:
        return "bags"
    if c in _ACCESSORIES:
        return "accessories"
    return None


def fraction(cat: np.ndarray, classes) -> float:
    return float(np.isin(cat, classes).mean()) if cat.size else 0.0


def face_present(cat: np.ndarray) -> bool:
    return fraction(cat, (FACE,)) >= FACE_MIN_FRAC


def no_person(cat: np.ndarray) -> bool:
    return fraction(cat, (BACKGROUND,)) > NO_PERSON_BACKGROUND_FRAC


def keep_mask(cat_in: np.ndarray, cat_out: np.ndarray, category) -> Optional[np.ndarray]:
    """Pixels to take from the REAL camera frame; None means pass the model output through.

    cat_in: segmentation of the camera frame; cat_out: of the generated frame (same shape).
    - garments: face | hair | background from the camera, minus clothes | others in the
      generated frame (a longer dress or new sleeves win). Body skin is never kept. With
      no face in view: background only, same minus rule.
    - shoes: face | hair if a face is found, else pass-through. Feet, background and
      'others' are never protected.
    - bags: face only if found, else pass-through.
    - accessories, unknown, no person in frame: pass-through.
    """
    kind = normalize_category(category)
    if kind is None or kind == "accessories" or cat_in.shape != cat_out.shape or no_person(cat_in):
        return None
    has_face = face_present(cat_in)
    if kind == "garment":
        classes = (FACE, HAIR, BACKGROUND) if has_face else (BACKGROUND,)
        keep = np.isin(cat_in, classes)
        return keep & ~np.isin(cat_out, (CLOTHES, OTHERS))
    if not has_face:
        return None
    if kind == "shoes":
        return np.isin(cat_in, (FACE, HAIR))
    return cat_in == FACE  # bags


def _blur_separable(a: np.ndarray, radius: int) -> np.ndarray:
    """Box blur (radius cells) on a 2-D float array; cheap feathering without cv2/scipy."""
    if radius <= 0:
        return a
    k = 2 * radius + 1
    pad = np.pad(a, radius, mode="edge")
    c = np.cumsum(pad, axis=0)
    c = np.concatenate([np.zeros((1, c.shape[1]), c.dtype), c], axis=0)
    a = (c[k:, :] - c[:-k, :]) / k
    c = np.cumsum(a, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1), c.dtype), c], axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def feather_alpha(keep: np.ndarray, feather_px: int) -> np.ndarray:
    """keep (bool HxW) -> alpha (float32 HxW, 1 = real camera pixel)."""
    alpha = keep.astype(np.float32)
    if feather_px > 0:
        alpha = _blur_separable(alpha, int(feather_px)).astype(np.float32)
    return np.clip(alpha, 0.0, 1.0)


def composite(generated: np.ndarray, real: np.ndarray, keep: Optional[np.ndarray], feather_px: int = 3) -> np.ndarray:
    """Blend the real frame over the generated one where `keep` is True (same HxWx3 uint8 layout)."""
    if keep is None:
        return generated
    if real.shape != generated.shape or keep.shape != generated.shape[:2]:
        raise ValueError(f"composite shapes differ: generated {generated.shape}, real {real.shape}, keep {keep.shape}")
    if feather_px <= 0:
        return np.where(keep[..., None], real, generated).astype(np.uint8)
    alpha = feather_alpha(keep, feather_px)[..., None]
    out = generated.astype(np.float32) * (1.0 - alpha) + real.astype(np.float32) * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def resize_category(cat: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize of a category map (no interpolation between classes)."""
    if cat.shape == (height, width):
        return cat
    ys = (np.arange(height) * cat.shape[0] / height).astype(np.int64)
    xs = (np.arange(width) * cat.shape[1] / width).astype(np.int64)
    return cat[ys[:, None], xs[None, :]]


def clean_reference(rgb: np.ndarray, cat: np.ndarray, feather_px: int = 2) -> Optional[np.ndarray]:
    """On-model product photo -> the garment alone on white. None = leave the photo as it is
    (flat-lay / ghost photo, no person found, or the garment area is too small to be safe)."""
    if cat.shape != rgb.shape[:2]:
        return None
    if fraction(cat, (FACE, BODY_SKIN, HAIR)) < REF_PERSON_MIN_FRAC:
        return None
    garment = np.isin(cat, (CLOTHES, OTHERS))
    if garment.mean() < REF_CLOTHES_MIN_FRAC:
        return None
    white = np.full_like(rgb, 255)
    return composite(white, rgb, garment, feather_px)


def cells_any(mask: np.ndarray, cell: int = 16) -> np.ndarray:
    """Max-pool a bool HxW mask to (H//cell, W//cell) cells (True if any pixel is True)."""
    h, w = mask.shape[0] // cell, mask.shape[1] // cell
    m = mask[: h * cell, : w * cell].reshape(h, cell, w, cell)
    return m.any(axis=(1, 3))


def dilate_cells(cells: np.ndarray, n: int) -> np.ndarray:
    out = cells.copy()
    for _ in range(max(0, int(n))):
        p = np.pad(out, 1)
        out = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:] | out
    return out


def _blur3(a: np.ndarray) -> np.ndarray:
    """FluxRT's gaussian_blur(kernel 3, sigma 0.5) on an HxWxC float array, separable."""
    k = np.array([0.1065, 0.7870, 0.1065], dtype=np.float32)
    p = np.pad(a, ((1, 1), (0, 0), (0, 0)), mode="reflect")
    a = p[:-2] * k[0] + p[1:-1] * k[1] + p[2:] * k[2]
    p = np.pad(a, ((0, 0), (1, 1), (0, 0)), mode="reflect")
    return p[:, :-2] * k[0] + p[:, 1:-1] * k[1] + p[:, 2:] * k[2]


class CacheMaskBuilder:
    """Manual spatial-cache mask for FluxRT (FLUXRT_CACHE_MASK=seg).

    Cell values: 2 = recompute and update the cache, 0 = reuse the cache. The item region
    is never skipped: cells are 2 where FluxRT's own motion rule fires (squared diff of the
    3x3-blurred frame vs the cached frame, max-pooled to cells, > 0.1, dilated twice; the
    cached frame is updated where it fired, as FluxRT does) OR where the camera segmentation
    is not background (dilated). The first frame, every `refresh` frames, after force_full()
    (prompt/garment changes: manual mode ignores FluxRT's own reset), and whenever the
    category is shoes/bags/accessories/unknown or no person is found, every cell is 2.
    """

    THRESHOLD = 0.1  # on frames scaled to [-1, 1], as FluxRT's pipeline preprocesses them

    def __init__(self, height: int, width: int, cell: int = 16, dilate: int = 2, refresh: int = 48):
        self.h, self.w = height // cell, width // cell
        self.cell, self.dilate, self.refresh = cell, max(0, int(dilate)), max(1, int(refresh))
        self.cached: Optional[np.ndarray] = None
        self.count = 0
        self.force = True

    def force_full(self) -> None:
        self.force = True

    def full(self) -> np.ndarray:
        return np.full((self.h, self.w), 2, dtype=np.int32)

    def motion_cells(self, frame_rgb: np.ndarray) -> np.ndarray:
        f = _blur3(frame_rgb.astype(np.float32) / 127.5 - 1.0)
        if self.cached is None:
            self.cached = f
            return np.ones((self.h, self.w), dtype=bool)
        diff = ((self.cached - f) ** 2).mean(axis=2)
        hc, wc = self.h * self.cell, self.w * self.cell
        pooled = diff[:hc, :wc].reshape(self.h, self.cell, self.w, self.cell).max(axis=(1, 3))
        fired = dilate_cells(pooled > self.THRESHOLD, 2)
        up = np.repeat(np.repeat(fired, self.cell, axis=0), self.cell, axis=1)
        self.cached[:hc, :wc][up] = f[:hc, :wc][up]
        return fired

    def build(self, frame_rgb: np.ndarray, cat: Optional[np.ndarray], category) -> np.ndarray:
        self.count += 1
        motion = self.motion_cells(frame_rgb)
        kind = normalize_category(category)
        if self.force or self.count == 1 or self.count % self.refresh == 0 or kind != "garment" \
                or cat is None or cat.shape != frame_rgb.shape[:2] or no_person(cat):
            self.force = False
            return self.full()
        person = dilate_cells(cells_any(cat != BACKGROUND, self.cell), self.dilate)
        return np.where(motion | person, 2, 0).astype(np.int32)


class MediaPipeSegmenter:
    """Category map (H x W uint8) per RGB uint8 frame. Loads the tflite once."""

    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"segmentation model not found: {model_path}")
        self._mp = mp
        opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=str(path)),
                                            output_category_mask=True)
        self._seg = vision.ImageSegmenter.create_from_options(opts)
        # One segmenter is shared by the camera loop, the output poller's worker thread and
        # reference cleaning; MediaPipe task objects are not safe to call concurrently.
        self._lock = threading.Lock()

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(rgb, dtype=np.uint8)
        with self._lock:
            res = self._seg.segment(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr))
            return res.category_mask.numpy_view().copy()


def load_segmenter(model_path: str):
    """The segmenter, or None (logged) so every feature that needs it turns itself off."""
    try:
        return MediaPipeSegmenter(model_path)
    except Exception as e:  # noqa: BLE001 - the live app must keep working without it
        log.error("segmenter unavailable (%s: %s); composite / ref-clean / seg cache mask are OFF", type(e).__name__, e)
        return None

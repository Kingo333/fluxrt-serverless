"""Per-session live-quality logic for the FluxRT path, kept out of server.py so it can be
unit-tested with fakes (no GPU engine). server.py creates one LiveSession per WebSocket.

With every switch off, prepare_input() is exactly the old crop and finish_output()
returns the frame untouched.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from livecam.pacing import FrameRing, composite_source_ts
from livecam.segment import CacheMaskBuilder, composite, keep_mask, normalize_category, resize_category

log = logging.getLogger("fluxrt-lb.session")


class LiveSession:
    def __init__(self, proc: Any, crop_fn: Callable, segmenter: Optional[Callable], *,
                 width: int, height: int, out_width: int, out_height: int,
                 composite_on: bool = False, feather: int = 3, lag_override_s: Optional[float] = None,
                 out_seg_every: int = 1,
                 cache_mask: str = "auto", mask_dilate: int = 2, mask_refresh: int = 48,
                 anchor_on: bool = False, anchor_delay_s: float = 1.5, ring_size: int = 16):
        self.proc, self.crop_fn, self.segmenter = proc, crop_fn, segmenter
        self.w, self.h, self.out_w, self.out_h = width, height, out_width, out_height
        self.composite_on = bool(composite_on and segmenter is not None)
        self.feather, self.lag_override_s = feather, lag_override_s
        self.out_seg_every = max(1, int(out_seg_every))
        self.mask_builder: Optional[CacheMaskBuilder] = None
        if cache_mask == "seg" and segmenter is not None:
            self.mask_builder = CacheMaskBuilder(height, width, 16, mask_dilate, mask_refresh)
        self.anchor_on = bool(anchor_on and hasattr(proc, "set_anchor_image"))
        self.anchor_delay_s = anchor_delay_s
        self.anchor_due_at: Optional[float] = None
        self.anchor_set = False
        self.category: Optional[str] = None
        self.ring = FrameRing(ring_size)
        self._cat_out: Optional[np.ndarray] = None
        self._out_count = 0
        self._logged: set = set()
        self.stats = {"composited": 0, "passed_through": 0, "masks": 0, "anchors": 0}

    # ---- events from the app -------------------------------------------------
    def set_category(self, category) -> None:
        self.category = str(category).strip() if isinstance(category, str) and category.strip() else None

    def garment_changed(self, now: Optional[float] = None) -> None:
        """set_prompt / set_reference_image: new garment context. Re-anchor later, recompute everything."""
        now = time.time() if now is None else now
        if self.anchor_on:
            if self.anchor_set:
                self._call("set_anchor_image", None)
            self.anchor_set = False
            self.anchor_due_at = now + self.anchor_delay_s
        if self.mask_builder is not None:
            self.mask_builder.force_full()

    # ---- camera frame -> FluxRT input -------------------------------------------
    def prepare_input(self, frame_bgr: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        now = time.time() if now is None else now
        resized = self.crop_fn(frame_bgr, self.h, self.w)
        if self.segmenter is None or not (self.composite_on or self.mask_builder is not None):
            return resized
        cat = None
        try:
            cat = self.segmenter(np.ascontiguousarray(resized[:, :, ::-1]))
        except Exception as e:  # noqa: BLE001
            self._log_once("seg_fail", "segmentation failed (%s); composite / cache mask pass through", e)
        if self.mask_builder is not None:
            mask = self.mask_builder.build(np.ascontiguousarray(resized[:, :, ::-1]), cat, self.category)
            self._call("set_mask", mask)
            self.stats["masks"] += 1
        if self.composite_on:
            full = resized if (self.out_h, self.out_w) == (self.h, self.w) else self.crop_fn(frame_bgr, self.out_h, self.out_w)
            self.ring.push(now, (full, cat))
        return resized

    # ---- FluxRT output -> frame for the app --------------------------------------
    def finish_output(self, output_bgr: np.ndarray, now: Optional[float] = None,
                      last_processing_time: Optional[float] = None) -> np.ndarray:
        now = time.time() if now is None else now
        out = output_bgr
        if self.composite_on:
            out = self._composite(output_bgr, now, last_processing_time)
        if self.anchor_on and not self.anchor_set and self.anchor_due_at is not None and now >= self.anchor_due_at:
            self._take_anchor(out)
        return out

    def _composite(self, output_bgr: np.ndarray, now: float, last_processing_time: Optional[float]) -> np.ndarray:
        if last_processing_time is None:
            try:
                last_processing_time = float(self.proc.get_last_processing_time())
            except Exception:  # noqa: BLE001
                last_processing_time = 0.0
        src = self.ring.at(composite_source_ts(now, last_processing_time, self.lag_override_s))
        if src is None or src[1] is None or output_bgr.shape[:2] != (self.out_h, self.out_w):
            self.stats["passed_through"] += 1
            return output_bgr
        full, cat_in = src
        self._out_count += 1
        if self._cat_out is None or self._out_count % self.out_seg_every == 1 or self.out_seg_every == 1:
            try:
                small = Image.fromarray(np.ascontiguousarray(output_bgr[:, :, ::-1])).resize((self.w, self.h), Image.BILINEAR)
                self._cat_out = self.segmenter(np.asarray(small))
            except Exception as e:  # noqa: BLE001
                self._log_once("seg_out_fail", "output segmentation failed (%s); passing through", e)
                self.stats["passed_through"] += 1
                return output_bgr
        keep = keep_mask(resize_category(cat_in, self.out_h, self.out_w),
                         resize_category(self._cat_out, self.out_h, self.out_w), self.category)
        if keep is None:
            self.stats["passed_through"] += 1
            self._log_once("pass", "composite pass-through (category=%r: no person, unknown category or accessory)",
                           normalize_category(self.category) or self.category)
            return output_bgr
        self.stats["composited"] += 1
        return composite(output_bgr, full, keep, self.feather)

    def _take_anchor(self, out_bgr: np.ndarray) -> None:
        rgb = Image.fromarray(np.ascontiguousarray(out_bgr[:, :, ::-1])).resize((self.w, self.h), Image.BILINEAR)
        if self._call("set_anchor_image", np.asarray(rgb).copy()):
            self.anchor_set = True
            self.stats["anchors"] += 1
            if self.mask_builder is not None:
                self.mask_builder.force_full()
            log.info("anchor reference set (category=%r)", self.category)

    def _call(self, name: str, *args) -> bool:
        try:
            getattr(self.proc, name)(*args)
            return True
        except Exception as e:  # noqa: BLE001 - a failed optional call must not end the session
            self._log_once(name, "%s failed (%s); feature continues without it", name, e)
            return False

    def _log_once(self, key: str, msg: str, *args) -> None:
        if key not in self._logged:
            self._logged.add(key)
            log.warning(msg, *args)

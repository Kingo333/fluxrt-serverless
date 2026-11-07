"""Output pacing helpers for the FluxRT live path (FLUXRT_OUTPUT_PACING).

FluxRT's scheduler writes its 2**interpolation_exp frames into the output tensor spaced
over the last processing time. server.py used to read that tensor only when a camera frame
arrived, so most of those frames were never seen. With pacing on, a poller reads the tensor
on its own clock; `FrameChangeDetector` tells it cheaply whether the content changed, and
`FrameRing` holds recent camera frames so the composite can use the one that produced the
output (FluxRT runs behind the camera by about one processing time).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import numpy as np


class FrameChangeDetector:
    """Change detection on a strided sample of the frame (about 2 KB per 576x1024 frame)."""

    def __init__(self, stride: int = 32):
        self.stride = max(1, int(stride))
        self._last: Optional[bytes] = None

    def changed(self, frame: np.ndarray) -> bool:
        sample = np.ascontiguousarray(frame[:: self.stride, :: self.stride]).tobytes()
        if sample == self._last:
            return False
        self._last = sample
        return True


class FrameRing:
    """Recent camera frames with their arrival time; `at(ts)` picks the one closest to ts.

    Pushed from the event loop (camera frames) and read from the poller's worker thread, so
    every access takes a lock (a deque raises if it changes while being iterated)."""

    def __init__(self, size: int = 16):
        self._items: deque = deque(maxlen=max(2, int(size)))
        self._lock = threading.Lock()

    def push(self, ts: float, payload) -> None:
        with self._lock:
            self._items.append((ts, payload))

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def latest(self):
        with self._lock:
            return self._items[-1][1] if self._items else None

    def at(self, ts: float):
        """The payload whose arrival time is closest to ts (None when empty)."""
        with self._lock:
            if not self._items:
                return None
            return min(self._items, key=lambda it: abs(it[0] - ts))[1]


def composite_source_ts(now: float, last_processing_time: float, lag_override_s: Optional[float] = None) -> float:
    """When the frame now in FluxRT's output was captured: about one processing time ago
    (the model works on the newest frame it had when it started), unless an explicit lag is set."""
    lag = lag_override_s if lag_override_s is not None else max(0.0, float(last_processing_time or 0.0))
    return now - lag

"""Short runs of consecutive frames where the clothes move (fxreal6, owner 2026-11-15).

Live FluxRT draws about 6 pictures a second, so two pictures in a row are about 0.17 s apart. A
"window" is K frames that far apart (K=4: about half a second of a step, a turn or a swinging hem).
The trainer keeps a window in one batch with the same noise, and its sequence term lets the garment,
face and skin change between the frames only as much as the real frames do.

Pure functions here (the masks come from data_tools.vivid_motion's segmentation pass):

  stride_for(fps)                      frames to skip between two pictures of a window
  pick_windows(masks, k, stride, n)    the n windows with the most clothes movement, not overlapping
"""

from __future__ import annotations

from .vivid_motion import change

LIVE_GAP_S = 0.17  # 1 / 6 base frames per second


def stride_for(fps: float, gap_s: float = LIVE_GAP_S) -> int:
    """Source frames between two pictures of a window (at least 1)."""
    if not fps or fps <= 0:
        return 5  # 30 fps footage
    return max(1, round(fps * gap_s))


def window_frames(start: int, k: int, stride: int) -> list[int]:
    return [start + i * stride for i in range(k)]


def window_motion(masks: dict, frames: list[int]) -> float | None:
    """Clothes movement over a window: the sum of the change between neighbouring frames, or None when
    a frame of the window was not segmented (too small a person, or past the end of the clip)."""
    if any(t not in masks for t in frames):
        return None
    return sum(change(masks[a], masks[b]) for a, b in zip(frames, frames[1:]))


def pick_windows(masks: dict, k: int = 4, stride: int = 5, n: int = 2, min_motion: float = 0.0) -> list[dict]:
    """Up to n windows of k frames, `stride` apart, ranked by clothes movement; windows never share a
    frame span, so two windows of one clip show two different moments. Earliest first on a tie."""
    if k < 2 or stride < 1 or n < 1:
        return []
    scored = []
    for start in sorted(masks):
        fr = window_frames(start, k, stride)
        m = window_motion(masks, fr)
        if m is not None and m >= min_motion:
            scored.append((m, start, fr))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked: list[dict] = []
    for m, start, fr in scored:
        if all(fr[-1] < p["frames"][0] or fr[0] > p["frames"][-1] for p in picked):
            picked.append({"frames": fr, "motion": round(m, 4)})
            if len(picked) == n:
                break
    return sorted(picked, key=lambda p: p["frames"][0])

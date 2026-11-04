"""What the live JoyAI session runs, as plain numbers (no torch), so training can assert it
matches and tests can check it.

Live (deploy/xvideo/serving/joyomni_streaming.py at JoyAI d875895, our image's commit):
  scheduler FlowMatchDiscreteScheduler(num_train_timesteps=1000, shift=5.159, reverse=True)
  set_timesteps(JOYAI_STEPS=2): sigmas = shift(linspace(1, 0, 3)) = [1.0, 0.8376, 0.0] (0.83764)
  timesteps = sigmas[:-1] * 1000 = [1000.0, 837.64]
  chunk_size 1 latent frame, causal, local_window_size 3 with the first chunk as a sink,
  max_temporal_ids 8, garment reference basesize 768, no CFG, 480x840 portrait at 24 fps.
"""

from __future__ import annotations

import random

LIVE_SHIFT = 5.159
LIVE_STEPS = 2
LIVE_TRAIN_TIMESTEPS = 1000
LIVE_MAX_TEMPORAL_IDS = 8
LIVE_REF_BASESIZE = 768
LIVE_HEIGHT, LIVE_WIDTH = 840, 480
LIVE_FPS = 24
LIVE_CHUNK_SIZE = 1
LIVE_WINDOW = 3


def sd3_shift(sigma: float, shift: float = LIVE_SHIFT) -> float:
    """FlowMatchDiscreteScheduler.sd3_time_shift."""
    return shift * sigma / (1 + (shift - 1) * sigma)


def live_sigmas(steps: int = LIVE_STEPS, shift: float = LIVE_SHIFT) -> list[float]:
    """set_timesteps(steps) with reverse=True: shift(linspace(1, 0, steps + 1))."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    return [sd3_shift(1 - i / steps, shift) for i in range(steps + 1)]


def live_timesteps(steps: int = LIVE_STEPS, shift: float = LIVE_SHIFT,
                   num_train_timesteps: int = LIVE_TRAIN_TIMESTEPS) -> list[float]:
    return [s * num_train_timesteps for s in live_sigmas(steps, shift)[:-1]]


def matches_live(timesteps, steps: int = LIVE_STEPS, tol: float = 1e-2) -> bool:
    want = live_timesteps(steps)
    got = [float(t) for t in timesteps]
    return len(got) == len(want) and all(abs(a - b) <= tol for a, b in zip(got, want))


EXIT_MODES = ("random", "last")


def choose_exit(mode: str, num_steps: int, rng: random.Random | None = None) -> int:
    """Which of the 2 live steps gets the gradient for a chunk.

    random (upstream Resilient-DMD): step 0 (x0 from pure noise at t=1000) or step 1 (x0 at
    t=837.64 after one live Euler step), uniformly; trains both steps the live session runs.
    last: always step 1, i.e. exactly the frame the live session outputs."""
    if mode not in EXIT_MODES:
        raise ValueError(f"exit mode must be one of {EXIT_MODES}, got {mode!r}")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if mode == "last" or num_steps == 1:
        return num_steps - 1
    return (rng or random).randrange(num_steps)


def latent_frames(pixel_frames: int, temporal_factor: int = 8) -> int:
    """Causal VAE: 1 + (T - 1) / factor latent frames; T must be factor*n + 1."""
    if pixel_frames < 1 or (pixel_frames - 1) % temporal_factor:
        raise ValueError(f"clips need {temporal_factor}n+1 frames, got {pixel_frames}")
    return 1 + (pixel_frames - 1) // temporal_factor


def chunk_windows(num_chunks: int, window: int = LIVE_WINDOW, sink: bool = True) -> list[list[int]]:
    """Pure copy of the streaming pipeline's _get_chunk_windows (selected chunk ids per chunk),
    used to state in the logs which history each trained chunk sees."""
    out = []
    for c in range(num_chunks):
        if sink and c > 0:
            tail = max(window - 1, 1)
            out.append([0] + list(range(max(1, c - tail + 1), c + 1)))
        else:
            out.append(list(range(max(0, c - window + 1), c + 1)))
    return out

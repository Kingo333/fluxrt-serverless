# JoyAI cold start: what it costs and what we do about it

Prices are for an RTX 5090 worker on the endpoint: **$0.0005/s ≈ $1.80/h** (extension reading, 2026-11-01).

## Done in code (free)

| Item | Status |
|---|---|
| Inductor, Triton and CUDA kernel caches on the network volume (`/runpod-volume/joyai/cache_rtx5090`, `TORCHINDUCTOR_FX_GRAPH_CACHE=1`) | **Already on** (`start_upstream.sh`). Compiled kernels are reused across cold starts. CUDA graphs are captured again on every start; they cannot be cached |
| **Warm up portrait only** (`JOYOMNI_WARMUP_ORIENTATION=portrait`, patch `joyai-warmup-orientation.patch`) | **New.** Upstream compiled and warmed both 480×840 and 840×480, including a 4-chunk full-pipeline warm-up for each. The app only sends portrait. This removes about half of the warm-up work (the exact saving needs the next cold-start log) |
| FlashBoot | Already on (endpoint setting) |

## Options that cost money (owner's choice)

| Option | What it buys | Cost |
|---|---|---|
| **Pre-warm when Live Cam opens** (before Start) | Hides the cold start when the user takes longer to tap Start than the warm-up takes. Usually it only helps partly: people tap within seconds | Each open without a session costs one cold start plus the idle timeout. That is about 3–5 min + 5 min ≈ **$0.25–0.30 per open**. It also needs a Worker/edge-function change (deploy approval) and a per-user rate limit, so the page can't be used to keep GPUs busy |
| **Longer idle timeout** (300 s → 900 s) | A second session within 15 min starts warm | Up to **+$0.30 per session** (10 more idle minutes), paid even when no one comes back |
| **Active workers = 1 during business hours** | No cold start at all in those hours | **$1.80/h**: 10 h/day ≈ $18/day ≈ **$540/month** |

## Recommendation

1. Ship the free changes and read the next cold-start log for the step times: container start → weights loaded → warm-up done → ready.
2. If the weights load from the network volume dominates (32.5 GB DiT + 16 GB text encoder), no compile setting fixes it.
   Only a warm worker (idle timeout or active worker) avoids it.
3. Then, if needed: a longer idle timeout only at peak times, before pre-warm or an always-on worker.

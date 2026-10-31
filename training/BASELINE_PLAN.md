# Baseline benchmark plan (Phase A2/A3)

The goal is to measure today's JoyAI before any training, so every later change can be compared against it.

## What we measure

| Metric | How |
|---|---|
| Camera fps (sent) | App counter: frames sent per second |
| Output fps (shown) | App counter: frames drawn per second |
| Round-trip latency per frame | App sends `seq` + `t_client_ms`, the adapter echoes them with the matching output frame, and the app subtracts on **its own clock**. No cross-device clock math |
| Server time per frame | `server_ms` from the adapter (adapter receive → output, one clock) |
| Network + app time | round trip − server time |
| JoyAI stage times | `profile` per chunk, when `JOYAI_PROFILE_TIMINGS=true` (test window only) |
| Time to first transformed frame | App: session start → first frame |
| Garment swap gap | App: swap sent → first frame of the new garment |
| Cold and warm start | RunPod logs: container start → `JoyAI ready after …s` |
| GPU memory and utilisation | RunPod worker metrics or the logs (`device memory used`) |
| Frames dropped | sent minus received, plus the 2 MB buffer guard |
| Cost per active minute | RunPod $/hr ÷ 60 |

The app keeps these stats in `window.__azyahLiveCam.stats`, so the extension can read them in desktop Chrome with one line of JavaScript.

**Test footage:** the owner, or a consenting tester, in front of a laptop or phone camera. No real shoppers and no recordings.

## Script for the test window (about 15 minutes once warm)

1. Warm start: open Live Cam, pick garment A, and wait until it's running.
2. Stand still for 60 s, then read the stats.
3. Move normally for 60 s (turns, arm raises), then read the stats.
4. Swap to garment B and note the swap gap. Repeat 3 times.
5. End the session. Start again within 5 minutes for a warm-start time.
6. Copy the RunPod logs, the stats and the screenshots.

## Cost estimate for the test (A3), needs the owner's approval

| Item | Value |
|---|---|
| Endpoint | `azyah-runpod-fluxrt-bootstrap` (production, used only for this window) |
| GPU | 32 GB PRO (RTX 5090) **about $1.58/hr**. If the endpoint is set to 96 GB PRO (RTX PRO 6000), **about $3.49/hr**. The extension must read the setting first |
| Time | about 10 min cold start (new image) + about 20 min testing ≈ **30–40 min** |
| Expected cost | about **$1.05** (5090) or about **$2.35** (PRO 6000) |
| Maximum budget | **$4.00**. Stop at 60 minutes whatever happens |
| Stop conditions | 60 min reached, $4 spent, or the worker restarts or errors twice |
| After | Set Active workers back to 0. Report the actual cost and balance |
| Needs first | The worker image with the timing passthrough (this PR, built after merge) and the app timing change (deployed by Vercel after merge). Both need the owner's merge approval |

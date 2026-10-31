# Garment switching: what happens today, and how we'll measure it (roadmap Part 11)

## Today (from the code)
1. The app sends `set_reference_image` and then `set_prompt` on the open socket (`useLiveCamSession.ts` swap path).
2. The adapter acknowledges both and **coalesces them into one JoyAI `start`** after 0.25 s (`bridge.py _schedule_restart`). It's the same socket, with no reconnect.
3. JoyAI begins a **new edit session**: the KV cache and first-chunk anchor reset, and the new garment photo is encoded.
4. **Frames sent before JoyAI answers `started` are dropped** (`_forward_frame`). So the screen keeps the last old-garment frame until the first new-garment chunk arrives.

## Expected effects to measure
- **Gap:** time from swap to the first new frame. The app records it as `lastSwapGapMs` in `window.__azyahLiveCam.stats()`.
- **Black or frozen frames:** count of identical output frames after the swap. The app redraws the last frame, so a gap shows as a freeze, not black.
- **Previous-garment contamination:** compare the first 2 s after the swap with the new garment photo using `eval.metrics.garment_colour`. Low scores in the first chunks that then rise mean leftover state.
- **Identity or background change after the swap:** `eval.metrics.preservation` before and after.

## Plan
- **Measurement (paid):** a 3-swap run inside the P3 baseline window, already in the A3 script. No extra GPU time.
- **Possible fixes (free code, later, after measurement):**
  - keep sending frames during the restart instead of dropping them
  - prefetch-encode the next garment photo
  - test whether JoyAI can swap the reference without a full reset; that needs an upstream code read, since it isn't exposed today

  Not changed blindly: some reset is needed so the old garment doesn't persist.

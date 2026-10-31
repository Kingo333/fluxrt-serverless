# Transport: WebRTC research note (roadmap Part 13), DEFERRED

**Status:** deferred until Stage 1 or Stage 2 shows quality gains. It's a separate engineering project from training.

**Today:** camera frames go as JPEG, base64 inside JSON, over a WebSocket to RunPod, and come back the same way to a canvas.
- base64 adds about 33%
- there's no frame pacing
- head-of-line blocking on bad networks
- this is likely the main cause of FluxRT's choppiness

**Target:** camera WebRTC track → media server → GPU worker → WebRTC track → `<video>` (what Decart Lucy uses).

**Constraints and unknowns:**
- **UDP on RunPod Pods:** the roadmap says RunPod Pods don't support UDP. That's **unverified here**, because `docs.runpod.io` is blocked in this sandbox. If true, the media server (LiveKit Cloud, or LiveKit self-hosted on a UDP-capable VM) runs outside RunPod, and the GPU worker connects to it as a participant over a supported path (TCP/TURN or a server-to-server link).
- **A cheaper first step:** JoyAI's server already supports an **H.264 downlink** (`output_codec`; `_H264Stream` in `serve_joyomni_streaming.py`). Switching only the return path from JPEG to H.264 over the existing WebSocket cuts bandwidth for the same quality. That's an adapter and app change, testable in the P3 window.
- **Measure first:** end-to-end latency (per-frame round trip is now measured), dropped frames and mobile battery, against the WebSocket baseline. WebRTC goes behind a feature flag, with the WebSocket path kept as fallback.

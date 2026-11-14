# Live Cam engines: switch FluxRT ⇄ JoyAI on one RunPod endpoint

One image, `ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1-<sha>`, holds both engines.
The `LIVECAM_ENGINE` environment variable picks which one a worker runs.

**Two images (2026-11-14).**
- **Dual image** (FluxRT + JoyAI): `Dockerfile.livecam`, tags `livecam-dual-v1-<sha>`, built by `build-livecam-dual`. This is still how JoyAI is built.
- **FluxRT-only image**: `Dockerfile.livecam-fluxrt`, tags `livecam-fluxrt-v1-<sha>`, built by `build-livecam-fluxrt`.
  - It has the same base, system packages, FluxRT steps and app code, without JoyAI's venv, SageAttention, CUTLASS/joyomni_ops or adapter.
  - It is about 10.8 GB instead of 15.3 GB compressed, and the build takes about a third of the time.
  - FluxRT runs the same in both: JoyAI is never loaded when `LIVECAM_ENGINE=fluxrt`.
  - `tests/test_dockerfiles_fluxrt_only.py` keeps the shared steps word for word identical.
  - On the FluxRT-only image, `LIVECAM_ENGINE=joyai` stops at start-up with "use the livecam-dual image".

**Thin builds.** When only FluxRT app files changed (`server.py`, `bootstrap.py`, `session_token.py`,
`livecam/`), run the `build-livecam-dual` workflow on the branch with `base_commit` = the commit of an existing
image (e.g. the live one). The new `livecam-dual-v1-<sha>` is that image plus the app files
(`Dockerfile.livecam-patch`): no apt, PPA or pip step, every package unchanged, a few minutes instead of ~25.
The workflow refuses if anything else that goes into the image changed; main always gets a full build.

| | FluxRT (Standard, default) | JoyAI-Video-Edit (Premium HD) |
|---|---|---|
| `LIVECAM_ENGINE` | `fluxrt` (or unset) | `joyai` |
| GPU (endpoint "GPU configuration") | **32 GB PRO** or **96 GB PRO** (also older 48 GB cards) | **32 GB PRO** (RTX 5090) only |
| Network volume | none needed | **azyah-joyai** (weights at `/runpod-volume/joyai`) |
| Resolution | input `FLUXRT_WIDTH/HEIGHT` (288×512 today), output 2× (576×1024) with the Flow Upscaler | 480×840 portrait, 24 fps |
| Code | `server.py` (unchanged features: playback scheduler, latest-frame-wins) | `joyai/` adapter + unmodified JoyAI on 127.0.0.1:8080 |

## FluxRT engine version (pinned)

FluxRT is pinned to `32206e8` (2026-06-13, "flow upscaler and taef2") in both Dockerfiles.
Change `FLUXRT_REF` on purpose, never back to `main`.

Two upgrades are **on by default**. Their weights download at warm-up, next to FLUX.2-klein-4B, at pinned revisions:

| Extension | Weights | Size | Licence | What it does |
|---|---|---|---|---|
| TAEF2 (`ENABLE_TINY_VAE`) | `madebyollin/taef2` @ `bd244eb` → `taef2/` | 11 MB | MIT | faster tiny decoder |
| Flow Upscaler (`ENABLE_FLOW_UPSCALER`) | `TensorForger/FlowUpscaler` @ `ae33429` → `FlowUpscaler/` | 237 MB | Unlicense | 2× latent upscale of every frame (59M params, about 8 ms) |

FluxRT itself is Unlicense (public domain).

Other FluxRT settings in this image:
- **Garment reference: 384×512** (`FLUXRT_REF_WIDTH/HEIGHT`, multiples of 16). FluxRT's default is 256×256; before this change we used the camera size.
  The worker now pads the garment photo with white to 3:4 instead of letting FluxRT stretch it.
  It also passes the photo as RGB; before, it passed BGR, so red and blue were swapped in the garment.
- **Frames:** input stays at the model size, output is 2× that, JPEG quality `FLUXRT_OUTPUT_JPEG_QUALITY` (82).
  The `ready` message now tells the app `engine: "fluxrt"`, the input and output size, `input_fps` 10 and `input_jpeg_quality` 0.8
  (`FLUXRT_CLIENT_HINTS=false` turns this off). The upscaler magnifies JPEG blocks, so the app sends better frames.
  FluxRT renders about 5–12 frames a second on a 5090 (more are dropped).
- **PyTorch:** the dual image now installs torch 2.9.1 cu128 for FluxRT. The old cu124 build cannot run on the Blackwell cards (5090, PRO 6000).

**Rollback.** Two options:
- Fast, no rebuild: set `ENABLE_FLOW_UPSCALER=false` and `ENABLE_TINY_VAE=false` on the endpoint.
  Add `FLUXRT_REF_WIDTH=288` and `FLUXRT_REF_HEIGHT=512` for the old reference size.
- Full: go back to the image the endpoint runs today (tag ending `62a8da4f`).
  Note it in the extension's log before P2. That image installs FluxRT `main` as it was on its build date, with the cu124 torch build.

Both engines speak the same WebSocket messages and use the same `SESSION_SIGNING_SECRET`,
so the app and the Cloudflare Worker don't change when you flip.

## FluxRT live-quality switches (2026-11-06)

All OFF by default: with none of them set the worker behaves exactly as before. Each is an
endpoint environment variable; turning one off is editing the endpoint and letting the
workers restart (minutes, no code change). `/ping` shows what they resolved to under
`live_switches`, and warm-up logs the same line.

| Variable | Default | On | What it does | Off / rollback |
|---|---|---|---|---|
| `FLUXRT_OUTPUT_PACING` | `false` | `1` | Reads FluxRT's output on its own clock (`FLUXRT_OUTPUT_POLL_HZ`, default 30) so its x4 interpolated frames reach the app, instead of only when a camera frame arrives. Adds `playback_buffer`/`playback_fps` to the `ready` hints so the app enables its jitter buffer. Both sender modes consume the frames unchanged. | unset |
| `FLUXRT_COMPOSITE` | `false` | `1` | Keeps the camera frame's face, hair and background and takes only the clothing from FluxRT (MediaPipe multiclass selfie segmenter, baked into the image). Rules by the item's category from the app: garments keep face/hair/background minus the new clothing (no face in view: background only); shoes keep face/hair only; bags face only; accessories, unknown category, no person or a failed segmentation = pass-through. `FLUXRT_COMPOSITE_FEATHER` (3 px edge), `FLUXRT_COMPOSITE_LAG_MS` (unset = one processing time), `FLUXRT_COMPOSITE_OUT_SEG_EVERY` (1). | unset |
| `FLUXRT_REF_CLEAN` | `false` | `1` | Cuts the garment out of an on-model product photo (keeps it on white) before FluxRT sees it. Flat-lay photos, photos without a person and garments under 5% of the photo are left untouched. | unset |
| `FLUXRT_REF_CLEAN_DRESS` | same as `FLUXRT_REF_CLEAN` | `1` turns it on by itself (only dress photos are cleaned; every other photo is used as sent); `0` turns it off | Dresses (the app's `dresses`: abayas, kaftans, gowns) are cleaned harder: clothes only and nothing above the chin, so a hijab, scarf or necklace on the product photo is not drawn on the shopper. Only the garment photo is changed, never the camera frames, so the shopper's own hijab is not touched. The app sends the photo before `set_prompt` names the category, so the photo is re-cleaned when a dress's `set_prompt` arrives; a photo sent again for the same item keeps the current category. A dress is the app's `garment_type` `dresses` when the app sends it (an abaya filed as "modestwear" included), else a stored category of dress/dresses or with a dress word (abaya, kaftan, gown, jalabiya, jumpsuit). | `0` |
| `FLUXRT_CACHE_MASK` | `auto` | `seg` | Drives FluxRT's spatial cache from the segmentation: the person/garment and anything that moved are recomputed every frame, static background cells reuse the cache; everything is recomputed on the first frame, every `FLUXRT_CACHE_MASK_REFRESH` (48) frames, after a prompt/garment change, and always for shoes/bags/accessories/unknown categories (identical to today for them). `FLUXRT_CACHE_MASK_DILATE` (2 cells). | `auto` |
| `FLUXRT_ANCHOR` | `false` | `1` | `FLUXRT_ANCHOR_DELAY_S` (1.5) after a garment is set, keeps one generated frame and passes it as a third reference every frame, so the garment rendering stays consistent (needs the FluxRT patch in this image; about 20-30% more per frame). Cleared on every garment swap. | unset |
| `FLUXRT_ANCHOR_REFRESH_S` | `0` | e.g. `4` | With `FLUXRT_ANCHOR=1`: replaces the anchor with the latest generated frame every N seconds (minimum 2), so the outfit stays steady but follows new poses. Never refreshes on a frame with nobody in it (looks again a second later). Each refresh recomputes one whole frame. `0` = one anchor per garment, as before. | unset |
| `FLUXRT_FACE_PASTE` | `false` | `1` | Pastes only the camera's face onto FluxRT's output, so the shopper stays recognisable: face skin with its holes filled (eyes, glasses, open mouth) plus facial hair in the lower part of the face; never background, scalp hair, caps, scarves or hijabs. Only where both the camera frame and the generated frame see that face, and only when their outlines overlap at least `FLUXRT_FACE_PASTE_MIN_OVERLAP` (0.6); each face separately (two people); fades in over 6 outputs and out over 3. Soft edge `FLUXRT_FACE_PASTE_FEATHER` (8 px) ending `FLUXRT_FACE_PASTE_SHRINK` (4 px) inside the outline. Garments, shoes and bags only; accessories, unknown categories, no face in view or far-away faces = FluxRT's frame untouched. Real face colours are never changed. Off when `FLUXRT_COMPOSITE=1` (it keeps the face already). About 8 ms per output frame plus one segmentation of each camera and output frame (CPU). `livecam pacing` logs `face_pasted` and `face_gain`. | unset |
| `FLUXRT_MU` | unset | e.g. `1.9` | Fixed sampler shift instead of diffusers' resolution-based value (lower = sharper detail, higher = softer and steadier). | unset |
| `FLUXRT_INPUT_SCALE` | `1` | `2` | Asks the app for camera frames at model size x 2 (576x1024 for 288x512); FluxRT still runs at model size, the composite keeps the sharp real pixels. About 2x the upload. | `1` |

Owner's FluxRT test after Release #40 (7 Nov 2026): `FLUXRT_OUTPUT_PACING=1`, `FLUXRT_REF_CLEAN=1`,
`FLUXRT_ANCHOR=1`, `FLUXRT_ANCHOR_DELAY_S=3`, `FLUXRT_ANCHOR_REFRESH_S=4`, `FLUXRT_FACE_PASTE=1`;
`FLUXRT_COMPOSITE`, `FLUXRT_CACHE_MASK` and `FLUXRT_INPUT_SCALE` unset (the full composite pasted
background over the generated cap and body when frames did not line up).

Earlier recommended combination for the owner's app test (live endpoint values in brackets):
`FLUXRT_OUTPUT_PACING=1` with `ENABLE_PLAYBACK_SCHEDULER=true` (already true), `PLAYBACK_TARGET_FPS=24`
(was 12) and `REPEAT_LAST_FRAME_WHEN_IDLE=true` (already true): the poller refreshes the latest
output up to 30 times a second and the scheduler sends it at 24 fps. If the scheduler's target
is above the real output rate, it repeats the last frame, as today. Then `FLUXRT_COMPOSITE=1`,
`FLUXRT_REF_CLEAN=1`, `FLUXRT_INPUT_SCALE=2`; second round `FLUXRT_CACHE_MASK=seg`,
`FLUXRT_ANCHOR=1`. Quality test matrix (one at a time): `FLUXRT_WIDTH=384 FLUXRT_HEIGHT=672`
(both multiples of 16; the ready hints follow), `ENABLE_FLOW_UPSCALER=false`, `FLUXRT_MU=1.9` vs `2.5`.

If the segmenter fails to load, `FLUXRT_COMPOSITE`, `FLUXRT_REF_CLEAN`, `FLUXRT_FACE_PASTE` and `FLUXRT_CACHE_MASK=seg`
turn themselves off (logged, visible in `/ping`); the session runs as today.

## WebRTC media path (2026-11-12, off by default)

With `LIVECAM_WEBRTC=1` the app can carry camera frames and generated frames on a WebRTC connection
instead of JPEGs on the WebSocket. The WebSocket stays open for control and for signalling: the app
sends `webrtc_offer` (SDP plus its iceServers), the worker answers `webrtc_answer`, and `webrtc_state`
reports connected / failed / closed. FluxRT is unchanged:
- camera frames enter the same latest-frame-wins slot;
- outputs leave from the same finished-output step (composite, face paste, sharpen), so the picture is the same before compression.

While WebRTC is connected the worker sends no JPEG frames. If it drops, or the app sends `webrtc_close`, the JPEG path takes over again.

- **No inbound ports needed.** RunPod serverless forwards only HTTP/WebSocket, so the worker reaches the app through a TURN relay.
  - The provider is not wired in. The app forwards the iceServers it got from our backend: a self-hosted coturn with time-limited REST credentials, or Cloudflare Realtime TURN.
  - aiortc uses one TURN URL. The worker first sends a STUN check over UDP to that relay. If it answers, it uses TURN over UDP; otherwise TURN over TLS on 443. The log line `livecam webrtc: outbound UDP to ...` says which.
- **Codecs.** H.264 is preferred both ways, so phones and the kiosk use hardware video; VP8 is accepted. Encode and decode run on the CPU (aiortc + PyAV), never on the GPU.
- **Packages.** aiortc 1.16.0 (pulls PyAV), listed in `livecam/requirements-webrtc.txt`.
  - The thin build installs it, and since 2026-11-14 the full build (`Dockerfile.livecam`) does too.
  - Without the packages the worker logs why and stays on JPEG.

| Variable | Default | What it does |
|---|---|---|
| `LIVECAM_WEBRTC` | `false` | `1` lets the app offer WebRTC (`ready` then has `webrtc: true`). Unset = exactly today. |
| `LIVECAM_WEBRTC_MAX_KBPS` | `3000` | Ceiling of the generated-video bitrate. The output is 576x1024, so it needs about 2.5–4 Mbit/s to match today's JPEG detail. The app's bandwidth estimate moves the rate below this. |
| `LIVECAM_WEBRTC_START_KBPS` | `2000` | Starting bitrate of the generated video. |
| `LIVECAM_WEBRTC_STUN_HOST` | `stun.cloudflare.com` | Only used for the UDP check when the app's iceServers name no STUN or UDP TURN host. |

`/ping` shows `webrtc` and `webrtc_max_kbps` under `live_switches`. Every `LIVECAM_LOG_EVERY` frames,
`livecam webrtc:` logs the state, the TURN transport, frames in, the input size and frames out.

## FP8 transformer and compile (2026-11-14, off by default)

Today's model is INT8 weight-only (optimum-quanto: 8-bit weights, 16-bit maths). It saves memory, not time.
`FLUXRT_PRECISION=fp8` builds the same favourite #2 with FP8 weights and FP8 maths.
- **Same favourite.** It starts from the full-precision FLUX.2-klein-4B transformer with the LoRA fused at `FLUXRT_LORA_SCALE`, exactly as the INT8 build does.
- **What changes.** The Linear layers inside the transformer blocks switch to FP8: torchao, dynamic per-tensor scales, FP8 tensor cores (RTX 4090, RTX 5090, RTX PRO 6000).
- **What stays the same.**
  - The embedders, modulation and final layers stay 16-bit.
  - The text encoder stays INT8.
  - The VAE, frame interpolation and every live switch are unchanged.
- **Safe start.** If FP8 cannot run on that GPU (no FP8 tensor cores, a kernel error in the start-up test multiply), FluxRT logs `[azyah] FP8 transformer unavailable (...); falling back to INT8` and builds today's INT8 model.
- **Where torchao lives.** It sits in its own folder, `/opt/azyah-torchao`, outside the normal package path.
  - PEFT 0.21, which loads the LoRA, refuses to run once it can see a torchao older than 0.16.
  - 0.14.1 is the torchao release made for the image's torch 2.9.
  - So only the FP8 step adds that folder, and only after the LoRA is fused. The INT8 path never sees torchao.

`FLUXRT_COMPILE=1` turns on FluxRT's own `torch.compile` of the transformer and the frame interpolation.
- FluxRT also wraps the VAE, but it calls the VAE's encode/decode directly, which that wrapper doesn't cover.
- The first start of a new worker takes minutes longer.
- The compiler is slow the first time it meets each new number of changed tokens. So before the worker says ready, it runs **compile practice**:
  - a few practice frames with different amounts of change (60%, 15%, 0%, 85%, ...), with a stand-in garment set as in a session;
  - shoppers never see those pauses;
  - afterwards the garment slot and the mask are reset.
- `FLUXRT_COMPILE_CACHE_DIR` (e.g. `/runpod-volume/fluxrt_compile_cache`) keeps the compiled kernels for the next worker. The cache is only used when that disk has at least 2 GB free.
  - Note that a network volume reports the whole shared filesystem, not its own quota.

**Fewer CPU-GPU waits (`FLUXRT_FAST_MASK`, on by default).** FluxRT's token-skipping code checks the mask in every layer.
- Each check makes the CPU wait for the GPU: about 440 waits per generated frame.
- Its leftover timing helper adds 12 more full waits per frame, although its printout is switched off.
- The mask is the same for every layer within one model call, so `fluxrt/patches/fluxrt-speed.patch` works the changed tokens out once per call (3 waits) and drops the helper's waits.
- The output is identical: tested on FluxRT's own transformer with mixed masks.
- `FLUXRT_FAST_MASK=0` is FluxRT's original code path.

**Sizes are rounded, never fatal.**
- `FLUXRT_WIDTH`, `FLUXRT_HEIGHT`, `FLUXRT_REF_WIDTH` and `FLUXRT_REF_HEIGHT` are rounded to the nearest multiple of 16 and kept within 128..1024.
- A value that isn't a number uses the default.
- Each adjustment is logged at warm-up (`warmup: setting ...`) and listed in `setting_notes`.

All settable in the RunPod env, no rebuild:

| Variable | Default | Range | What it does |
|---|---|---|---|
| `FLUXRT_PRECISION` | `int8` | `int8`, `fp8` | `fp8`: FP8 transformer blocks. Needs `ENABLE_INT8` (default on) and the LoRA fused (`FLUXRT_LORA_FUSE`, default on); otherwise the worker logs why and stays INT8. |
| `FLUXRT_FAST_MASK` | on | `0`/`1` | Changed tokens worked out once per model call (fewer waits, same output). `0` = original. |
| `FLUXRT_COMPILE` | off | `0`/`1` | FluxRT's `compile_models`. |
| `FLUXRT_COMPILE_CACHE_DIR` | unset | a folder | Keeps the compile cache across workers. Used only with 2 GB free. |
| `FLUXRT_COMPILE_WARMUP` | on | `0`/`1` | Compile practice frames before ready (only with compile). |
| `FLUXRT_COMPILE_WARMUP_PASSES` | `4` | `1`..`12` | How many practice frames. More = longer first start, fewer surprises. |
| `TORCHDYNAMO_SUPPRESS_ERRORS` | on (with compile) | `0`/`1` | A part that fails to compile runs the normal way and is logged. `0` = stop on compile errors (debugging only). |
| `FLUXRT_COMPILE_RECOMPILE_LIMIT` | `64` | `8`..`512` | Variants the compiler may build per piece of code before running that piece uncompiled (PyTorch's own default is 8). |
| `FLUXRT_WIDTH` × `FLUXRT_HEIGHT` | `576` × `320` | multiples of 16, 128..1024 | Model size. The test endpoint uses 288×512; 304×544 is +12% pixels (about +6% detail per side, about 5% slower). |
| `FLUXRT_SHARPEN` | `0` (off) | `0`..`1` | Unsharp mask on each output frame; above 0.5 edges halo. |

The `warmup: ready; live switches {...}` line shows every value in use:
- `precision`, `precision_active`, `fast_mask`
- `compile`, `compile_cache`, `compile_warmup_passes`, `compile_suppress_errors`, `compile_recompile_limit`
- `size`, `output_size`, `ref_size`, `sharpen`, `steps`, `setting_notes`

The compile practice logs one line per pass: `warmup: compile practice 2/4 (15% changing): 3 frames in N s, slowest M s`.

**Where the models are.** FLUX.2-klein-4B (full precision, with `transformer/`) and FLUX.2-klein-4B-int8 are downloaded from Hugging Face (`HF_TOKEN`) at start-up.
- They go to the worker's own disk: `/workspace/FluxRT/FLUX.2-klein-4B` and `/workspace/FluxRT/FLUX.2-klein-4B-int8`.
- They are not on the network volume. Only the LoRA (`/runpod-volume/fluxrt_lora/...`) and JoyAI's weights are.
- FP8 needs no new download: the fused-LoRA INT8 build already loads that full-precision transformer.

**What the logs show.**
- At warm-up:
  - `warmup: model full precision: ...` and `warmup: model int8: ...` give each folder and its size.
  - `warmup: worker disk ...` and `warmup: network volume ...` give the free space.
  - `warmup: LoRA ...`.
  - `warmup: transformer precision asked: ...`.
  - `warmup: transformer built in fp8 (N layers)` (or `int8`, or `int8 (FP8 unavailable: ...)`).
- Every 5 s of a session, the `livecam pacing:` line ends in `model=...`, so the precision stays visible after the warm-up lines scroll away.
- `/ping` `live_switches` has `precision`, `precision_active`, `compile` and `compile_cache`.

**Package versions.** FluxRT's own `requirements.txt` is unpinned, so the full build installs from `fluxrt/constraints.txt`.
- That file holds the exact versions in the image the test endpoints run today: diffusers 0.42.1, transformers 5.22.2, peft 0.21.2, optimum-quanto 0.2.7, accelerate 1.15.0 and the rest.
- So a rebuild changes only what the commit changes.

## Switch to JoyAI

On the endpoint → **Edit endpoint**:
1. Container image: `ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1-<sha>` (once; it also runs FluxRT).
2. GPU configuration: untick **48 GB**, tick **32 GB PRO**.
3. Network volume: attach **azyah-joyai** (the endpoint then runs only in that volume's datacenter).
4. Environment variables: set `LIVECAM_ENGINE` = `joyai` and `PING_READY_WHILE_LOADING` = `true`.
   Keep the others; FluxRT-only ones are ignored.
5. Container disk: at least **60 GB** (the dual image is large).
6. Save. Workers restart; anyone mid-session is cut off.

First JoyAI start compiles kernels into the volume's cache and is slow. Warm it once:
Cloudflare Worker `POST /admin/load` (orchestrator key), or call the endpoint's
`/warmup?wait=true` with an `X-Session-Token`. `/ping` answers 204 while loading, 200 when ready.

## Switch back to FluxRT (fallback)

Edit endpoint → `LIVECAM_ENGINE` = `fluxrt` (or delete it) → tick **48 GB** and untick **32 GB PRO**
→ the volume can stay attached (FluxRT ignores it) → Save.

## Notes

- **Max workers must be ≥ 1** or nothing runs (the endpoint showed 0).
- **`PING_READY_WHILE_LOADING=true` is needed for JoyAI.** RunPod stops a load-balancing worker
  whose `/ping` still answers 204 after about 8 minutes, and a first JoyAI start (48 GB of weights
  plus kernel compile) takes longer, so the worker restarts forever. With this set, `/ping`
  answers 200 while loading and a session that arrives early gets `warming` until JoyAI is ready.
  Errors still answer 503.
- JoyAI serves **one session per worker**; raise Max workers for more shoppers.
- Unknown `LIVECAM_ENGINE` values stop the container instead of guessing.
- `ACCEPTED_TIERS` (default `standard,premium`) lets either engine accept tokens for either
  tier, so flipping needs no Worker change. Set `ACCEPTED_TIERS=premium` on a dedicated
  Premium endpoint if you later run both engines side by side (same image, two endpoints).
- `MAX_SESSION_SECONDS` (default 300) caps streaming time in both engines.
- The old FluxRT-only tags (`fluxrt-playback-scheduler-v1`, `fluxrt-bootstrap-v1`, …) are
  untouched; switching the image back to one of them is the full rollback.
- YOLOv8n person gate is off (`ENABLE_PERSON_GATE=false`): it's AGPL-3.0. The face gate stays on.

See `joyai/README.md` for the adapter, message mapping, env vars and licenses, and
`joyai/RUNPOD_SETUP.md` for creating and filling the volume.

## Option B: two endpoints from the same image (Standard and Premium side by side)

Same image, no code change: a second Load Balancing endpoint with `LIVECAM_ENGINE=joyai`.
Then the Cloudflare Worker's `PREMIUM_RUNPOD_LB_ENDPOINT_ID` points at it, Standard keeps
running on the existing endpoint, and the app's Premium HD button reaches JoyAI.

| Premium endpoint setting | Value |
|---|---|
| Type | Load Balancing (same as Standard), HTTP port **8765** |
| Image | `ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1-<sha>` + GHCR registry credentials (package is private) |
| GPU | **32 GB PRO** (RTX 5090) only |
| Max workers | your concurrent Premium sessions (one shopper per GPU) |
| Active workers | 1 during testing hours, 0 otherwise (a warm worker bills all the time) |
| Idle timeout | 300 s (longer than Standard, so shoppers don't hit cold starts) |
| FlashBoot | On |
| Network volume | `azyah-joyai` (locks the endpoint to that datacenter) |
| Container disk | start at 60 GB; lower it once you see the pulled image size |
| Env | `LIVECAM_ENGINE=joyai`, `SESSION_SIGNING_SECRET` (same as the Worker, as a RunPod Secret), `ACCEPTED_TIERS=premium`, `MAX_SESSION_SECONDS=300` |

## JoyAI upgrade switches (2026-11: upstream main, face++, prompt enhancement)

The image now pins JoyAI upstream **main `45ff66d9`** (6 Nov 2026) instead of `d875895`. What that
brings: the "cached attention consistency" fix (2 Nov), prompt enhancement with an external API backend,
the full-body entry gate, and `JOYOMNI_LOAD_WARMUP_TIMEOUT_S` (so our old warm-up-timeouts patch is gone).
**With every switch below at its default, the worker behaves exactly as before:** checkpoint 0811, no
prompt enhancement (keys unset), the app's wording, the near face gate.

| Switch (endpoint env) | Default | On | Turn off again |
|---|---|---|---|
| `JOYOMNI_DIT_CKPT` | unset = `/runpod-volume/joyai/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth` | `=/runpod-volume/joyai/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth` serves upstream's newer "face++" DiT (same layout as 0811; upstream main's default since Nov 2026) | delete the variable |
| `JOYAI_USE_PE` | `0` | `1`: JoyAI's own prompt rewriter looks at the garment photo and the first camera frames once per start/swap and rewrites the instruction (layers, sleeves, colours, what to keep). Needs the three PE variables below. Adds 2–8 s to a start/swap; after `JOYAI_PE_TIMEOUT_S` (20) it falls back to the raw instruction | `0` |
| `OPENAI_BASE_URL` / `PE_MODEL` / `OPENAI_API_KEY` | unset | the PE endpoint: OpenAI-compatible, e.g. Gemini's `https://generativelanguage.googleapis.com/v1beta/openai` with `PE_MODEL=gemini-2.5-flash` and the app's existing Gemini key as a **RunPod secret**; or an Anthropic-protocol URL (contains `/anthropic`). If any is empty, PE stays off with a warning. Values are never logged | unset |
| `JOYOMNI_PE_BACKEND` | `api` when PE is on | leave; `qwen` would load a 23 GB local model (96 GB card only) | — |
| `JOYAI_REF_WORDING` | `reference` | `image1`: "the garment in the reference image" → "the garment from Image 1" (an older upstream try-on tip). Only matters with PE off: PE's output always says "the reference image" | `reference` |
| `JOYAI_DETECTION_MODE` | `near` | `full`: upstream's full-body entry gate for sessions that have a face gate (`JOYAI_FULL_BODY_GATE=1` makes it the server default too). The app's per-garment `face_gate=false` still turns every gate off | `near` |
| `JOYAI_FULL_BODY_GATE` | unset | `1`: start JoyAI with `--full-body-detection` | unset |
| `JOYOMNI_FA4_ATTN` | `0` | upstream tries FlashAttention-4 first; `1` only on B200 | `0` |
| `JOYOMNI_LOAD_WARMUP_TIMEOUT_S` | `900` | boot warm-up wait (upstream default 600) | — |
| `JOYAI_KV_RESET_FRAMES` | `600` | `0` = never restart the edit on a timer (sessions are ≤ 5 min; JoyAI only keeps its last 8 chunks); `2400` ≈ every 100 s. 600 restarted the edit about every 25 s (owner test, 7 Nov) | `600` |
| `JOYAI_FACE_GATE_SCORE` | unset (JoyAI 0.35) | e.g. `0.25`: finds faces shaded by a cap or in low light, so a cap no longer looks like "person left" | delete |
| `JOYAI_PRESENCE_ABSENT_FRAMES` | unset (JoyAI 12 ≈ 0.5 s) | e.g. `36` ≈ 1.5 s of an unclear face before the hold-and-restart | delete |
| `JOYAI_PERSON_COUNT_CHANGE_FRAMES` | unset (JoyAI 24 ≈ 1 s) | e.g. `72` ≈ 3 s a new face count must hold before a re-edit | delete |
| `JOYAI_PRESENCE_RETURN_FRAMES` / `JOYAI_FACE_PRESENT_MIN_RATIO` | unset (JoyAI 24 / 0.15) | advanced; leave unset | delete |

**face++ on the volume.** azyah-joyai is 100 GB and already full (0811 32.5 GB + VAE + MiMo-VL 16 GB +
compile caches). Resize it first in RunPod (Storage → azyah-joyai → Edit; volumes grow, never shrink;
150 GB is enough), then on a one-off Pod with the volume at `/workspace`:
`FACEPP=1 VOLUME_ROOT=/workspace/joyai bash joyai/scripts/populate_volume.sh`. The script checks free space,
names the shortfall and refuses otherwise. **Never delete 0811: it is the rollback.** Optional check on the
same Pod: `python3 joyai/scripts/check_dit_checkpoint.py <face++> --compare <0811>` must print
"same layout" (894 tensors). face++ was added upstream on 1 Nov 2026 (HF commit 64ebcad, "Add Face++ video
editing checkpoint"); it has no release notes yet, so the owner's phone test is the acceptance test.

**Rollback:** set `JOYOMNI_DIT_CKPT` back / delete it, `JOYAI_USE_PE=0`, `JOYAI_REF_WORDING=reference`,
`JOYAI_DETECTION_MODE=near`, delete `JOYAI_FULL_BODY_GATE`; workers restart in minutes. Full rollback of the
upstream code = the previous image tag.

### Who is in view: how JoyAI behaves per try-on case

The app sends `face_gate=false` in `set_prompt` whenever no face is expected (shoes, bottoms, bags,
accessories, the back camera); the adapter turns that into `gate_enabled=false` **and** `detection_mode=off`,
which on upstream main disables the entry gate, the full-body gate and the mid-session "no person" hold
(`entry_gate = detection_mode != "off"`; `gate_on = entry_gate`). So:

| Case | Face in view? | Gate | What happens |
|---|---|---|---|
| Shoes, camera at the feet | no | off | edits start at once; no face needed, no black hold if the face never appears |
| Bag | no / partial | off | same |
| Watch, ring, close-up | no | off | same |
| Trousers, lower body only | no | off | same |
| Top, upper body in view | yes | near (default) or full | waits for a centred, still face (1 s), then edits; the person-presence hold blacks out if the face leaves for ~0.5 s |
| Mirror selfie (phone in hand) | yes | near / full | as a top; the phone stays (it is not clothing) |
| Two people side by side | yes | near / full | the gate skips the centring check for 2+ comparable faces; a later third person triggers a re-edit after 1 s |

Nothing here needs a shoe or garment mask; the category rules live in the app.

## JoyAI tuning (Lucy-VTON-like feel)

JoyAI reaches real-time with 2 denoising steps, FP8 weights and short chunks, keeping the
first chunk as an anchor so the person doesn't drift. JD reports ~30 fps at 720×1280 on a
B200; there are no published RTX 5090 numbers, so treat these as starting points.

| Knob | Setting | Default here | Notes |
|---|---|---|---|
| Denoising steps | `JOYAI_STEPS` | `2` | The distilled real-time path |
| FP8 | `JOYOMNI_FP8_IMG` / `JOYOMNI_FP8_TXT` | `1` / `1` | Plus `JOYOMNI_FP8_FAST_ACCUM=1` on the 5090 |
| Compile / CUDA graphs | `JOYOMNI_CUDA_GRAPH` | `1` | torch.compile caches persist on the volume |
| Attention / precision | `JOYOMNI_SAGE_ATTN`, `JOYOMNI_FP8_FAST_ACCUM`, `JOYOMNI_LOW_VRAM` | per GPU | 5090: int8 SageAttention, fp16 accumulate, low-VRAM (needed). RTX PRO 6000: cuDNN, fp32 accumulate, full layout (more faithful). Picked by `joyai/start_upstream.sh` |
| Output JPEG | `JOYAI_OUTPUT_QUALITY` | `90` | Was 70, which blurred prints and logos |
| Camera frames | sent to the app in `ready` | 480×840, 24 fps, JPEG 0.85 | The app adopts these when the engine is JoyAI |
| Mode | reference image in `start` | on | Garment photo drives the edit (IV2V) |
| Garment swap | app → `set_reference_image` + `set_prompt` | live | Coalesced into one JoyAI restart on the same socket, no reconnect |
| Prompt | app's `instruction` | "Substitute the current top with the garment in the reference image. …" | Optional wrapper: `JOYAI_PROMPT_TEMPLATE` (default `{prompt}`) |
| Chunk size / first-chunk anchor | built into JoyAI | native | Not exposed as settings upstream |
| Recent-chunk memory | `JOYAI_MAX_TEMPORAL_IDS` | `8` | Smaller = faster, larger = steadier |
| KV reset | `JOYAI_KV_RESET_FRAMES` | `600` | |
| Resolution | `JOYAI_SESSION_WIDTH/HEIGHT` | `480` / `840` | For 720p also set `JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720`, session `720`×`1248`; only if the 5090 holds 24 fps |
| Boot warm-up | built into JoyAI's load | on | `POST /load` runs a 4-chunk warm-up in both orientations before `/ping` turns 200 |
| CUDA / PyTorch | image | CUDA 12.8, torch 2.9.1 cu128 | Blackwell (sm_120a) |

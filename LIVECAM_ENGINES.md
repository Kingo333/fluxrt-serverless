# Live Cam engines: switch FluxRT ⇄ JoyAI on one RunPod endpoint

One image, `ghcr.io/kingo333/fluxrt-serverless:livecam-dual-v1-<sha>`, holds both engines.
The `LIVECAM_ENGINE` environment variable picks which one a worker runs.

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

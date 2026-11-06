#!/usr/bin/env bash
# Starts unmodified JoyAI-Video-Edit on loopback for the adapter.
# Mirrors deploy/run_server.sh, except:
#   - binds 127.0.0.1:8080 (only the adapter is public)
#   - --no-preload: the adapter loads the model via POST /load so /ping can report 204
#   - NO --record-dir: shoppers' camera streams are never written to disk
#   - prompt enhancement off unless JOYAI_USE_PE=1, mjpeg both ways
set -euo pipefail

JOYAI_DIR="${JOYAI_DIR:-/opt/JoyAI-Video-Edit/deploy}"
VOL="${JOYAI_VOLUME_ROOT:-/runpod-volume/joyai}"
CKPT_ROOT="${JOYOMNI_CKPT_ROOT:-$VOL/checkpoints}"
# In the dual-engine image plain `python` is FluxRT's 3.12; JoyAI has its own venv.
PY="${JOYAI_PYTHON:-python}"

# Per-GPU defaults, from upstream DEPLOYMENT.md:
#   RTX 5090 (32 GB): int8 SageAttention, fp16-accumulate FP8 GEMMs, low-VRAM layout (needed to fit).
#   48 GB+ cards (RTX PRO 6000 ...): cuDNN attention, fp32 accumulate, full layout:
#   higher numeric fidelity at the same 480p @ 24 FPS.
# Anything set on the endpoint wins over these.
GPU_INFO="$("$PY" -c 'import torch; p = torch.cuda.get_device_properties(0); print(f"{p.name}|{p.total_memory >> 30}")' 2>/dev/null || echo "unknown|0")"
GPU_NAME="${GPU_INFO%|*}"
GPU_GIB="${GPU_INFO##*|}"
if [[ "$GPU_NAME" != *5090* ]] && [ "${GPU_GIB:-0}" -ge 46 ]; then
  GPU_PROFILE=full; PROFILE_LOW_VRAM=0; PROFILE_SAGE=0; PROFILE_FAST_ACCUM=0; PROFILE_CACHE=cache_full
else
  GPU_PROFILE=rtx5090; PROFILE_LOW_VRAM=1; PROFILE_SAGE=1; PROFILE_FAST_ACCUM=1; PROFILE_CACHE=cache_rtx5090
fi
echo "[start_upstream] GPU: $GPU_NAME (${GPU_GIB} GiB) -> profile $GPU_PROFILE"
CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$VOL/$PROFILE_CACHE}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/nv_compute}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$JOYAI_DIR"
export JOYOMNI_FP8_IMG="${JOYOMNI_FP8_IMG:-1}"
export JOYOMNI_FP8_TXT="${JOYOMNI_FP8_TXT:-1}"
export JOYOMNI_CUDA_GRAPH="${JOYOMNI_CUDA_GRAPH:-1}"
export JOYOMNI_SAGE_ATTN="${JOYOMNI_SAGE_ATTN:-$PROFILE_SAGE}"
export JOYOMNI_FP8_FAST_ACCUM="${JOYOMNI_FP8_FAST_ACCUM:-$PROFILE_FAST_ACCUM}"
export JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-$PROFILE_LOW_VRAM}"
# The app always sends portrait 480x840, so skip compiling/warming landscape (cold start).
# Set "both" on the endpoint if a landscape client is ever added.
export JOYOMNI_WARMUP_ORIENTATION="${JOYOMNI_WARMUP_ORIENTATION:-portrait}"
# Upstream main (45ff66d): FA4 is tried first unless 0; it needs B200 (DEPLOYMENT.md says 0 on
# RTX 5090 / RTX PRO 6000). The boot warm-up wait is now an upstream knob (was our patch).
export JOYOMNI_FA4_ATTN="${JOYOMNI_FA4_ATTN:-0}"
export JOYOMNI_LOAD_WARMUP_TIMEOUT_S="${JOYOMNI_LOAD_WARMUP_TIMEOUT_S:-900}"
echo "[start_upstream] SAGE_ATTN=$JOYOMNI_SAGE_ATTN FP8_FAST_ACCUM=$JOYOMNI_FP8_FAST_ACCUM LOW_VRAM=$JOYOMNI_LOW_VRAM FA4=$JOYOMNI_FA4_ATTN warmup=$JOYOMNI_WARMUP_ORIENTATION warmup_wait=${JOYOMNI_LOAD_WARMUP_TIMEOUT_S}s cache=$CACHE_ROOT"

# Prompt enhancement (JOYAI_USE_PE=1): JoyAI's own prompt rewriter looks at the garment photo
# and the first camera frames once per session start / garment swap, through an external
# OpenAI-compatible (or Anthropic-protocol) endpoint: OPENAI_BASE_URL + PE_MODEL + OPENAI_API_KEY
# (a RunPod secret). Default off: today's behaviour, keys unset so nothing leaks in.
PE_ARGS=(--no-use-pe --pe-backend none)
if [ "${JOYAI_USE_PE:-0}" = "1" ]; then
  if [ -n "${OPENAI_BASE_URL:-}" ] && [ -n "${PE_MODEL:-}" ] && [ -n "${OPENAI_API_KEY:-}" ]; then
    PE_ARGS=(--use-pe --pe-backend "${JOYOMNI_PE_BACKEND:-api}" --pe-model "$PE_MODEL" --pe-timeout-s "${JOYAI_PE_TIMEOUT_S:-20}")
    echo "[start_upstream] prompt enhancement ON: backend=${JOYOMNI_PE_BACKEND:-api} model=$PE_MODEL base_url=$OPENAI_BASE_URL timeout=${JOYAI_PE_TIMEOUT_S:-20}s (key: set)"
  else
    echo "[start_upstream] WARNING: JOYAI_USE_PE=1 but OPENAI_BASE_URL, PE_MODEL or OPENAI_API_KEY is empty: starting with prompt enhancement OFF" >&2
    unset OPENAI_API_KEY OPENAI_BASE_URL PE_MODEL
  fi
else
  unset OPENAI_API_KEY OPENAI_BASE_URL PE_MODEL
fi
# Full-body entry gate (upstream main): person-box geometry instead of the near face gate.
GATE_ARGS=()
if [ "${JOYAI_FULL_BODY_GATE:-0}" = "1" ]; then
  GATE_ARGS=(--full-body-detection)
fi

DEFAULT_DIT="${JOYOMNI_DIT_CKPT:-$CKPT_ROOT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth}"
# JOYAI_MODEL_VERSION picks a trained model from the registry on the volume (staging);
# unset or joyai-baseline = the deployed upstream model. Unknown versions stop the worker.
MODEL_VERSION="${JOYAI_MODEL_VERSION:-joyai-baseline}"
REGISTRY="${JOYAI_MODEL_REGISTRY:-$VOL/models/registry.json}"
DIT_CKPT="$("$PY" "$(dirname "$0")/tools/resolve_model.py" "$MODEL_VERSION" "$REGISTRY" "$DEFAULT_DIT")" || {
  echo "[start_upstream] cannot load model version $MODEL_VERSION (see above)" >&2; exit 2; }
# Which DiT: 0811 (default) or e.g. JOYOMNI_DIT_CKPT=$CKPT_ROOT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth
# (upstream main's default; populate_volume.sh FACEPP=1 downloads it next to 0811).
echo "[start_upstream] model version: $MODEL_VERSION -> $DIT_CKPT ($(stat -c %s "$DIT_CKPT" 2>/dev/null || echo '?') bytes)"
VAE_CKPT="${JOYOMNI_VAE_CKPT:-$CKPT_ROOT/JoyAI-Video-Edit/vae}"
TE_CKPT="${JOYOMNI_TEXT_ENCODER_CKPT:-$CKPT_ROOT/MiMo-VL-7B-RL-2508}"
FACE_ONNX="${JOYOMNI_FACE_ONNX:-$CKPT_ROOT/face_detection_yunet_2023mar.onnx}"
# YOLOv8n is AGPL-3.0: off unless ENABLE_PERSON_GATE=true. A missing file disables that gate upstream.
if [ "${ENABLE_PERSON_GATE:-false}" = "true" ]; then
  PERSON_ONNX="${JOYOMNI_PERSON_ONNX:-$CKPT_ROOT/yolov8n.onnx}"
else
  PERSON_ONNX="/nonexistent/yolov8n.onnx"
fi

for f in "$DIT_CKPT" "$VAE_CKPT" "$TE_CKPT"; do
  if [ ! -e "$f" ]; then
    echo "[start_upstream] missing weights: $f (run scripts/populate_volume.sh on the network volume)" >&2
    exit 2
  fi
done

DEVICE="${JOYOMNI_DEVICE:-cuda:0}"
cd "$JOYAI_DIR"
exec "$PY" xvideo/serving/serve_joyomni_streaming.py \
  --dit-ckpt "$DIT_CKPT" \
  --vae-ckpt "$VAE_CKPT" \
  --text-encoder-ckpt "$TE_CKPT" \
  --face-detector-onnx "$FACE_ONNX" \
  --person-detector-onnx "$PERSON_ONNX" \
  --device "$DEVICE" \
  --vae-encode-device "$DEVICE" \
  --vae-decode-device "$DEVICE" \
  --vae-pseudo-device "$DEVICE" \
  --postprocess-device "$DEVICE" \
  --width "${JOYOMNI_WIDTH:-840}" --height "${JOYOMNI_HEIGHT:-480}" \
  --fps "${JOYOMNI_FPS:-24}" \
  --num-inference-steps "${JOYAI_STEPS:-2}" \
  --kv-reset-frames "${JOYAI_KV_RESET_FRAMES:-600}" \
  "${PE_ARGS[@]}" \
  ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} \
  --no-preload \
  --uplink-codec mjpeg --downlink-codec mjpeg \
  --host 127.0.0.1 --port "${JOYAI_UPSTREAM_PORT:-8080}"

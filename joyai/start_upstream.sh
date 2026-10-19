#!/usr/bin/env bash
# Starts unmodified JoyAI-Video-Edit on loopback for the adapter.
# Mirrors deploy/run_server.sh, except:
#   - binds 127.0.0.1:8080 (only the adapter is public)
#   - --no-preload: the adapter loads the model via POST /load so /ping can report 204
#   - NO --record-dir: shoppers' camera streams are never written to disk
#   - prompt enhancement off, mjpeg both ways
set -euo pipefail

JOYAI_DIR="${JOYAI_DIR:-/opt/JoyAI-Video-Edit/deploy}"
VOL="${JOYAI_VOLUME_ROOT:-/runpod-volume/joyai}"
CKPT_ROOT="${JOYOMNI_CKPT_ROOT:-$VOL/checkpoints}"
CACHE_ROOT="${JOYOMNI_CACHE_ROOT:-$VOL/cache_rtx5090}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/nv_compute}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$JOYAI_DIR"
# RTX 5090 settings from upstream DEPLOYMENT.md (480p @ 24 FPS, fits 32 GB).
export JOYOMNI_FP8_IMG="${JOYOMNI_FP8_IMG:-1}"
export JOYOMNI_FP8_TXT="${JOYOMNI_FP8_TXT:-1}"
export JOYOMNI_CUDA_GRAPH="${JOYOMNI_CUDA_GRAPH:-1}"
export JOYOMNI_SAGE_ATTN="${JOYOMNI_SAGE_ATTN:-1}"
export JOYOMNI_FP8_FAST_ACCUM="${JOYOMNI_FP8_FAST_ACCUM:-1}"
export JOYOMNI_LOW_VRAM="${JOYOMNI_LOW_VRAM:-1}"
# D3: no prompt enhancement. Make sure no key leaks in from the environment.
unset OPENAI_API_KEY OPENAI_BASE_URL PE_MODEL

DIT_CKPT="${JOYOMNI_DIT_CKPT:-$CKPT_ROOT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth}"
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
# In the dual-engine image plain `python` is FluxRT's 3.12; JoyAI has its own venv.
PY="${JOYAI_PYTHON:-python}"
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
  --no-use-pe \
  --no-preload \
  --uplink-codec mjpeg --downlink-codec mjpeg \
  --host 127.0.0.1 --port "${JOYAI_UPSTREAM_PORT:-8080}"

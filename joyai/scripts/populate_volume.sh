#!/usr/bin/env bash
# Download every JoyAI weight onto the RunPod network volume. Run ONCE from a
# one-off pod that has the volume attached (pods mount it at /workspace;
# serverless workers see the same volume at /runpod-volume).
#
#   bash populate_volume.sh                 # DiT 0811 + VAE + MiMo-VL + YuNet (~51 GB)
#   bash populate_volume.sh --with-yolo     # also export YOLOv8n (AGPL-3.0, see README)
#
# Env: VOLUME_ROOT (default /workspace/joyai), HF_TOKEN (optional).
set -euo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/workspace/joyai}"
CKPT="$VOLUME_ROOT/checkpoints"
WITH_YOLO=false
[ "${1:-}" = "--with-yolo" ] && WITH_YOLO=true

mkdir -p "$CKPT" "$VOLUME_ROOT/cache_rtx5090"
df -h "$VOLUME_ROOT" | tail -1

python3 -m pip install -q --upgrade "huggingface_hub[cli]>=0.34"

echo "== 1/3 DiT (0811) + xVAE"
hf download jdopensource/JoyAI-Video-Edit \
  --repo-type model \
  --local-dir "$CKPT/JoyAI-Video-Edit" \
  --include "dit/joyai_video_edit_dit_0811.pth" "vae/*"

echo "== 2/3 MiMo-VL-7B-RL-2508 (text/vision encoder)"
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 \
  --repo-type model \
  --local-dir "$CKPT/MiMo-VL-7B-RL-2508"

echo "== 3/3 YuNet face detector (MIT)"
curl -fL -o "$CKPT/face_detection_yunet_2023mar.onnx" \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

if $WITH_YOLO; then
  echo "== optional: YOLOv8n person detector, exported at imgsz=320 in a throwaway venv"
  python3 -m venv /tmp/yolo-export
  /tmp/yolo-export/bin/pip install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    ultralytics onnx onnxslim opencv-python-headless
  (cd /tmp && /tmp/yolo-export/bin/python -c \
    "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12)")
  mv /tmp/yolov8n.onnx "$CKPT/yolov8n.onnx"
  rm -rf /tmp/yolo-export
fi

echo "== verify"
for f in \
  "$CKPT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth" \
  "$CKPT/JoyAI-Video-Edit/vae/config.json" \
  "$CKPT/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors" \
  "$CKPT/MiMo-VL-7B-RL-2508/config.json" \
  "$CKPT/face_detection_yunet_2023mar.onnx"; do
  if [ -s "$f" ]; then echo "ok      $f"; else echo "MISSING $f"; exit 1; fi
done
du -sh "$CKPT"
echo "Done. Weights are on the volume; the serverless worker reads them from /runpod-volume/joyai/checkpoints."

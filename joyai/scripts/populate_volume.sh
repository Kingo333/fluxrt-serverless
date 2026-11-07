#!/usr/bin/env bash
# Download every JoyAI weight onto the RunPod network volume. Run ONCE from a
# one-off pod that has the volume attached (pods mount it at /workspace;
# serverless workers see the same volume at /runpod-volume).
#
#   bash populate_volume.sh                 # DiT 0811 + VAE + MiMo-VL + YuNet (~51 GB)
#   bash populate_volume.sh --with-yolo     # also export YOLOv8n (AGPL-3.0, see README)
#   FACEPP=1 bash populate_volume.sh        # ONLY the face++ DiT (31 GB) next to 0811, nothing else
#
# Env: VOLUME_ROOT (default /workspace/joyai), HF_TOKEN (optional).
# face++ = dit/joyai_video_edit_dit_1101_face++.pth (upstream main's default since 2026-11,
# "Add Face++ video editing checkpoint", HF commit 64ebcad). 0811 stays: it is the rollback.
# The 100 GB azyah-joyai volume is full with 0811 + VAE + MiMo-VL + caches: resize it first
# (RunPod volumes grow, never shrink; 150 GB is enough). Never delete 0811 to make room.
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1

VOLUME_ROOT="${VOLUME_ROOT:-/workspace/joyai}"
CKPT="$VOLUME_ROOT/checkpoints"
WITH_YOLO=false
[ "${1:-}" = "--with-yolo" ] && WITH_YOLO=true

mkdir -p "$CKPT" "$VOLUME_ROOT/cache_rtx5090"
df -h "$VOLUME_ROOT" | tail -1

# The Hugging Face CLI needs pip. Some pod images (runpod/base) have no pip: then the face++
# download (FACEPP=1) falls back to curl from the public resolve URL, resumable with -C -.
HAVE_HF=0
if python3 -m pip --version >/dev/null 2>&1; then
  python3 -m pip install -q --upgrade "huggingface_hub[cli]>=0.34" && HAVE_HF=1
fi
command -v hf >/dev/null 2>&1 || HAVE_HF=0

if [ "${FACEPP:-0}" = "1" ]; then
  FACEPP_FILE="$CKPT/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth"
  FACEPP_BYTES=32527642399
  if [ -s "$FACEPP_FILE" ] && [ "$(stat -c %s "$FACEPP_FILE")" -eq "$FACEPP_BYTES" ]; then
    echo "face++ already on the volume: $FACEPP_FILE"; exit 0
  fi
  FREE_BYTES=$(( $(df -P --block-size=1 "$VOLUME_ROOT" | awk 'NR==2 {print $4}') ))
  NEED_BYTES=$(( FACEPP_BYTES + 1073741824 ))   # the file + 1 GB headroom
  if [ "$FREE_BYTES" -lt "$NEED_BYTES" ]; then
    echo "NOT ENOUGH SPACE for face++: need $(( NEED_BYTES / 1073741824 )) GB free, have $(( FREE_BYTES / 1073741824 )) GB" \
         "(short by $(( (NEED_BYTES - FREE_BYTES) / 1073741824 + 1 )) GB)." >&2
    echo "Resize the volume in RunPod (Storage -> azyah-joyai -> Edit; volumes grow, never shrink) and run again." >&2
    echo "Do NOT delete 0811: it is the rollback checkpoint." >&2
    exit 3
  fi
  echo "== face++ DiT (31 GB) -> $FACEPP_FILE"
  if [ "$HAVE_HF" = "1" ]; then
    hf download jdopensource/JoyAI-Video-Edit \
      --repo-type model \
      --local-dir "$CKPT/JoyAI-Video-Edit" \
      dit/joyai_video_edit_dit_1101_face++.pth
  else
    echo "(no pip / hf CLI on this pod: downloading with curl, resumable)"
    mkdir -p "$(dirname "$FACEPP_FILE")"
    curl -fL --retry 5 --retry-delay 5 -C - -o "$FACEPP_FILE" \
      "https://huggingface.co/jdopensource/JoyAI-Video-Edit/resolve/main/dit/joyai_video_edit_dit_1101_face%2B%2B.pth"
  fi
  # A missing file counts as 0 bytes (stat on a missing file would otherwise make the test
  # error out and the script report "Done").
  GOT_BYTES=$(stat -c %s "$FACEPP_FILE" 2>/dev/null || echo 0)
  if [ "$GOT_BYTES" -ne "$FACEPP_BYTES" ]; then
    echo "face++ download is incomplete ($GOT_BYTES of $FACEPP_BYTES bytes); run again" >&2; exit 4
  fi
  ls -la "$CKPT/JoyAI-Video-Edit/dit/"
  df -h "$VOLUME_ROOT" | tail -1
  echo "Done. Serve it with JOYOMNI_DIT_CKPT=/runpod-volume/joyai/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_1101_face++.pth"
  exit 0
fi

if [ "$HAVE_HF" != "1" ]; then
  echo "The full download needs pip and the Hugging Face CLI. Use a pod template with Python and pip" \
       "(e.g. RunPod PyTorch), or FACEPP=1 for the face++ file alone (works with curl)." >&2
  exit 5
fi

echo "== 1/3 DiT (0811) + xVAE"
hf download jdopensource/JoyAI-Video-Edit \
  --repo-type model \
  --local-dir "$CKPT/JoyAI-Video-Edit" \
  dit/joyai_video_edit_dit_0811.pth vae/config.json vae/diffusion_pytorch_model.safetensors

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

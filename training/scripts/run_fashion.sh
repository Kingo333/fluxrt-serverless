#!/usr/bin/env bash
# Stage 1 fashion training run (paid; only with the owner's approval and a budget).
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/main/training/scripts/run_fashion.sh | \
#     GPUS=2 HSDP=2 AZ_MANIFEST_DIR=/workspace/azyah-vton/manifests/commercial-train AZ_EXP=fashion-exp1 AZ_MAX_STEPS=500 bash
# Auto-resumes from the latest checkpoint of the same AZ_EXP. Stops at AZ_MAX_STEPS.
set -euo pipefail
SCRIPTS_URL="${SCRIPTS_URL:-https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/main/training/scripts}"
source <(curl -fsSL "$SCRIPTS_URL/common.sh")
: "${AZ_MANIFEST_DIR:?set AZ_MANIFEST_DIR (output of data_tools.manifests)}"
GPUS="${GPUS:-1}"
export AZ_HSDP_SHARD="${HSDP:-$GPUS}" AZ_SP_SIZE="${SP:-1}" AZ_EXP="${AZ_EXP:-fashion}"
export JOYAI_OUTPUT_DIR="$AZ_OUT/stage1_fashion"
export PATH="$AZ_ROOT/venv/bin:$PATH"
[ -f "$AZ_MANIFEST_DIR/selection.json" ] && { echo "== data selection"; cat "$AZ_MANIFEST_DIR/selection.json"; }
curl -fsSL "$SCRIPTS_URL/make_fashion_config.py" -o "$AZ_ROOT/make_fashion_config.py"
python "$AZ_ROOT/make_fashion_config.py" "$AZ_UPSTREAM/training"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$AZ_REPORTS/fashion_${AZ_EXP}_$STAMP.log"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 10 > "${LOG%.log}_gpu.csv" &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null || true' EXIT
cd "$AZ_UPSTREAM/training"
bash scripts/train/stage1_ar.sh configs/azyah_fashion_stage1.py "$GPUS" 2>&1 | tee "$LOG"
echo "checkpoints:"; find "$JOYAI_OUTPUT_DIR" -name "*.pth" -printf "%p %s bytes\n" | sort | tail -5

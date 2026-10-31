#!/usr/bin/env bash
# Phase B smoke test: 20 Stage 1 steps on ToyData, then resume to 30 steps.
# Success = steps run (forward + backward), a checkpoint is saved, and resume continues from it.
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/scripts/run_smoke.sh | GPUS=4 HSDP=4 bash
# GPUS = processes (one per GPU). HSDP = how many GPUs share the model (shard); use GPUS for one node.
set -euo pipefail
# Pin every download to one commit (review R4). AZ_REF = a commit SHA; if unset, main's
# current SHA is looked up once and logged, so each run records exactly which code it used.
AZ_REF="${AZ_REF:-$(curl -fsSL https://api.github.com/repos/Kingo333/fluxrt-serverless/commits/main | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')}"
export AZ_REF
SCRIPTS_URL="${SCRIPTS_URL:-https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/scripts}"
echo "[azyah] scripts ref: $AZ_REF"
source <(curl -fsSL "$SCRIPTS_URL/common.sh")
GPUS="${GPUS:-1}"
export AZ_HSDP_SHARD="${HSDP:-$GPUS}"
export AZ_SP_SIZE="${SP:-1}"
export JOYAI_T2I_DATA="$AZ_DATA/toy/t2i.json" JOYAI_I2I_DATA="$AZ_DATA/toy/i2i.json"
export JOYAI_T2V_DATA="$AZ_DATA/toy/t2v.json" JOYAI_V2V_DATA="$AZ_DATA/toy/v2v.json"
export JOYAI_OUTPUT_DIR="$AZ_OUT/stage1_smoke"
export PATH="$AZ_ROOT/venv/bin:$PATH"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$AZ_REPORTS/smoke_$STAMP.log"
GPULOG="$AZ_REPORTS/smoke_${STAMP}_gpu.csv"

# GPU memory/utilisation every 5 s, for the peak-memory report.
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$GPULOG" &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null || true' EXIT

cd "$AZ_UPSTREAM/training"
echo "== run 1: 20 steps (GPUS=$GPUS HSDP=$AZ_HSDP_SHARD SP=$AZ_SP_SIZE)" | tee "$LOG"
start=$(date +%s)
AZ_MAX_STEPS=20 AZ_AUTO_RESUME=0 bash scripts/train/stage1_ar.sh configs/azyah_smoke_stage1.py "$GPUS" 2>&1 | tee -a "$LOG"
t1=$(( $(date +%s) - start ))

echo "== run 2: resume to 30 steps" | tee -a "$LOG"
start=$(date +%s)
AZ_MAX_STEPS=30 AZ_AUTO_RESUME=1 bash scripts/train/stage1_ar.sh configs/azyah_smoke_stage1.py "$GPUS" 2>&1 | tee -a "$LOG"
t2=$(( $(date +%s) - start ))

echo "== summary" | tee -a "$LOG"
{
  echo "run1_seconds=$t1 run2_seconds=$t2"
  echo "checkpoints:"; find "$JOYAI_OUTPUT_DIR" -name "*.pth" -printf "%p %s bytes\n" 2>/dev/null | sort
  echo "resume lines:"; grep -iE "auto-resume|resuming from" "$LOG" | tail -3
  echo "peak GPU memory (MiB) per GPU:"
  awk -F', ' 'NR>1 {gsub(/ MiB/,"",$3); if ($3+0>m[$2]) m[$2]=$3+0; t[$2]=$4} END {for (g in m) print "  gpu" g ": " m[g] " / " t[g]}' "$GPULOG"
  echo "errors:"; grep -E "Error|Traceback|out of memory|Killed" "$LOG" | tail -5 || echo "  none"
} | tee -a "$LOG"
echo "Logs: $LOG and $GPULOG"

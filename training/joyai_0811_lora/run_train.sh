#!/usr/bin/env bash
# 0811 LoRA: a LoRA on the CLONE of the live JoyAI model (joyai_video_edit_dit_0811.pth on the
# training volume), trained through the live 2-step streaming rollout against real targets.
# ONE GPU Pod (1x RTX PRO 6000, 96 GB), volume azyah-vton-train at /workspace. Paid: owner's "yes" first.
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/joyai_0811_lora/run_train.sh | \
#     AZ_REF=<SHA> AZ_EXP=joyai0811-exp2 bash
# Step 3a data (build_data.sh): AZ_WITH_VIVID_DRESSES=1 (default, owner's Option A) trains on
# manifests/joyai0811-research-train (Glance upper + lower + ViViD dress stills; research-only,
# commercial_use_cleared false); AZ_WITH_VIVID_DRESSES=0 trains the commercial-clean LoRA on
# manifests/joyai0811-commercial-train (Glance only). AZ_MANIFEST_DIR overrides both.
#
# Never touches the live endpoint, the live azyah-joyai volume (/runpod-volume) or the live image:
# the base must be a file on this volume, and the output is LoRA saves under outputs/joyai0811_lora.
# Resumable (same AZ_EXP continues from its newest save). Every save is kept (no pruning).
set -euo pipefail
AZ_REF="${AZ_REF:-$(curl -fsSL https://api.github.com/repos/Kingo333/fluxrt-serverless/commits/main | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')}"
export AZ_REF
RAW="${RAW_URL:-https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training}"
export SCRIPTS_URL="$RAW/scripts"
echo "[azyah] scripts ref: $AZ_REF"
source <(curl -fsSL "$RAW/scripts/common.sh")
# AZ_WITH_VIVID (or the older name AZ_WITH_VIVID_DRESSES): 1 = research manifest with the ViViD
# real-frame stills (dresses, tops, bottoms); 0 = the commercial-clean Glance-only LoRA.
WITH_VIVID_DRESSES="${AZ_WITH_VIVID:-${AZ_WITH_VIVID_DRESSES:-1}}"
if [ "$WITH_VIVID_DRESSES" = "1" ]; then
  SRC_MANIFESTS="${AZ_MANIFEST_DIR:-$AZ_ROOT/manifests/joyai0811-research-train}"; DATA_MODE="${AZ_DATA_MODE:-research}"
else
  SRC_MANIFESTS="${AZ_MANIFEST_DIR:-$AZ_ROOT/manifests/joyai0811-commercial-train}"; DATA_MODE="${AZ_DATA_MODE:-commercial}"
fi
[ -f "$SRC_MANIFESTS/i2i.json" ] || { echo "no $SRC_MANIFESTS/i2i.json: run build_data.sh (step 3b) first"; exit 1; }
STAGING_DIRS=("$AZ_DATA/staging/glance_upper" "$AZ_DATA/staging/glance_lower" "$AZ_DATA/staging/vivid_dress_stills"
              "$AZ_DATA/staging/vivid_upper_stills" "$AZ_DATA/staging/vivid_lower_stills")

export AZ_EXP="${AZ_EXP:-joyai0811-lora}"
export AZ_MAX_STEPS="${AZ_MAX_STEPS:-1000}"
export AZ_LR="${AZ_LR:-3e-5}"
export AZ_LORA_RANK="${AZ_LORA_RANK:-32}"
export AZ_LORA_ALPHA="${AZ_LORA_ALPHA:-$AZ_LORA_RANK}"
export AZ_CKPT_INTERVAL="${AZ_CKPT_INTERVAL:-250}"
export AZ_WARMUP="${AZ_WARMUP:-50}"
export AZ_STATIC_FRAMES="${AZ_STATIC_FRAMES:-25}"    # 8n+1; 25 = 4 latent chunks (sink + live window)
export AZ_EXIT="${AZ_EXIT:-random}"                  # random (upstream) | last (only the served output)
# Share of samples captioned with the app's full instruction + MiMo-VL garment details (when the
# pair has them); the rest get the plain instruction, as wardrobe items without a description do.
export AZ_DETAILED_CAPTION_PROB="${AZ_DETAIL_CAPTION_FRAC:-${AZ_DETAILED_CAPTION_PROB:-0.5}}"
export AZ_MASK_WEIGHT="${AZ_MASK_WEIGHT:-3.0}" AZ_EDGE_WEIGHT="${AZ_EDGE_WEIGHT:-0.5}"
DETAIL_OVERSAMPLE="${AZ_DETAIL_OVERSAMPLE:-2}"
export AZ_STAGE1_LORA=1 AZ_GRAD_ACCUM="${AZ_GRAD_ACCUM:-1}" AZ_HSDP_SHARD=1 AZ_SP_SIZE=1 AZ_AUTO_RESUME=1
# Real-motion clips (owner 2026-11-05: garment accuracy while the person moves and turns): the
# research manifest's v2v group holds ONLY ViViD clips whose input was made the live way and passed
# the steadiness check (build_data.sh packs with --only-live-source), never the flickering P8 clips.
VIVID_VIDEOS="${AZ_VIVID_VIDEOS:-$WITH_VIVID_DRESSES}"
CLIP_REPEAT="${AZ_CLIP_REPEAT:-3}"   # each clip used 3x (a new random 25-frame window each time)
if [ "$VIVID_VIDEOS" = "1" ]; then SAMPLES="${AZ_SAMPLES:-2500}"; else SAMPLES="${AZ_SAMPLES:-4000}"; fi   # still triples
TOOLS="$AZ_ROOT/azyah-tools/training"
DATA="$AZ_DATA/joyai0811_lora/$AZ_EXP"
export JOYAI_OUTPUT_DIR="$AZ_OUT/joyai0811_lora"
EXP_DIR="$JOYAI_OUTPUT_DIR/${AZ_EXP}_sp1_world1"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$AZ_REPORTS/joyai0811_${AZ_EXP}_$STAMP.log"
PY="$AZ_ROOT/venv/bin/python"

echo "== 1. env + upstream patches (stage1-lora, text-encoder-sdpa, dataset-video-bytes)"
curl -fsSL "$SCRIPTS_URL/setup_training_pod.sh" | bash

echo "== 2. base: the 0811 clone on the training volume"
BASE="${AZ_BASE_0811:-$(find "$AZ_CKPT" -name joyai_video_edit_dit_0811.pth 2>/dev/null | head -1)}"
[ -n "$BASE" ] && [ -f "$BASE" ] || { echo "0811 clone not found under $AZ_CKPT (set AZ_BASE_0811)"; exit 1; }
BASE="$(readlink -f "$BASE")"
case "$BASE" in
  /runpod-volume/*) echo "REFUSING: $BASE is on the live volume; use the clone on azyah-vton-train"; exit 1 ;;
  "$AZ_ROOT"/*) ;;
  *) echo "REFUSING: $BASE is not on the training volume ($AZ_ROOT)"; exit 1 ;;
esac
ls -la "$BASE"
export JOYAI_BASE_MODEL="$BASE"
echo "{\"joyai0811_lora\": \"$AZ_EXP\", \"base\": \"$BASE\", \"azyah_scripts_ref\": \"$AZ_REF\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$AZ_REPORTS/code_refs.jsonl"

echo "== 3. tools"
for f in data_tools/__init__.py data_tools/sources.py data_tools/shards.py data_tools/manifests.py \
         eval/__init__.py eval/metrics.py \
         scripts/make_fashion_config.py joyai_0811_lora/__init__.py joyai_0811_lora/live_schedule.py \
         joyai_0811_lora/losses.py joyai_0811_lora/make_config.py joyai_0811_lora/make_static_clips.py \
         joyai_0811_lora/train.py; do
  mkdir -p "$TOOLS/$(dirname "$f")"
  curl -fsSL "$RAW/$f" -o "$TOOLS/$f"
done
"$PY" -c "import av" 2>/dev/null || "$AZ_ROOT/venv/bin/pip" install -q av

echo "== 4. data: steady static clips from the still triples ($SAMPLES, ${AZ_STATIC_FRAMES} frames, 480x840)"
if [ ! -f "$DATA/build_summary.json" ]; then
  EXTRA=(); [ "$VIVID_VIDEOS" = "1" ] && EXTRA=(--vivid-videos --clip-repeat "$CLIP_REPEAT") && DATA_MODE=research
  "$PY" "$TOOLS/joyai_0811_lora/make_static_clips.py" --manifest-dir "$SRC_MANIFESTS" --out "$DATA" \
    --frames "$AZ_STATIC_FRAMES" --max "$SAMPLES" --mode "$DATA_MODE" "${EXTRA[@]}" \
    --staging-dirs "${STAGING_DIRS[@]}" --detail-oversample "$DETAIL_OVERSAMPLE"
fi
cat "$DATA/build_summary.json"
echo "== data selection (provenance)"; cat "$DATA/manifest/selection.json"
export AZ_MANIFEST_DIR="$DATA/manifest"   # the config reads v2v.json from here (no i2i: live is video)
export AZ_MASK_DIR="$DATA/masks"          # garment masks per sample key (losses.py)

echo "== 5. config + trainer"
AZ_FASHION_CONFIG_PY="$TOOLS/scripts/make_fashion_config.py" "$PY" "$TOOLS/joyai_0811_lora/make_config.py" "$AZ_UPSTREAM/training"
cp -f "$TOOLS/joyai_0811_lora/train.py" "$AZ_UPSTREAM/training/src/azyah_train_0811_lora.py"
export AZ_TOOLS_DIR="$TOOLS/joyai_0811_lora"
mkdir -p "$EXP_DIR"
cp -f "$DATA/manifest/selection.json" "$EXP_DIR/selection.json"

echo "== 6. train (resumes from the newest save of $AZ_EXP; all saves kept)"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 10 > "${LOG%.log}_gpu.csv" &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null || true' EXIT
cd "$AZ_UPSTREAM/training"
export PATH="$AZ_ROOT/venv/bin:$PATH"
TRAIN_EXIT=0
torchrun --nproc_per_node 1 -m src.azyah_train_0811_lora --config configs/azyah_joyai0811_lora.py 2>&1 | tee "$LOG" || TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"

echo "== 7. saves (each must hold adapter tensors)"
GOOD=0
for ck in $(find "$EXP_DIR/checkpoints" -name "step_*.pth" 2>/dev/null | sort -V); do
  n=$("$PY" -c "import sys,torch; print(len(torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True).get('generator_lora') or {}))" "$ck" 2>/dev/null || echo 0)
  echo "$ck adapter tensors: $n"
  [ "$n" -gt 0 ] && GOOD=$((GOOD + 1))
done
echo "GOOD_SAVES=$GOOD"
[ "$GOOD" -gt 0 ] || exit 1
exit "$TRAIN_EXIT"

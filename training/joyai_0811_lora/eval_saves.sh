#!/usr/bin/env bash
# Evaluate every 0811 LoRA save against the untouched 0811 clone, with the live settings
# (2 steps, no CFG, garment basesize 768, the app's full instruction) on steady REAL clips
# (--cross: a real ViViD test clip + another item's garment photo of the same category).
# Same Pod type as training (1x RTX PRO 6000) and the same volume. Paid: owner's "yes" first.
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/joyai_0811_lora/eval_saves.sh | \
#     AZ_REF=<SHA> AZ_EXP=joyai0811-exp1 bash
# Each save is fused onto 0811 at load (generate_vivid --base), so no 32 GB merged files are written.
# Writes $AZ_ROOT/eval/joyai0811_<exp>/ (clips, report.md, grid videos). Re-running skips finished clips.
#
# Step 3a: 16 clips (AZ_EVAL_ITEMS) that always start with the owner's regression set from step 1,
# with the SAME garments (--pairs-from the step 1 run: vivid_test_000000 belt, 000002 bow collar +
# print + doubled bag, 000010 ruffles/rib/hem, 000029), then 12 more across categories.
# Models: live0811 (baseline), live0811_ref1024 (same model, garment photo at basesize 1024: the
# resolution experiment; serving stays 768 unless the owner approves a change), then every save.
# Grids: all clips at 320 px, and the regression set at full height (832 px) for detail.
set -euo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
export AZ_REF
RAW="${RAW_URL:-https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training}"
export SCRIPTS_URL="$RAW/scripts"
source <(curl -fsSL "$RAW/scripts/common.sh")
: "${AZ_EXP:?set AZ_EXP to the trained experiment}"
export AZ_MANIFEST_DIR="${AZ_MANIFEST_DIR:-$AZ_ROOT/manifests/research-train}"
ITEMS="${AZ_EVAL_ITEMS:-16}"
PAIRS_FROM="${AZ_PAIRS_FROM:-$AZ_ROOT/eval/p9e_cross}"   # step 1's --cross run (cross.json per clip)
REGRESSION="${AZ_REGRESSION_KEYS:-vivid_test_000000 vivid_test_000002 vivid_test_000010 vivid_test_000029}"
EXP_DIR="$AZ_OUT/joyai0811_lora/${AZ_EXP}_sp1_world1"
OUT="$AZ_ROOT/eval/joyai0811_${AZ_EXP}"
PY="$AZ_ROOT/venv/bin/python"

BASE="${AZ_BASE_0811:-$(find "$AZ_CKPT" -name joyai_video_edit_dit_0811.pth 2>/dev/null | head -1)}"
BASE="$(readlink -f "$BASE")"
case "$BASE" in "$AZ_ROOT"/*) ;; *) echo "REFUSING: base $BASE is not the clone on the training volume"; exit 1 ;; esac
export JOYAI_BASE_MODEL="$BASE"   # never the bid pretrain checkpoint (it may be deleted to free space)

curl -fsSL "$SCRIPTS_URL/setup_training_pod.sh" | bash
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF"
A="$AZ_ROOT/azyah-code/training"
cd "$AZ_UPSTREAM/training"
"$PY" "$A/scripts/make_fashion_config.py" "$PWD"
export PYTHONPATH="$PWD:$A" PATH="$AZ_ROOT/venv/bin:$PATH"
G=(torchrun --nproc_per_node 1 -m eval.generate_vivid --config configs/azyah_fashion_stage1.py
   --staging "$AZ_DATA/staging/vivid" --out "$OUT" --max-items "$ITEMS" --live --cross)
[ -d "$PAIRS_FROM" ] && G+=(--pairs-from "$PAIRS_FROM" --motion-first 4) || echo "WARNING: $PAIRS_FROM missing; regression pairs not fixed"

MODELS=(live0811 live0811_ref1024)
"${G[@]}" --ckpt "$BASE" --name live0811
"${G[@]}" --ckpt "$BASE" --name live0811_ref1024 --ref-basesize 1024
for ck in $(find "$EXP_DIR/checkpoints" -name "step_*.pth" | sort -V); do
  step="$(basename "$ck" .pth)"; step="${step#step_}"
  "${G[@]}" --ckpt "$ck" --base "$BASE" --name "lora$step"
  MODELS+=("lora$step")
done
echo "GEN_DONE models: ${MODELS[*]}"
# Longer clips for drift (owner 2026-11-05): the 4 regression clips at 97 frames (24 chunks, ~4 s)
# for live0811 and every save. The 16-clip set stays at 49 frames.
LONG="${OUT}_long"
GL=(torchrun --nproc_per_node 1 -m eval.generate_vivid --config configs/azyah_fashion_stage1.py
    --staging "$AZ_DATA/staging/vivid" --out "$LONG" --max-items 8 --num-frames 97 --live --cross)
[ -d "$PAIRS_FROM" ] && GL+=(--pairs-from "$PAIRS_FROM" --motion-first 4)
LONG_MODELS=(live0811)
"${GL[@]}" --ckpt "$BASE" --name live0811
for ck in $(find "$EXP_DIR/checkpoints" -name "step_*.pth" | sort -V); do
  step="$(basename "$ck" .pth)"; step="${step#step_}"
  "${GL[@]}" --ckpt "$ck" --base "$BASE" --name "lora$step"
  LONG_MODELS+=("lora$step")
done
cd "$A"
"$PY" -c "import cv2" 2>/dev/null || "$AZ_ROOT/venv/bin/pip" install -q opencv-python-headless || true   # face_keep
"$PY" -m eval.run_eval "$OUT" --out "$OUT/report.md" --ref live0811; cat "$OUT/report.md"
"$PY" -m eval.grid "$OUT" "${OUT}_grid" --models "${MODELS[@]}" --require-all --max-items 8   # owner reviews 5-10 items; all 16 are scored in report.md
echo "grid videos: ${OUT}_grid (all.mp4 = every clip; panels: input | garment | ${MODELS[*]})"
# Motion clips (owner 2026-11-05): the 4 test clips that move most (turns, side/back views) are
# reviewed next to the owner's 4 regression clips.
MOTION="$(cut -d' ' -f1 "$OUT/motion_keys.txt" 2>/dev/null | tr '\n' ' ')"
echo "motion clips: $MOTION"
REGRESSION="$REGRESSION $MOTION"
REG="${OUT}_regression"
mkdir -p "$REG"
for k in $REGRESSION; do [ -d "$OUT/$k" ] && ln -sfn "$OUT/$k" "$REG/$k"; done
"$PY" -m eval.grid "$REG" "${OUT}_regression_grid" --models "${MODELS[@]}" --require-all --height 832
echo "regression grids (832 px): ${OUT}_regression_grid ($REGRESSION)"
# Every frame, for the owner: frame strips (every 6th frame + each row's worst frame in red).
STRIP_MODELS=(live0811 "${MODELS[@]:2}")
"$PY" -m eval.frame_strip "$OUT" "${OUT}_frames" --models "${STRIP_MODELS[@]}" --clips $REGRESSION --every 6 --height 240
"$PY" -m eval.run_eval "$LONG" --out "$LONG/report.md" --ref live0811; cat "$LONG/report.md"
"$PY" -m eval.frame_strip "$LONG" "${LONG}_frames" --models "${LONG_MODELS[@]}" --every 8 --height 200
echo "frame strips: ${OUT}_frames (49 frames) and ${LONG}_frames (97 frames)"

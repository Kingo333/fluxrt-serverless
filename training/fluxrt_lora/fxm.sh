#!/usr/bin/env bash
# Favourite #2 (owner, 2026-11-09): ONE LoRA file = fxreal3 and fxreal4 blended 50/50 at strength 0.65, exactly,
# then proven equal to the live blend by rendering the same frames both ways. Output folder (copy it to the
# endpoint's volume azyah-joyai as /runpod-volume/fluxrt_lora/blend05s065/):
#   /workspace/azyah-vton/outputs/fluxrt_lora/blend05s065/pytorch_lora_weights.safetensors  (+ selection.json, README.txt)
# Renders only (verification): any GPU with >= 20 GB. Volume azyah-vton-train at /workspace. Never stops the Pod.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fxm.sh -o /workspace/fxm.sh
#   AZ_REF=<SHA> bash /workspace/fxm.sh 2>&1 | tee /workspace/fxm.log
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
export AZ_REF AZ_ROOT=/workspace/azyah-vton
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
C="$AZ_ROOT/checkpoints"; PY="$AZ_ROOT/venv-fluxlora/bin/python"
L3="$AZ_ROOT/outputs/fluxrt_lora/fxreal3/pytorch_lora_weights.safetensors"
L4="$AZ_ROOT/outputs/fluxrt_lora/fxreal4/pytorch_lora_weights.safetensors"
OUT="$AZ_ROOT/outputs/fluxrt_lora/${AZ_MERGE_NAME:-blend05s065}"
CLIPS="$AZ_ROOT/eval/fxa/clips10.json"
START=$(date +%s); mins() { echo "$(( ($(date +%s) - START) / 60 )) min"; }

echo "== code at $AZ_REF"
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || { echo CODE_FAILED; exit 1; }
T="$AZ_ROOT/azyah-code/training"; export PYTHONPATH="$T"
for f in "$L3" "$L4" "$CLIPS"; do [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }; done

echo "== 1. merge: 0.5 x fxreal3 + 0.5 x fxreal4, strength 0.65 baked in ($(mins))"
( cd "$T" && "$PY" -m fluxrt_lora.merge_lora --inputs "$L3" "$L4" --weights 0.5 0.5 --scale "${AZ_MERGE_SCALE:-0.65}" \
    --out "$OUT" --selection-from "$AZ_ROOT/outputs/fluxrt_lora/fxreal3" \
    --note "Owner's favourite FluxRT option #2 (2026-11-09): blend 0.5 of fxreal3/fxreal4 at strength 0.65" ) || { echo MERGE_FAILED; exit 1; }
cat "$OUT/README.txt"

echo "== 2. proof: same frames rendered with the two LoRAs live-blended vs the one merged file ($(mins))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( cd "$T" && "$PY" -m fluxrt_lora.verify_merge --model-dir "$C/FLUX.2-klein-4B" --inputs "$L3" "$L4" --weights 0.5 0.5 \
    --scale "${AZ_MERGE_SCALE:-0.65}" --merged "$OUT/pytorch_lora_weights.safetensors" --clip-list "$CLIPS" --items 2 --frames 3 ) \
  && echo "MERGE_VERIFIED" || echo "MERGE_MISMATCH: do not ship this file"
echo "== 3. the folder to copy to azyah-joyai (as /runpod-volume/fluxrt_lora/$(basename "$OUT")/)"
ls -la "$OUT"; sha256sum "$OUT/pytorch_lora_weights.safetensors"
echo "FXM_DONE after $(mins)"

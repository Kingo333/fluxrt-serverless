#!/usr/bin/env bash
# The owner's 5-item re-test on the fxreal2 LoRA (2026-11-08). Two steps, run on a Pod with azyah-vton-train:
#   bash five_test.sh prompts [--desc N="..."] [--category N=...]   fix the case text, pick 5, PRINT the 5 prompts
#   bash five_test.sh render                                         render base, l2000, l1500; gate; one video
# FIVE_LORA=l1500 bash five_test.sh video   remakes the video with l1500 as the LoRA (no GPU work).
# Needs nothing from the network except this repo's raw files at $FIVE_REF. The pod is NOT stopped.
set -uo pipefail
FIVE_REF="${FIVE_REF:-feat/fluxrt-base-lora-real}"
AZ_ROOT=/workspace/azyah-vton
T="$AZ_ROOT/azyah-code/training"; E="$AZ_ROOT/eval/fx2"; R="$E/results"; C="$AZ_ROOT/checkpoints"
PY="$AZ_ROOT/venv-fluxlora/bin/python"; MP="$AZ_ROOT/venv-mp/bin/python"
OUT="$AZ_ROOT/outputs/fluxrt_lora/fxreal2"
W=/workspace/five; mkdir -p "$W"
RAW="https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$FIVE_REF/training/eval"
for f in fix_case_text.py five_test.py; do
  curl -fsSL "$RAW/$f" -o "$W/$f" || { echo "DOWNLOAD_FAILED $f"; exit 1; }
done
mode="${1:-prompts}"; shift || true
case "$mode" in
  prompts)
    python3 "$W/fix_case_text.py" "$AZ_ROOT" > "$W/fix_case_text.log" && echo "case text fixed (log: $W/fix_case_text.log)"
    rm -rf "$R/cases5"  # renders of an older prompt list must not be reused
    ( cd "$T" && "$PY" "$W/five_test.py" select "$AZ_ROOT" "$@" )
    ;;
  render)
    [ -f "$E/cases5/clip_list.json" ] || { echo "run: bash five_test.sh prompts  (first)"; exit 1; }
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --ref-size 288x512 --steps 2 --prompt-style compact --clip-list $E/cases5/clip_list.json --staging $AZ_ROOT/data/staging/vivid --out $R/cases5 --max-items 5"
    ( cd "$T"
      $G --name base 2>&1 | grep -E '"model"|Error|error' | tail -6
      $G --name l2000 --lora "$OUT/pytorch_lora_weights.safetensors" --lora-scale 1.0 2>&1 | grep -E '"model"|Error|error' | tail -6
      $G --name l1500 --lora "$OUT/checkpoint-1500/pytorch_lora_weights.safetensors" --lora-scale 1.0 2>&1 | grep -E '"model"|Error|error' | tail -6
      true )
    echo "== gate (face / skin / background vs base)"
    ( cd "$T" && "$MP" -m eval.preserve_check "$R/cases5" --base base --candidates l2000 l1500 \
        --out "$R/cases5/preserve.md" --model "$AZ_ROOT/selfie_multiclass_256x256.tflite" ) || echo "GATE_EXIT=$?"
    ( cd "$T" && "$PY" "$W/five_test.py" video "$AZ_ROOT" --lora "${FIVE_LORA:-l2000}" )
    echo "FIVE_DONE: video $AZ_ROOT/eval/five_test/five_test.mp4, key $AZ_ROOT/eval/five_test_KEY/key.json"
    ;;
  video)
    ( cd "$T" && "$PY" "$W/five_test.py" video "$AZ_ROOT" --lora "${FIVE_LORA:-l2000}" )
    ;;
  *) echo "usage: bash five_test.sh prompts|render|video"; exit 2 ;;
esac
true

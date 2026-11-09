#!/usr/bin/env bash
# Step B (owner, 2026-11-09): the two strength blends only, on 10 NEW held-out clips, every panel labelled.
# Two steps because the owner writes the garment descriptions himself (like the live app):
#   AZ_REF=<SHA> bash /workspace/fxb.sh pick     picks 10 unused held-out clips (tops, dresses, bottoms in turn),
#                                               each with another held-out clip's garment of the same type;
#                                               prints every garment photo path; writes an empty
#                                               eval/fxa/set2_descriptions.json to fill in; STOPS (no GPU work)
#   AZ_REF=<SHA> nohup bash /workspace/fxb.sh render > /workspace/fxb.log 2>&1 &
#                                               refuses unless all 10 descriptions are filled in; renders the
#                                               original (needed by the face/skin check only), blend 0.5 @0.8 and
#                                               blend 0.5 @0.65; fixed check; table; one labelled video with
#                                               garment | camera | blend 0.5 @0.8 | blend 0.5 @0.65
# Renders only: any GPU with >= 20 GB. Volume azyah-vton-train at /workspace. Never stops the Pod.
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
MODE="${1:-pick}"
SET="${AZ_SET:-set2}"
export AZ_REF AZ_ROOT=/workspace/azyah-vton
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
C="$AZ_ROOT/checkpoints"; S="$AZ_ROOT/data/staging"; E="$AZ_ROOT/eval/fxa"; R="$E/results/$SET"
PY="$AZ_ROOT/venv-fluxlora/bin/python"; MP="$AZ_ROOT/venv-mp/bin/python"
SEG="$AZ_ROOT/selfie_multiclass_256x256.tflite"
L3="$AZ_ROOT/outputs/fluxrt_lora/fxreal3/pytorch_lora_weights.safetensors"
L4="$AZ_ROOT/outputs/fluxrt_lora/fxreal4/pytorch_lora_weights.safetensors"
START=$(date +%s); mins() { echo "$(( ($(date +%s) - START) / 60 )) min"; }

echo "== code at $AZ_REF"
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || { echo CODE_FAILED; exit 1; }
T="$AZ_ROOT/azyah-code/training"; export PYTHONPATH="$T"

case "$MODE" in
  pick)
    ( cd "$T" && "$PY" -m eval.step_a pick "$AZ_ROOT" --set "$SET" --n "${AZ_N:-10}" ) || { echo "PICK_FAILED"; exit 1; }
    echo "PICK_DONE: fill in $E/${SET}_descriptions.json (all 10), then run: AZ_REF=$AZ_REF nohup bash /workspace/fxb.sh render > /workspace/fxb.log 2>&1 &"
    ;;
  render)
    nvidia-smi --query-gpu=name,memory.total --format=csv
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
    [ "${VRAM:-0}" -ge 20000 ] || { echo "GPU_TOO_SMALL: ${VRAM} MiB; renders need >= 20 GB"; exit 1; }
    for f in "$L3" "$L4" "$SEG" "$E/$SET.json"; do [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }; done
    "$MP" -c "import mediapipe, av" 2>/dev/null || { echo "NO_MEDIAPIPE (venv-mp)"; exit 1; }
    echo "== 1. the owner's descriptions -> prompts"
    ( cd "$T" && "$PY" -m eval.step_a apply "$AZ_ROOT" --set "$SET" ) || { echo "DESCRIPTIONS_INCOMPLETE: fill every item in $E/${SET}_descriptions.json, then rerun"; exit 2; }
    mkdir -p "$R"
    echo "== 2. renders: original (for the check only), blend 0.5 @0.8, blend 0.5 @0.65 ($(mins))"
    G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --ref-size 288x512 --steps 2 --prompt-style compact --clip-list $E/$SET.json --staging $S/vivid --out $R --max-items 10"
    render() {
      local n="$1"; shift
      ( cd "$T" && timeout -s INT 25m $G --name "$n" "$@" ) > "$E/render_${SET}_$n.log" 2>&1
      echo "render $n: $(grep -c '"model"' "$E/render_${SET}_$n.log") clips ($(mins))"; grep -iE "error|traceback" "$E/render_${SET}_$n.log" | tail -3; return 0
    }
    render base
    render blend05s08 --lora "$L3" --lora2 "$L4" --blend 0.5 --lora-scale 0.8
    render blend05s065 --lora "$L3" --lora2 "$L4" --blend 0.5 --lora-scale 0.65
    CANDS="blend05s08 blend05s065"
    echo "== 3. fixed face / skin / background check vs the original ($(mins))"
    ( cd "$T" && "$MP" -m eval.preserve_check "$R" --base base --candidates $CANDS --out "$E/preserve_$SET.md" --model "$SEG" ) || echo "GATE_EXIT=$?"
    ( cd "$T" && "$PY" - "$R" <<'PYEOF'
import json, sys
from pathlib import Path
v = json.loads(Path(sys.argv[1], "preserve_check.json").read_text())["verdict"]
for c, r in v.items():
    fails = [f"{x['clip']}: {', '.join(x['worse_on'])}" for x in r["fails"]]
    print(f"GATE {c}: {'PASS' if r['pass'] else 'FAIL'} ({r['clips']} clips)" + ("" if r["pass"] else f"; worse on {fails}"))
PYEOF
    ) || true
    echo "== 4. steadiness / sharpness / colour ($(mins))"
    ( cd "$T" && "$MP" -m eval.stability_report "$R" --models base $CANDS --model "$SEG" --out "$E/stability_$SET.md" ) || echo "STABILITY_EXIT=$?"
    echo "== 5. labelled video: garment | camera | blend 0.5 @0.8 | blend 0.5 @0.65 ($(mins))"
    ( cd "$T" && "$PY" -m eval.step_a video "$AZ_ROOT" --set "$SET" --cols 4 --no-legend \
        --panels "blend05s08=blend 0.5 @0.8" "blend05s065=blend 0.5 @0.65" ) || echo "VIDEO_EXIT=$?"
    ( cd "$E" && tar cf "$AZ_ROOT/eval/fxb_review.tar" "$SET.json" "${SET}_descriptions.json" "preserve_$SET.md" "stability_$SET.md" render_${SET}_*.log 2>/dev/null )
    echo "FXB_DONE after $(mins): video $E/${SET}_labelled.mp4; reports $AZ_ROOT/eval/fxb_review.tar"
    ;;
  *) echo "usage: bash fxb.sh pick|render"; exit 2 ;;
esac

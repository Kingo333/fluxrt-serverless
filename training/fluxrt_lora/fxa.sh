#!/usr/bin/env bash
# Step A (owner, 2026-11-09): no training. Blends of the fxreal3 and fxreal4 LoRAs (their weight changes
# averaged: 50/50 and 30/70) and a strength sweep (0.8, 0.65) on the 50/50 blend, rendered with the distilled
# klein-4B at 2 steps on 10 clips: the owner's 5 (same prompts as before) + 5 held-out ViViD clips trying on
# another held-out clip's garment. Then the FIXED face/skin/background check (skin compared only where it is
# skin in the camera AND every result), the steadiness/sharpness/colour table, and ONE labelled video
# (every panel named on screen; not blind). Renders only: any GPU with >= 20 GB works (RTX 4090 / PRO 4000
# are fine and cheaper). Volume azyah-vton-train at /workspace. Approved cost ~$1-2. Never stops the Pod.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fxa.sh -o /workspace/fxa.sh
#   AZ_REF=<SHA> nohup bash /workspace/fxa.sh > /workspace/fxa.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
export AZ_REF AZ_ROOT=/workspace/azyah-vton
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
C="$AZ_ROOT/checkpoints"; S="$AZ_ROOT/data/staging"; E="$AZ_ROOT/eval/fxa"; R="$E/results/clips10"
PY="$AZ_ROOT/venv-fluxlora/bin/python"; MP="$AZ_ROOT/venv-mp/bin/python"
SEG="$AZ_ROOT/selfie_multiclass_256x256.tflite"
L3="$AZ_ROOT/outputs/fluxrt_lora/fxreal3/pytorch_lora_weights.safetensors"
L4="$AZ_ROOT/outputs/fluxrt_lora/fxreal4/pytorch_lora_weights.safetensors"
mkdir -p "$R"
START=$(date +%s); mins() { echo "$(( ($(date +%s) - START) / 60 )) min"; }

echo "== 0. GPU, files"
nvidia-smi --query-gpu=name,memory.total --format=csv
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
[ "${VRAM:-0}" -ge 20000 ] || { echo "GPU_TOO_SMALL: ${VRAM} MiB; renders need >= 20 GB"; exit 1; }
for f in "$L3" "$L4" "$SEG" "$AZ_ROOT/eval/fx2/cases5/clip_list.json"; do [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }; done
[ -d "$C/FLUX.2-klein-4B" ] || { echo "MISSING: $C/FLUX.2-klein-4B"; exit 1; }

echo "== 1. code at $AZ_REF"
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || { echo CODE_FAILED; exit 1; }
T="$AZ_ROOT/azyah-code/training"; export PYTHONPATH="$T"
"$MP" -c "import mediapipe, av" 2>/dev/null || { echo "NO_MEDIAPIPE (venv-mp)"; exit 1; }

echo "== 2. the 10 clips and their prompts ($(mins))"
( cd "$T" && "$PY" -m eval.step_a clips "$AZ_ROOT" ) || { echo "CLIPS_FAILED"; exit 1; }
echo "== 2b. reuse the renders that already exist for the owner's 5 clips"
( cd "$T" && "$PY" -m eval.step_a reuse "$AZ_ROOT" ) || echo "REUSE_EXIT=$?"

echo "== 3. renders (missing ones only), distilled klein-4B, 2 steps, 288x512 ($(mins))"
G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --ref-size 288x512 --steps 2 --prompt-style compact --clip-list $E/clips10.json --staging $S/vivid --out $R --max-items 10"
render() {  # name [lora args...]
  local n="$1"; shift
  ( cd "$T" && timeout -s INT 25m $G --name "$n" "$@" ) > "$E/render_$n.log" 2>&1
  echo "render $n: $(grep -c '"model"' "$E/render_$n.log") new clips ($(mins))"; grep -iE "error|traceback" "$E/render_$n.log" | tail -3; return 0
}
render base
render fx3 --lora "$L3"
render fx4 --lora "$L4"
render blend05 --lora "$L3" --lora2 "$L4" --blend 0.5
render blend03 --lora "$L3" --lora2 "$L4" --blend 0.3
render blend05s08 --lora "$L3" --lora2 "$L4" --blend 0.5 --lora-scale 0.8
render blend05s065 --lora "$L3" --lora2 "$L4" --blend 0.5 --lora-scale 0.65
CANDS="fx3 fx4 blend05 blend03 blend05s08 blend05s065"

echo "== 4. FIXED face / skin / background check (skin only where it stays skin) ($(mins))"
( cd "$T" && "$MP" -m eval.preserve_check "$R" --base base --candidates $CANDS --out "$E/preserve.md" --model "$SEG" ) || echo "GATE_EXIT=$?"
( cd "$T" && "$PY" - "$R" <<'PYEOF'
import json, sys
from pathlib import Path
v = json.loads(Path(sys.argv[1], "preserve_check.json").read_text())["verdict"]
for c, r in v.items():
    fails = [f"{x['clip']}: {', '.join(x['worse_on'])}" for x in r["fails"]]
    print(f"GATE {c}: {'PASS' if r['pass'] else 'FAIL'} ({r['clips']} clips)" + ("" if r["pass"] else f"; worse on {fails}"))
PYEOF
) || true

echo "== 5. steadiness / sharpness / colour per model ($(mins))"
( cd "$T" && "$MP" -m eval.stability_report "$R" --models base $CANDS --model "$SEG" --out "$E/stability.md" ) || echo "STABILITY_EXIT=$?"

echo "== 6. one labelled video, all 10 clips, every panel named ($(mins))"
( cd "$T" && "$PY" -m eval.step_a video "$AZ_ROOT" ) || echo "VIDEO_EXIT=$?"
( cd "$E" && tar cf "$AZ_ROOT/eval/fxa_review.tar" clips10.json preserve.md stability.md stability.json render_*.log 2>/dev/null )
echo "FXA_DONE after $(mins): video $E/step_a_labelled.mp4; reports $AZ_ROOT/eval/fxa_review.tar"

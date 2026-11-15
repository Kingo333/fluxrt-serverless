#!/usr/bin/env bash
# fxreal6, step 3 (owner, 2026-11-15): favourite #2's own numbers on the test sets. Trains nothing.
# This sets the pass bar the new model has to beat: face match, skin tone, lightness, texture, logo leak,
# steadiness and the five garment priorities, all with the app's live prompt and the live garment photo
# size (528x704).
#   1. picks 24 new held-out clips (8 tops, 8 dresses, 8 bottoms): eval/fxa/fx6test.json
#   2. builds a small logo set (information only, no pass bar): the 4 plainest garments, each clip as filmed and with a logo stamped on the
#      person's own garment: eval/fxa/fx6logo.json
#   3. renders favourite #2 (blend05s065) on fx6test, fx6logo, dress10, set2, cases5 and hijab
#   4. reports: FACESKIN lines, LOGO_LEAK, steadiness, the garment priorities
# Needs a GPU with >= 30 GB (RTX 5090 or RTX PRO 6000) and the azyah-vton-train volume at /workspace.
# Never touches an endpoint or the azyah-joyai volume. About 45-60 minutes: about $1 on a 5090, $2.50 on a PRO 6000.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fx6_baseline.sh -o /workspace/fx6_baseline.sh
#   AZ_REF=<SHA> AZ_NO_STOP=1 nohup bash /workspace/fx6_baseline.sh > /workspace/fx6_baseline.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
export AZ_REF AZ_ROOT=/workspace/azyah-vton
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
REF_SIZE="${AZ_REF_SIZE:-528x704}"
TAG="r${REF_SIZE%%x*}"
FAV="fav2_$TAG"
GUARD=/workspace/fx6_baseline_started
stop_pod() {
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: Pod left RUNNING (billing) - stop it by hand"; return; fi
  echo "== stopping the Pod"; runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
}
trap stop_pod EXIT
[ -e "$GUARD" ] && [ "${AZ_FORCE:-0}" != "1" ] && { echo "GUARD: $GUARD exists (already ran); AZ_FORCE=1 to run again"; exit 0; }
touch "$GUARD"
for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo "waiting for network"; sleep 10; done
C="$AZ_ROOT/checkpoints"; S="$AZ_ROOT/data/staging"; E="$AZ_ROOT/eval/fxa"; R="$E/results"
PY="$AZ_ROOT/venv-fluxlora/bin/python"; MP="$AZ_ROOT/venv-mp/bin/python"
SEG="$AZ_ROOT/selfie_multiclass_256x256.tflite"
INIT="$AZ_ROOT/outputs/fluxrt_lora/blend05s065/pytorch_lora_weights.safetensors"
mkdir -p "$E" "$R" "$AZ_ROOT/reports"
START=$(date +%s)
mins() { echo "$(( ($(date +%s) - START) / 60 )) min"; }
echo "== fxreal6 baseline: favourite #2, garment photo $REF_SIZE, live prompt"

echo "== 0. GPU, code, tools"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo NO_GPU; exit 1; }
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || { echo CODE_FAILED; exit 1; }
T="$AZ_ROOT/azyah-code/training"
export PYTHONPATH="$T"
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
[ -x "$PY" ] && "$PY" -c "import torch, diffusers, av" 2>/dev/null || { echo "NO_RENDER_VENV: $PY (fx5 made it; rerun its setup)"; exit 1; }
[ -x "$MP" ] && "$MP" -c "import mediapipe, av, numpy" 2>/dev/null || { echo "NO_MEDIAPIPE_VENV: $MP"; exit 1; }
[ -f "$INIT" ] || { echo "NO_FAVOURITE2: $INIT"; exit 1; }
sha256sum "$INIT"
[ -d "$C/FLUX.2-klein-4B" ] || { echo "NO_MODEL: $C/FLUX.2-klein-4B"; exit 1; }

echo "== 1. test sets ($(mins))"
[ -f "$E/fx6test.json" ] || ( cd "$T" && "$PY" -m eval.step_a pick "$AZ_ROOT" --set fx6test --n 24 --auto-describe ) || echo "PICK_EXIT=$?"
[ -f "$E/fx6test.json" ] || { echo NO_TEST_SET; exit 1; }
"$PY" -c "import json,collections;d=json.load(open('$E/fx6test.json'));print('FX6TEST', len(d), dict(collections.Counter(i['category'] for i in d)))"
[ -f "$E/fx6logo.json" ] || ( cd "$T" && "$MP" -m eval.logo_set "$AZ_ROOT" --from eval/fxa/fx6test.json --out eval/fxa/fx6logo.json --n 4 --model "$SEG" ) \
  || echo "LOGO_SET_EXIT=$?"
SETS=""
for s in fx6test fx6logo dress10 set2 cases5 hijab; do [ -f "$E/$s.json" ] && SETS="$SETS $s"; done
echo "sets:$SETS"

echo "== 2. renders: favourite #2 at garment size $REF_SIZE ($(mins))"
G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --steps 2 --prompt-style live"
for s in $SETS; do
  ( cd "$T" && timeout -s INT 40m $G --ref-size "$REF_SIZE" --lora "$INIT" --lora-scale 1.0 --name "$FAV" \
      --clip-list "$E/$s.json" --staging "$S/vivid" --out "$R/$s" --max-items 30 ) > "$E/render_fx6base_$s.log" 2>&1
  echo "render $s: $(grep -c '"model"' "$E/render_fx6base_$s.log") clips ($(mins))"
  grep -iE "error|traceback" "$E/render_fx6base_$s.log" | tail -3
done
grep -h '"fps"' "$E/render_fx6base_fx6test.log" 2>/dev/null | tail -1 | cut -c1-300

echo "== 3. face and skin, logo leak ($(mins))"
for s in $SETS; do
  LEAK=""; [ "$s" = "fx6logo" ] && LEAK="--leak-pairs"
  ( cd "$T" && "$MP" -m eval.face_skin_report "$R/$s" --models "$FAV" --model "$SEG" --out "$E/faceskin_fx6base_$s.md" $LEAK ) \
    2>&1 | grep -E "^(FACESKIN|LOGO_LEAK)" || echo "FACESKIN_${s}_EXIT=$?"
done

echo "== 4. steadiness and the garment priorities ($(mins))"
for s in $SETS; do
  ( cd "$T" && "$MP" -m eval.stability_report "$R/$s" --models "$FAV" --model "$SEG" --out "$E/stability_fx6base_$s.md" ) \
    > "$E/stability_fx6base_$s.log" 2>&1 || echo "STABILITY_${s}_EXIT=$?"
  ( cd "$T" && "$MP" -m eval.garment_report seg "$R/$s" --models "$FAV" --model "$SEG" ) > "$E/garment_seg_fx6base_$s.log" 2>&1 \
    || echo "GARMENT_SEG_${s}_EXIT=$?"
  echo "-- $s"
  ( cd "$T" && "$PY" -m eval.garment_report score "$R/$s" --models "$FAV" --baseline "$FAV" --out "$E/garment_fx6base_$s.md" \
      --json "$R/$s/garment_report_fx6base.json" ) > "$E/garment_score_fx6base_$s.log" 2>&1 || echo "GARMENT_SCORE_${s}_EXIT=$?"
  # one model only, so no "better / worse" lines: print favourite #2's own averages instead
  "$PY" - "$R/$s/garment_report_fx6base.json" "$FAV" "$s" <<'PYEOF' || true
import json, sys
from pathlib import Path
f = Path(sys.argv[1])
if f.exists():
    items = json.loads(f.read_text()).get("items", {})
    keys = sorted({k for it in items.values() for k, v in it.get(sys.argv[2], {}).items() if isinstance(v, (int, float))})
    means = {k: [it[sys.argv[2]][k] for it in items.values() if isinstance(it.get(sys.argv[2], {}).get(k), (int, float))] for k in keys}
    print(f"GARMENT {sys.argv[3]} {sys.argv[2]}: " + "; ".join(f"{k} {sum(v) / len(v):.3f}" for k, v in means.items() if v) + f" ({len(items)} clips)")
PYEOF
  grep -hE "wobble|sharpness|colour" "$E/stability_fx6base_$s.md" 2>/dev/null | tail -3 | cut -c1-200
done
( cd "$E" && tar cf "$AZ_ROOT/eval/fx6_baseline.tar" fx6test.json fx6logo.json faceskin_fx6base_*.md stability_fx6base_*.md \
    garment_fx6base_*.md results/*/face_skin.json render_fx6base_*.log 2>/dev/null )
echo "FX6_BASELINE_DONE after $(mins): reports $AZ_ROOT/eval/fx6_baseline.tar"

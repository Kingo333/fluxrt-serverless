#!/usr/bin/env bash
# fxreal6, steps 6 to 8 (owner, 2026-11-15): trial, training, evaluation. One stage per run:
#
#   AZ_STAGE=trial  200 steps from favourite #2 on the fxreal6 data. Prints the measured seconds per step and
#                   what 2, 3 and 4 passes over the data would take and cost. Its steps are kept (the training
#                   stage carries on from them).
#   AZ_STAGE=train  AZ_MAX_STEPS=<n> (from the trial). Carries on from the newest save; a save every 500 steps.
#                   Stops itself after AZ_MAX_MINUTES (default 300) and can be started again.
#   AZ_STAGE=eval   Renders favourite #2 (kept from the baseline), fxreal6, an earlier fxreal6 save and the
#                   75% blend back towards favourite #2 on every test set; then the owner's five priorities:
#                   face and skin (FACESKIN, FACESKIN_GATE), garment and colour (PRIORITY), steadiness,
#                   head covering (HIJAB), and one labelled video per set.
#
# The recipe is fxreal5's trainer, unchanged: FLUX.2-klein-base-4B, FluxRT's noise levels, rank 64 from
# favourite #2, LR 5e-5, batch 4, weight mask x3 (garment and face full, skin and hair half), pair term 1.0,
# edges 0.05, colour 0.5, texture 0.3, garment photo 528x704 (3 batches in 4) and 384x512.
# Needs an RTX PRO 6000 and the azyah-vton-train volume at /workspace. Never touches an endpoint.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fx6_train.sh -o /workspace/fx6_train.sh
#   AZ_REF=<SHA> AZ_STAGE=trial AZ_NO_STOP=1 nohup bash /workspace/fx6_train.sh > /workspace/fx6_trial.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
STAGE="${AZ_STAGE:?set AZ_STAGE to trial, train or eval}"
export AZ_REF AZ_ROOT=/workspace/azyah-vton
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
EXP="${AZ_EXP:-fxreal6}"
REF_SIZE="${AZ_REF_SIZE:-528x704}"
REF_SIZES="${AZ_REF_SIZES:-528x704:3,384x512:1}"
PRICE="${AZ_PRICE:-2.70}"          # dollars per hour of this Pod, for the cost lines only
MAX_MIN="${AZ_MAX_MINUTES:-300}"
TAG="r${REF_SIZE%%x*}"
stop_pod() {
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: Pod left RUNNING (billing) - stop it by hand"; return; fi
  echo "== stopping the Pod"; runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
}
trap stop_pod EXIT
for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo "waiting for network"; sleep 10; done
C="$AZ_ROOT/checkpoints"; D="$AZ_ROOT/data"; S="$D/staging"; E="$AZ_ROOT/eval/fxa"; R="$E/results"
PY="$AZ_ROOT/venv-fluxlora/bin/python"; MP="$AZ_ROOT/venv-mp/bin/python"
SEG="$AZ_ROOT/selfie_multiclass_256x256.tflite"
INIT="$AZ_ROOT/outputs/fluxrt_lora/blend05s065"
DATA="$D/fluxrt_lora/$EXP"; OUT="$AZ_ROOT/outputs/fluxrt_lora/$EXP"
mkdir -p "$E" "$R" "$AZ_ROOT/reports"
START=$(date +%s)
mins() { echo "$(( ($(date +%s) - START) / 60 )) min"; }
echo "== fxreal6 $STAGE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo NO_GPU; exit 1; }
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || { echo CODE_FAILED; exit 1; }
T="$AZ_ROOT/azyah-code/training"
export PYTHONPATH="$T"
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
[ -f "$INIT/pytorch_lora_weights.safetensors" ] || { echo "NO_FAVOURITE2: $INIT"; exit 1; }
# The MediaPipe venv: rebuilt when it does not work on this image. It gets its own Python 3.11 through uv,
# kept on the volume, so it no longer depends on the Pod template's Python.
mp_ok() { [ -x "$MP" ] && "$MP" -c "import mediapipe, av, numpy, pyarrow, PIL" 2>/dev/null; }
if ! mp_ok; then
  echo "rebuilding the MediaPipe venv ($(mins))"
  [ -e "$AZ_ROOT/venv-mp" ] && mv "$AZ_ROOT/venv-mp" "$AZ_ROOT/venv-mp-stale-$(date -u +%Y%m%dT%H%M%SZ)"
  export UV_PYTHON_INSTALL_DIR="$AZ_ROOT/uv-python"
  python3 -m pip install -q uv 2>/dev/null || pip install -q uv 2>/dev/null || true
  if python3 -m uv venv --python 3.11 "$AZ_ROOT/venv-mp" >/dev/null 2>&1; then
    python3 -m uv pip install -q --python "$MP" mediapipe==0.10.21 av pillow "numpy<2" "pyarrow==17.0.0" || true
  else
    python3 -m venv "$AZ_ROOT/venv-mp" && "$AZ_ROOT/venv-mp/bin/pip" install -q mediapipe==0.10.21 av pillow "numpy<2" "pyarrow==17.0.0" || true
  fi
  # MediaPipe's OpenCV needs these two system libraries; a bare image may lack them.
  mp_ok || { apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null 2>&1; }
fi
mp_ok || { echo "NO_MEDIAPIPE_VENV: $MP"; "$MP" -c "import mediapipe, av, numpy, pyarrow, PIL" 2>&1 | tail -3; exit 1; }
[ -f "$DATA/dataset_summary.json" ] && [ -f "$DATA/region_weight.json" ] || { echo "NO_DATASET: run fx6_data.sh first ($DATA)"; exit 1; }
ROWS=$("$PY" -c "import json; print(json.load(open('$DATA/region_weight.json')).get('rows_written', 0))" 2>/dev/null || echo 0)
echo "rows in the dataset: $ROWS"

train() {  # steps, save interval, minutes
  timeout -s INT "${3}m" bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/fluxrt_lora/run_train.sh | \
    AZ_REF=$AZ_REF AZ_MANIFEST_DIR=$AZ_ROOT/manifests/research-train AZ_EXP=$EXP \
    KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B AZ_TIME_SHIFT_MU=2.01 AZ_REF_SIZE=$REF_SIZE AZ_REF_SIZES=$REF_SIZES \
    AZ_MAX_STEPS=$1 AZ_BATCH=4 AZ_LORA_RANK=64 AZ_LR=${AZ_LR:-5e-5} \
    AZ_CKPT_INTERVAL=$2 AZ_CKPT_LIMIT=12 AZ_MASK_COLUMN=mask AZ_MASK_WEIGHT=${AZ_MASK_WEIGHT:-3.0} \
    AZ_PAIR_COLUMN=pair AZ_PAIR_WEIGHT=${AZ_PAIR_WEIGHT:-1.0} AZ_DETAIL_WEIGHT=${AZ_DETAIL_WEIGHT:-0.05} \
    AZ_COLOUR_WEIGHT=${AZ_COLOUR_WEIGHT:-0.5} AZ_TEXTURE_WEIGHT=${AZ_TEXTURE_WEIGHT:-0.3} \
    AZ_ANCHOR_COLUMN=${AZ_ANCHOR_COLUMN:-anchor} AZ_ANCHOR_PROB=${AZ_ANCHOR_PROB:-1.0} \
    AZ_INIT_LORA=$INIT bash" > "$AZ_ROOT/reports/fx6_${STAGE}_run.log" 2>&1
  echo "TRAIN_EXIT=$?"
  LOG=$(ls -t "$AZ_ROOT"/reports/fluxlora_${EXP}_*.log 2>/dev/null | head -1)
  grep -h "LoRA initialised from" "$LOG" 2>/dev/null | tail -1
  grep -h "^REF_SIZES" "$LOG" 2>/dev/null | tail -1
  grep -h "^ANCHOR_BATCHES" "$LOG" 2>/dev/null | tail -1
  grep -hE "^(CACHE_RAM|CACHE_CPU|NOT_ENOUGH_CPU_RAM)|OutOfMemoryError|CUDA out of memory|Traceback" "$LOG" "$AZ_ROOT/reports/fx6_${STAGE}_run.log" 2>/dev/null | tail -6
  grep -oE "(loss|detail|pair|colour|texture)=[0-9.e+-]+" "$LOG" 2>/dev/null | tail -10 | tr '\n' ' '; echo
  ls "$OUT" 2>/dev/null | tr '\n' ' '; echo
}

case "$STAGE" in
trial)
  echo "== trial: 200 steps ($(mins))"
  train 200 100 60
  LOG=$(ls -t "$AZ_ROOT"/reports/fluxlora_${EXP}_*.log 2>/dev/null | head -1)
  "$PY" - "$LOG" "$ROWS" "$PRICE" <<'PYEOF' || echo "TRIAL_PARSE_EXIT=$?"
import re, statistics, sys
text = open(sys.argv[1], errors="ignore").read() if sys.argv[1] else ""
rows, price = int(sys.argv[2] or 0), float(sys.argv[3])
# Only the training bar ("Steps: ..."). The "Caching latents" bar runs about 5 batches a second and must never be
# read as the training speed (2026-11-15: it was, the time limit came out about 6x too short).
segs = [x for x in re.split(r"[\r\n]", text) if x.lstrip().startswith("Steps:")]
secs = [float(x) for seg in segs for x in re.findall(r"([0-9.]+)s/it", seg)]
secs += [1 / float(x) for seg in segs for x in re.findall(r"([0-9.]+)it/s", seg) if float(x) > 0]
secs = secs[len(secs) // 3:]  # the first steps include start-up
if not secs or not rows:
    print("TRIAL no step times found in the log (or no row count): send the last 40 lines")
else:
    s = statistics.median(secs)
    print(f"TRIAL {s:.2f} s per step (batch 4), {rows} rows")
    for passes in (2, 3, 4):
        steps = round(passes * rows / 4 / 100) * 100
        hours = steps * s / 3600
        print(f"TRIAL {passes} passes = {steps} steps = {hours:.1f} h = ${hours * price:.0f}")
PYEOF
  echo "FX6_TRIAL_DONE after $(mins)"
  ;;
train)
  STEPS="${AZ_MAX_STEPS:?set AZ_MAX_STEPS (from the trial lines)}"
  echo "== training to step $STEPS, a save every 500, stops itself after $MAX_MIN minutes ($(mins))"
  train "$STEPS" 500 "$MAX_MIN"
  cp -f "$DATA/selection.json" "$OUT/selection.json" 2>/dev/null || true
  if [ -f "$OUT/pytorch_lora_weights.safetensors" ] && ls "$OUT" | grep -q "checkpoint-$STEPS\|pytorch_lora_weights"; then
    sha256sum "$OUT/pytorch_lora_weights.safetensors"
  fi
  LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  echo "newest save: step ${LAST:-none} of $STEPS"
  [ "${LAST:-0}" -ge "$STEPS" ] && echo "FX6_TRAIN_DONE after $(mins)" || echo "FX6_TRAIN_NOT_FINISHED after $(mins): run the same line again to carry on"
  ;;
eval)
  FX="$OUT/pytorch_lora_weights.safetensors"
  # A training run stopped by its time limit leaves no final file, or an older one: use the newest save then.
  LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  LASTF="$OUT/checkpoint-$LAST/pytorch_lora_weights.safetensors"
  if [ -n "$LAST" ] && [ -f "$LASTF" ] && { [ ! -f "$FX" ] || [ "$LASTF" -nt "$FX" ]; }; then
    cp "$LASTF" "$FX" && echo "using the newest save, step $LAST"
  else
    echo "using the final file of the training (newest save: step ${LAST:-none})"
  fi
  [ -f "$FX" ] || { echo "NO_FX6_LORA: $FX"; exit 1; }
  cp -f "$DATA/selection.json" "$OUT/selection.json" 2>/dev/null || true
  sha256sum "$FX"
  # The renders are kept per model NAME and an existing render is never made again. So the name must change when
  # the weights change, or an evaluation silently scores the videos of an older save (2026-11-16: the step-5800
  # weights were "evaluated" on the step-1000 videos, every render line said "0 new clips"). The name carries a
  # label: AZ_EVAL_LABEL (e.g. s5800), or by default the first 8 characters of the weights' checksum. A label that
  # was already used with different weights is refused: nothing is deleted or overwritten.
  SUM=$(sha256sum "$FX" | cut -c1-64)
  LABEL=$(printf '%s' "${AZ_EVAL_LABEL:-${SUM:0:8}}" | tr -cd 'A-Za-z0-9')
  [ -n "$LABEL" ] || { echo "BAD_AZ_EVAL_LABEL"; exit 1; }
  FAV="fav2_$TAG"; NEW="fx6${LABEL}_$TAG"; MIX="fx6b75${LABEL}_$TAG"; ANC="fx6anchor${LABEL}_$TAG"
  MARK="$R/.weights_fx6${LABEL}_$TAG"
  mkdir -p "$R"
  if [ -f "$MARK" ] && [ "$(cat "$MARK")" != "$SUM" ]; then
    echo "STALE_EVAL_LABEL: the label $LABEL was used for other weights ($(cut -c1-8 "$MARK")...), these are ${SUM:0:8}...; choose another AZ_EVAL_LABEL"
    exit 1
  fi
  echo "$SUM" > "$MARK"
  echo "evaluating weights ${SUM:0:8} as $NEW and $ANC"
  # Sets and models are trimmed to keep the scoring inside the budget (the baseline's scoring alone took 66 min for
  # one model): AZ_EVAL_SETS (default the 24 new test clips, the owner's 5 cases, the hijab set, the 10 dresses);
  # AZ_EVAL_BLEND=1 adds the 75% blend back towards favourite #2.
  SETS=""
  for s in ${AZ_EVAL_SETS:-fx6test cases5 hijab dress10}; do [ -f "$E/$s.json" ] && SETS="$SETS $s"; done
  CANDS="$NEW $ANC"; [ "${AZ_EVAL_BLEND:-0}" = "1" ] && CANDS="$CANDS $MIX"
  echo "== renders at garment size $REF_SIZE on:$SETS; models: favourite #2 (kept from the baseline), $CANDS ($(mins))"
  G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --steps 2 --prompt-style live --ref-size $REF_SIZE"
  render() {  # set, name, extra args...
    local s="$1" n="$2"; shift 2
    ( cd "$T" && timeout -s INT 40m $G "$@" --name "$n" --clip-list "$E/$s.json" --staging "$S/vivid" --out "$R/$s" --max-items 30 ) \
      > "$E/render_fx6_${s}_$n.log" 2>&1
    local have; have=$(ls "$R/$s"/*/out_"$n".mp4 2>/dev/null | wc -l)
    echo "render $s/$n: $(grep -c '"model"' "$E/render_fx6_${s}_$n.log") new clips, $have in total ($(mins))"
    grep -iE "error|traceback" "$E/render_fx6_${s}_$n.log" | tail -3
    [ "$have" -gt 0 ] || { echo "NO_RENDERS: $s/$n has no videos; the scores would mean nothing"; RENDER_MISSING=1; }
    return 0
  }
  RENDER_MISSING=0
  for s in $SETS; do
    render "$s" "$FAV" --lora "$INIT/pytorch_lora_weights.safetensors"      # kept from the baseline: nothing new to render
    render "$s" "$NEW" --lora "$FX"
    # with its own earlier picture as the third image, as live with FLUXRT_ANCHOR=1 (taken after 8 frames, renewed every 12)
    render "$s" "$ANC" --lora "$FX" --anchor-after 8 --anchor-every 12
    [ "${AZ_EVAL_BLEND:-0}" = "1" ] && render "$s" "$MIX" --lora "$FX" --lora2 "$INIT/pytorch_lora_weights.safetensors" --blend 0.75
  done
  [ "$RENDER_MISSING" = "0" ] || { echo "FX6_EVAL_NO_RENDERS: stopped before scoring"; exit 1; }
  # Scoring runs on the CPU, one process per set at the same time (the Pod has many cores; the GPU is free now).
  echo "== scoring: face and skin, background and steadiness, head covering, garment and colour; sets in parallel ($(mins))"
  score_set() {
    local s="$1" leak=""
    [ "$s" = "fx6logo" ] && leak="--leak-pairs"
    ( cd "$T" && "$MP" -m eval.face_skin_report "$R/$s" --models "$FAV" $CANDS --baseline "$FAV" --model "$SEG" \
        --out "$E/faceskin_fx6_$s.md" $leak ) > "$E/faceskin_fx6_$s.log" 2>&1 || echo "FACESKIN_${s}_EXIT=$?" >> "$E/faceskin_fx6_$s.log"
    ( cd "$T" && "$MP" -m eval.preserve_check "$R/$s" --base "$FAV" --candidates $CANDS --out "$E/preserve_fx6_$s.md" --model "$SEG" ) \
      > "$E/preserve_fx6_$s.log" 2>&1 || echo "GATE_${s}_EXIT=$?" >> "$E/preserve_fx6_$s.log"
    ( cd "$T" && "$MP" -m eval.stability_report "$R/$s" --models "$FAV" $CANDS --model "$SEG" --out "$E/stability_fx6_$s.md" ) \
      > "$E/stability_fx6_$s.log" 2>&1 || echo "STABILITY_${s}_EXIT=$?" >> "$E/stability_fx6_$s.log"
    ( cd "$T" && "$MP" -m eval.garment_report seg "$R/$s" --models "$FAV" $CANDS --model "$SEG" ) > "$E/garment_seg_fx6_$s.log" 2>&1 \
      || echo "GARMENT_SEG_${s}_EXIT=$?" >> "$E/garment_seg_fx6_$s.log"
    ( cd "$T" && "$PY" -m eval.garment_report score "$R/$s" --models "$FAV" $CANDS --baseline "$FAV" --out "$E/garment_fx6_$s.md" \
        --json "$R/$s/garment_report_fx6.json" ) > "$E/garment_score_fx6_$s.log" 2>&1 || echo "GARMENT_SCORE_${s}_EXIT=$?" >> "$E/garment_score_fx6_$s.log"
  }
  for s in $SETS; do score_set "$s" & done
  wait
  echo "== results ($(mins))"
  for s in $SETS; do
    echo "-- $s"
    grep -hE "^(FACESKIN|LOGO_LEAK)|_EXIT=" "$E/faceskin_fx6_$s.log" | cut -c1-600
    grep -hE "^PRIORITY|DINO|_EXIT=" "$E/garment_score_fx6_$s.log" "$E/garment_seg_fx6_$s.log" | cut -c1-400
    grep -hE "_EXIT=" "$E/preserve_fx6_$s.log" "$E/stability_fx6_$s.log"
    grep -hE "wobble|sharpness|colour" "$E/stability_fx6_$s.md" 2>/dev/null | head -6 | cut -c1-220 | sed "s/^/STEADY $s /"
  done
  ( cd "$T" && "$PY" - "$R" "$FAV" $SETS <<'PYEOF'
import json, sys
from pathlib import Path
root, base = sys.argv[1], sys.argv[2]
res = {}
for s in sys.argv[3:]:
    f = Path(root, s, "preserve_check.json")
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    for c, v in d["verdict"].items():
        r = res.setdefault(c, {"pass": True, "fails": [], "clips": 0})
        r["pass"] = r["pass"] and v["pass"]
        r["clips"] += v["clips"]
        r["fails"] += [f"{s}/{x['clip']}: {', '.join(x['worse_on'])}" for x in v["fails"]]
    if s == "hijab":
        pct = lambda v: "-" if v is None else f"{v * 100:.0f}%"  # noqa: E731
        for clip, sc in d["per_clip"].items():
            for m, v in sc.items():
                if m != base:
                    print(f"HIJAB {clip} {m}: head covering turned into hair/face/skin {pct(v.get('head_lost'))} "
                          f"(favourite #2 {pct(sc.get(base, {}).get('head_lost'))})")
for c, r in res.items():
    print(f"GATE {c} vs favourite #2: {'PASS' if r['pass'] else 'FAIL'} ({r['clips']} clips)" + ("" if r["pass"] else f"; worse on {r['fails']}"))
PYEOF
  ) || true
  echo "== labelled videos: garment | camera | favourite #2 (today) | fxreal6 | fxreal6 + earlier frame ($(mins))"
  for s in $SETS; do
    PANELS=("$FAV=favourite #2 (today)" "$NEW=fxreal6" "$ANC=fxreal6 + earlier frame")
    [ "${AZ_EVAL_BLEND:-0}" = "1" ] && PANELS+=("$MIX=fxreal6 at 75%")
    ( cd "$T" && "$PY" -m eval.step_a video "$AZ_ROOT" --set "$s" --cols $(( ${#PANELS[@]} + 2 )) --no-legend \
        --panels "${PANELS[@]}" ) > "$E/video_fx6_$s.log" 2>&1 || echo "VIDEO_${s}_EXIT=$?"
    [ -f "$E/${s}_labelled.mp4" ] && mv -f "$E/${s}_labelled.mp4" "$E/fx6_${s}_${TAG}.mp4"
  done
  ( cd "$E" && tar cf "$AZ_ROOT/eval/fx6_review.tar" faceskin_fx6_*.md preserve_fx6_*.md stability_fx6_*.md garment_fx6_*.md \
      results/*/face_skin.json results/*/garment_report_fx6.json render_fx6_*.log fx6_*_"$TAG".mp4 2>/dev/null )
  echo "FX6_EVAL_DONE after $(mins): videos $(ls "$E"/fx6_*_"$TAG".mp4 2>/dev/null | tr '\n' ' ')"
  echo "LoRA $FX (+ selection.json); reports and videos $AZ_ROOT/eval/fx6_review.tar"
  ;;
*) echo "BAD_AZ_STAGE: $STAGE (trial, train or eval)"; exit 1 ;;
esac

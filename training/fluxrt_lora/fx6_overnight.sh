#!/usr/bin/env bash
# fxreal6 overnight (owner, 2026-11-15): after the data build, the 200-step trial, a self-check, the full training
# sized to stay under the cost cap, the evaluation, then the Pod stops itself.
#
#   1. waits for the running data build (fx6_data.sh) to print FX6_DATA_DONE; checks the earlier-frame column is there
#   2. trial: 200 steps (fx6_train.sh AZ_STAGE=trial); stops everything on NaN loss, out of memory, a Traceback,
#      or no step time
#   3. sizes the training: 3 passes over the data, fewer if the money would run out. Running total = AZ_SPENT
#      (dollars spent before this script started) + this Pod's time since then x AZ_PRICE. The training gets what
#      is left of AZ_CAP after reserving the evaluation (AZ_EVAL_HOURS) and a margin (AZ_MARGIN). Under 800 steps:
#      stop, not worth it.
#   4. training (AZ_STAGE=train), with a time limit from the same sum, so it can never pass the cap; resumes saves
#   5. evaluation (AZ_STAGE=eval), with its own time limit; a training stopped by its time limit is judged at its
#      newest save
#   6. the Pod stops itself (unless AZ_NO_STOP=1), whatever happened
# Everything is logged with OVERNIGHT lines and a running cost; never touches an endpoint. Deletes nothing.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fx6_overnight.sh -o /workspace/fx6_overnight.sh
#   AZ_REF=<SHA> AZ_SPENT=13 nohup bash /workspace/fx6_overnight.sh > /workspace/fx6_overnight.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
SPENT0="${AZ_SPENT:?set AZ_SPENT to the dollars already spent (e.g. 13)}"
PRICE="${AZ_PRICE:-2.70}"
CAP="${AZ_CAP:-45}"
EVAL_HOURS="${AZ_EVAL_HOURS:-1.5}"
MARGIN="${AZ_MARGIN:-2}"
PASSES="${AZ_PASSES:-3}"
MIN_STEPS="${AZ_MIN_STEPS:-800}"
DATA_LOG="${AZ_DATA_LOG:-${AZ_WS:-/workspace}/fx6_data.log}"
WS="${AZ_WS:-/workspace}"
ROOT_DIR="${AZ_ROOT_DIR:-$WS/azyah-vton}"
REPORTS="$ROOT_DIR/reports"
OUT="$ROOT_DIR/outputs/fluxrt_lora/fxreal6"
START=$(date +%s)

hours() { awk -v s="$START" -v n="$(date +%s)" 'BEGIN { printf "%.2f", (n - s) / 3600 }'; }
spent_num() { awk -v a="$SPENT0" -v h="$(hours)" -v p="$PRICE" 'BEGIN { printf "%.2f", a + h * p }'; }
spent() { echo "\$$(spent_num)"; }
left_dollars() { awk -v c="$CAP" -v s="$(spent_num)" 'BEGIN { printf "%.2f", c - s }'; }
say() { echo "OVERNIGHT $(date -u +%H:%M) [$(spent)] $*"; }
fail() { say "STOPPED: $*"; exit 1; }
stop_pod() {
  say "end: $(spent) spent in total (cap \$$CAP)"
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: Pod left RUNNING (billing) - stop it by hand"; return; fi
  echo "== stopping the Pod"; runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
}
trap stop_pod EXIT
fetch() {
  [ "${AZ_SKIP_FETCH:-0}" = "1" ] && return 0   # tests: the script is already in place
  curl -fsSL "https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/fluxrt_lora/$1" -o "$WS/$1" \
    || fail "download $1"
}
newest_train_log() { ls -t "$REPORTS"/fluxlora_fxreal6_*.log 2>/dev/null | head -1; }

say "start: cap \$$CAP, price \$$PRICE/h, already spent \$$SPENT0"
fetch fx6_train.sh
RESUME="${AZ_RESUME:-0}"   # 1: the data and a training run exist; carry on training from the newest save

# ---- resume: no data build, no trial; the speed comes from the training bar of the earlier run ----------------------
if [ "$RESUME" = "1" ]; then
  TLOG=$(newest_train_log)
  [ -n "$TLOG" ] || fail "no earlier training log to measure the speed from"
  SEC=$(python3 - "$TLOG" <<'PYEOF'
import re, statistics, sys
text = open(sys.argv[1], errors="ignore").read()
segs = [x for x in re.split(r"[\r\n]", text) if x.lstrip().startswith("Steps:")]
secs = [float(x) for seg in segs for x in re.findall(r"([0-9.]+)s/it", seg)]
secs += [1 / float(x) for seg in segs for x in re.findall(r"([0-9.]+)it/s", seg) if float(x) > 0]
secs = secs[len(secs) // 3:]
print(f"{statistics.median(secs):.2f}" if secs else "")
PYEOF
)
  ROWS=$(python3 -c "import json; print(json.load(open('$ROOT_DIR/data/fluxrt_lora/fxreal6/region_weight.json')).get('rows_written', 0))" 2>/dev/null)
  DONE_STEP=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  [ -n "$SEC" ] && [ -n "$ROWS" ] && [ "$ROWS" -gt 0 ] || fail "could not read the training speed or the row count"
  say "resume: $SEC s per step measured on the training bar, $ROWS rows, newest save step ${DONE_STEP:-none}"
else
# ---- 1. data build -----------------------------------------------------------------------------------------------
say "1. waiting for the data build ($DATA_LOG)"
while true; do
  grep -q "FX6_DATA_DONE" "$DATA_LOG" 2>/dev/null && break
  if ! pgrep -f "fx6_data.sh" >/dev/null && ! pgrep -f "build_real.py" >/dev/null && ! pgrep -f "region_weight" >/dev/null; then
    tail -30 "$DATA_LOG" 2>/dev/null
    fail "the data build is not running and did not finish (see the lines above)"
  fi
  too_late=$(awk -v l="$(left_dollars)" -v e="$EVAL_HOURS" -v p="$PRICE" -v m="$MARGIN" 'BEGIN { print (l < e * p + m + 3) ? 1 : 0 }')
  [ "$too_late" = "1" ] && fail "the data build has used too much of the budget to train and evaluate"
  sleep 60
done
grep -E "^(DATASET|REGIONS|SAMPLES)" "$DATA_LOG" | cut -c1-500
ANCHORS=$(grep -oE '"anchors": [0-9]+' "$DATA_LOG" | tail -1 | grep -oE '[0-9]+$')
[ "${ANCHORS:-0}" -gt 0 ] || fail "the dataset has no earlier-frame rows (anchors=${ANCHORS:-none}): built without --anchor-frac"
say "1. data build done: $ANCHORS rows with the earlier frame"

# ---- 2. trial ----------------------------------------------------------------------------------------------------
say "2. trial: 200 steps"
AZ_REF=$AZ_REF AZ_STAGE=trial AZ_NO_STOP=1 AZ_PRICE=$PRICE bash "$WS/fx6_train.sh" > "$WS/fx6_trial.log" 2>&1
TLOG=$(newest_train_log)
grep -hE "LoRA initialised from|^REF_SIZES|^ANCHOR_BATCHES|^CACHE_RAM|^CACHE_CPU|^TRIAL|FX6_TRIAL_DONE|TRAIN_EXIT" \
  "$WS/fx6_trial.log" | cut -c1-300
if grep -qE "OutOfMemoryError|CUDA out of memory|NOT_ENOUGH_CPU_RAM|Traceback" \
     "$WS/fx6_trial.log" "$TLOG" "$REPORTS/fx6_trial_run.log" 2>/dev/null; then
  grep -hE "OutOfMemoryError|CUDA out of memory|NOT_ENOUGH_CPU_RAM|Error" "$WS/fx6_trial.log" "$TLOG" "$REPORTS/fx6_trial_run.log" \
    2>/dev/null | tail -8
  fail "the trial hit an error (lines above)"
fi
grep -qiE "loss=(nan|inf)" "$TLOG" 2>/dev/null && fail "the trial's loss is not a number (NaN / inf)"
grep -q "FX6_TRIAL_DONE" "$WS/fx6_trial.log" || fail "the trial did not finish"
SEC=$(grep -oE "^TRIAL [0-9.]+ s per step" "$WS/fx6_trial.log" | grep -oE "[0-9.]+" | head -1)
ROWS=$(grep -oE "^TRIAL [0-9.]+ s per step \(batch 4\), [0-9]+ rows" "$WS/fx6_trial.log" | grep -oE "[0-9]+ rows" | grep -oE "[0-9]+")
[ -n "$SEC" ] && [ -n "$ROWS" ] || fail "no step time in the trial log"
say "2. trial ok: $SEC s per step, $ROWS rows"

fi

# ---- 3. size the training ----------------------------------------------------------------------------------------
FULL=$(awk -v r="$ROWS" -v p="$PASSES" 'BEGIN { printf "%d", int(p * r / 4 / 100) * 100 }')
AFFORD=$(awk -v l="$(left_dollars)" -v e="$EVAL_HOURS" -v pr="$PRICE" -v m="$MARGIN" -v s="$SEC" \
  'BEGIN { h = (l - m) / pr - e; if (h < 0) h = 0; printf "%d", int(h * 3600 / s / 100) * 100 }')
DONE_STEP="${DONE_STEP:-0}"
STEPS=$FULL
[ $((DONE_STEP + AFFORD)) -lt "$STEPS" ] && STEPS=$((DONE_STEP + AFFORD))
TODO=$((STEPS - DONE_STEP))
say "3. $PASSES passes = $FULL steps; the money left (\$$(left_dollars), keeping $EVAL_HOURS h for evaluation and \$$MARGIN spare) pays for $AFFORD; training to $STEPS"
[ "$TODO" -ge "$MIN_STEPS" ] || fail "only $TODO more steps fit in the budget (under $MIN_STEPS): not worth training; top up or raise AZ_CAP"
# time limit: the steps at the measured speed plus 20 minutes of loading and saving, never past the money
MINUTES=$(awk -v n="$TODO" -v s="$SEC" -v l="$(left_dollars)" -v e="$EVAL_HOURS" -v pr="$PRICE" -v m="$MARGIN" \
  'BEGIN { t = n * s / 60 + 20; cap = ((l - m) / pr - e) * 60; if (t > cap) t = cap; printf "%d", t }')
say "3. training time limit $MINUTES min (at most about \$$(awk -v t="$MINUTES" -v p="$PRICE" 'BEGIN { printf "%.0f", t / 60 * p }'))"

# ---- 4. training -------------------------------------------------------------------------------------------------
say "4. training to step $STEPS"
AZ_REF=$AZ_REF AZ_STAGE=train AZ_MAX_STEPS=$STEPS AZ_MAX_MINUTES=$MINUTES AZ_NO_STOP=1 AZ_PRICE=$PRICE \
  bash "$WS/fx6_train.sh" > "$WS/fx6_train.log" 2>&1
grep -hE "LoRA initialised from|^ANCHOR_BATCHES|TRAIN_EXIT|newest save|FX6_TRAIN|OutOfMemoryError|Traceback|^[0-9a-f]{64}" \
  "$WS/fx6_train.log" | cut -c1-300
TLOG=$(newest_train_log)
grep -oE "(loss|detail|pair|colour|texture)=[0-9.e+-]+" "$TLOG" 2>/dev/null | tail -10 | tr '\n' ' '
echo
grep -qiE "loss=(nan|inf)" "$TLOG" 2>/dev/null && say "4. WARNING: NaN/inf in the training log; the evaluation shows whether the saves are usable"
ls "$OUT" 2>/dev/null | grep -qE "checkpoint-|pytorch_lora_weights" || fail "training left no save"
say "4. training finished or stopped at its limit"

# ---- 5. evaluation -----------------------------------------------------------------------------------------------
EVAL_MIN=$(awk -v l="$(left_dollars)" -v pr="$PRICE" -v m="$MARGIN" 'BEGIN { t = (l - m) / pr * 60; if (t > 150) t = 150; printf "%d", t }')
[ "$EVAL_MIN" -ge 20 ] || fail "no money left for the evaluation (the trained LoRA is saved; evaluate later)"
say "5. evaluation, time limit $EVAL_MIN min"
timeout -s INT "${EVAL_MIN}m" env AZ_REF="$AZ_REF" AZ_STAGE=eval AZ_NO_STOP=1 AZ_PRICE="$PRICE" bash "$WS/fx6_train.sh" \
  > "$WS/fx6_eval.log" 2>&1
grep -hE "^(render|FACESKIN|LOGO_LEAK|PRIORITY|GATE|HIJAB|STEADY)|_EXIT=|FX6_EVAL_DONE|NO_FX6_LORA|using the newest save" \
  "$WS/fx6_eval.log" | cut -c1-400
if grep -q "FX6_EVAL_DONE" "$WS/fx6_eval.log"; then
  say "5. evaluation done"
else
  say "5. evaluation did not finish inside its limit (partial results above)"
fi
say "DONE: LoRA $OUT/pytorch_lora_weights.safetensors; reports and videos $ROOT_DIR/eval/fx6_review.tar"

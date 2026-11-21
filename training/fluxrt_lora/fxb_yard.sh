#!/usr/bin/env bash
# The better yardstick (owner, 2026-11-21, step B of the plan). Trains nothing and never touches an endpoint.
#
# Why: favourite #2 with another random seed scored up to +4.2% against itself (flicker +8.5%), so one render of
# the 39 test clips cannot show a difference under about 4%. This run measures on more clips, at several seeds,
# with a 95% range on every figure (eval.yardstick), and answers two questions:
#   1. Is there a better seed for favourite #2? The best seeds of the sweep (step A) were picked on the old test
#      clips, so they are judged again on FRESH clips that no run has used. A middle and the worst seed of the sweep
#      go along, to see whether the sweep's order means anything on new clips.
#   2. Are fxreal8's steps 750 and 1000 different from favourite #2? Each is rendered at the live seed and at the
#      two best seeds, and compared with favourite #2 AT THE SAME SEED, on all clips.
#
# What it does:
#   1. picks the fresh set (AZ_FRESH_SET, default fx12test: AZ_FRESH_N clips, tops, dresses and bottoms in turn,
#      none used before) and renders favourite #2 on it at the live seed (52)
#   2. the seeds: the best AZ_TOP_SEEDS_N (3) of the sweep in eval/fxa/fx6b, one from the middle, the worst
#      (AZ_YARD_SEEDS="54 61" names them instead)
#   3. renders (GPU, one after the other; clips that have a video are never rendered again) and scores (CPU, beside
#      it; scores an earlier run made on the old clips are reused): favourite #2 at those seeds, then the models
#      (AZ_YARD_MODELS, default "fxreal8:750 fxreal8:1000") at the live seed and at the best AZ_MODEL_SEEDS_N (2)
#   4. the report: YARD lines (eval/fxa/yard/yardstick.json), two labelled videos on the fresh clips (the seeds
#      side by side; the models side by side), eval/yard_review.tar. Then the Pod stops itself.
# Money: never past AZ_CAP (counting AZ_SPENT) nor past AZ_BALANCE minus AZ_BALANCE_KEEP (1.50); what is rendered
# and scored when the money ends is kept, and the report is made from it. Safe to start again.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fxb_yard.sh -o /workspace/fxb_yard.sh
#   AZ_REF=<SHA> AZ_SPENT=86.00 AZ_CAP=100 AZ_PRICE=2.80 AZ_BALANCE=40.00 nohup bash /workspace/fxb_yard.sh >> /workspace/yard.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
SPENT0="${AZ_SPENT:?set AZ_SPENT to the dollars already spent on training (e.g. 86.00)}"
AZ_ROOT="${AZ_ROOT:-/workspace/azyah-vton}"
export AZ_REF AZ_ROOT
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
CAP="${AZ_CAP:-100}"
PRICE="${AZ_PRICE:-2.80}"
BALANCE="${AZ_BALANCE:-}"
KEEP="${AZ_BALANCE_KEEP:-1.50}"
END_MIN="${AZ_END_MIN:-12}"                  # kept at the very end: the report, the videos, stop
FRESH="${AZ_FRESH_SET:-fx12test}"
FRESH_N="${AZ_FRESH_N:-48}"
OLD="${AZ_OLD_SETS:-fx6test dress10 cases5}"
TOP_N="${AZ_TOP_SEEDS_N:-3}"
SEEDS="${AZ_YARD_SEEDS:-}"
MODEL_SEEDS_N="${AZ_MODEL_SEEDS_N:-2}"
MODELS="${AZ_YARD_MODELS-fxreal8:750 fxreal8:1000}"
WEIGHTS="${AZ_SET_WEIGHTS:-fx6test=1,dress10=1,cases5=0.5,$FRESH=2}"
FAV2_SHA="${AZ_FAV2_SHA:-8a2f5635}"
LIVE_SEED=52
REF_SIZE=528x704
TAG=r528
FAV="fav2_$TAG"
C="$AZ_ROOT/checkpoints"; S="$AZ_ROOT/data/staging"; E="$AZ_ROOT/eval/fxa"; R="$E/results"
Y="$E/yard"; SWEEP="$E/fx6b"
PY="$AZ_ROOT/venv-fluxlora/bin/python"; MP="$AZ_ROOT/venv-mp/bin/python"
SEG="$AZ_ROOT/selfie_multiclass_256x256.tflite"
INIT="$AZ_ROOT/outputs/fluxrt_lora/blend05s065"
START=$(date +%s)
mins() { echo "$(( ($(date +%s) - START) / 60 ))"; }
spent_run() { awk -v s="$START" -v n="$(date +%s)" -v p="$PRICE" 'BEGIN { printf "%.2f", (n - s) / 3600 * p }'; }
say() { echo "YARD $(date -u +%H:%M) [$(mins) min, \$$(awk -v a="$SPENT0" -v b="$(spent_run)" 'BEGIN { printf "%.2f", a + b }') total] $*"; }

# the same spend record as the training runs: a second start of the SAME line counts what the first one spent
SPENT_GIVEN="$SPENT0"; BALANCE_GIVEN="${BALANCE:-none}"
LEDGER="$AZ_ROOT/eval/fxa/fx6b/spend"
mkdir -p "$LEDGER" 2>/dev/null || true
PRIOR=$(cat "$LEDGER"/*.run 2>/dev/null | awk -v a="$SPENT_GIVEN" -v b="$BALANCE_GIVEN" '$1 == a && $2 == b { s += $3 } END { printf "%.2f", s + 0 }')
SPENT0=$(awk -v a="$SPENT_GIVEN" -v p="$PRIOR" 'BEGIN { printf "%.2f", a + p }')
[ -n "$BALANCE" ] && BALANCE=$(awk -v a="$BALANCE" -v p="$PRIOR" 'BEGIN { printf "%.2f", a - p }')
RUN_FILE="$LEDGER/$START.run"
record_spend() { echo "$SPENT_GIVEN $BALANCE_GIVEN $(spent_run)" > "$RUN_FILE.tmp" 2>/dev/null && mv -f "$RUN_FILE.tmp" "$RUN_FILE"; }
ALLOWED=$(awk -v c="$CAP" -v s="$SPENT0" -v p="$PRICE" -v b="$BALANCE" -v k="$KEEP" 'BEGIN {
  m = c - s; if (b != "") { bb = b - k; if (bb < m) m = bb }
  t = m / p * 60; if (t < 0) t = 0; printf "%d", t }')
left_min() { echo $(( ALLOWED - $(mins) )); }
out_of_money() { [ "$(left_min)" -le "$END_MIN" ]; }
( while :; do record_spend; sleep 60; done ) >/dev/null 2>&1 &
SPEND_PID=$!
SCORE_PID=""
stop_pod() {
  [ -n "$SCORE_PID" ] && kill "$SCORE_PID" 2>/dev/null
  kill "$SPEND_PID" 2>/dev/null; record_spend
  say "end: \$$(spent_run) this run (cap \$$CAP${BALANCE:+, balance at start \$$BALANCE})"
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: Pod left RUNNING (billing) - stop it by hand"; return; fi
  echo "== stopping the Pod"; runpodctl stop pod "${RUNPOD_POD_ID:-}" 2>/dev/null || true
}
trap stop_pod EXIT
fail() { say "STOPPED: $*"; exit 1; }

[ "$PRIOR" != "0.00" ] && say "earlier starts of this same line spent \$$PRIOR (counted)"
say "start: cap \$$CAP, already spent \$$SPENT0, price \$$PRICE/h${BALANCE:+, RunPod balance \$$BALANCE (keeping \$$KEEP)}: $ALLOWED min allowed"
[ "$ALLOWED" -ge $(( END_MIN + 30 )) ] || fail "only $ALLOWED minutes of money left (need at least $(( END_MIN + 30 ))): top up and/or raise AZ_CAP"

# ---- 0. GPU, code, tools
for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo "waiting for network"; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || fail NO_GPU
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || fail CODE_FAILED
T="$AZ_ROOT/azyah-code/training"
export PYTHONPATH="$T"
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
[ -x "$PY" ] && "$PY" -c "import torch, diffusers, pyarrow" 2>/dev/null || fail "NO_RENDER_VENV: $PY"
[ -x "$MP" ] && "$MP" -c "import mediapipe, av, numpy, pyarrow, PIL" 2>/dev/null || fail "NO_MEDIAPIPE_VENV: $MP"
INIT_W="$INIT/pytorch_lora_weights.safetensors"
[ -f "$INIT_W" ] || fail "NO_FAVOURITE2: $INIT"
FAV_SUM=$(sha256sum "$INIT_W" | cut -c1-64)
case "$FAV_SUM" in "$FAV2_SHA"*) ;; *) fail "WRONG_FAVOURITE2: $INIT is ${FAV_SUM:0:8}..., expected $FAV2_SHA...";; esac
say "0. favourite #2 confirmed (${FAV_SUM:0:8}...)"
mkdir -p "$Y"

# ---- 1. the test clips: the old sets as they are, and a fresh set no run has used
SETS=""
for s in $OLD; do
  if [ -f "$E/$s.json" ] && ls "$R/$s"/*/out_"$FAV".mp4 >/dev/null 2>&1; then SETS="$SETS $s"
  else say "NOTE: old test set $s missing (no $E/$s.json or no favourite #2 videos): left out"; fi
done
OLD="${SETS# }"
[ -n "$OLD" ] || fail "NO_TEST_SETS"
n_items() { "$PY" -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" "$E/$1.json" 2>/dev/null || echo 0; }
n_videos() { ls "$R/$1"/*/out_"$2".mp4 2>/dev/null | wc -l; }
if [ -n "$FRESH" ] && [ ! -f "$E/$FRESH.json" ]; then
  ( cd "$T" && "$PY" -m eval.step_a pick "$AZ_ROOT" --set "$FRESH" --n "$FRESH_N" --auto-describe ) > "$Y/pick_$FRESH.log" 2>&1 \
    || say "NOTE: the fresh set could not be picked (see $Y/pick_$FRESH.log)"
fi
if [ -n "$FRESH" ] && [ "$(n_items "$FRESH")" -ge "${AZ_FRESH_MIN:-12}" ]; then
  say "1. fresh test set $FRESH: $("$PY" -c "
import collections, json, sys
d = json.load(open(sys.argv[1]))
c = collections.Counter(i.get('category') for i in d)
print(len(d), 'clips no run has used (' + ', '.join(f'{v} {k}' for k, v in sorted(c.items())) + ')')" "$E/$FRESH.json")"
else
  [ -n "$FRESH" ] && say "1. NOTE: no fresh set ($(n_items "$FRESH") unused held-out clips found, need ${AZ_FRESH_MIN:-12}): the old clips only"
  FRESH=""
fi
SETS="$OLD${FRESH:+ $FRESH}"

G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --steps 2 --prompt-style live --ref-size $REF_SIZE"
render() {  # set, name, weights, extra renderer arguments -> 0 when every clip of the set has that video
  local s="$1" n="$2" w="$3" x="${4:-}" want have
  want=$(n_items "$s"); have=$(n_videos "$s" "$n")
  [ "$want" -gt 0 ] && [ "$have" -ge "$want" ] && return 0
  ( cd "$T" && timeout -s INT 50m $G --lora "$w" --name "$n" $x --clip-list "$E/$s.json" --staging "$S/vivid" --out "$R/$s" --max-items 200 ) \
    > "$Y/render_${s}_$n.log" 2>&1
  have=$(n_videos "$s" "$n")
  say "render $s/$n: $(grep -c '"model"' "$Y/render_${s}_$n.log") new clips, $have of $want"
  [ "$have" -gt 0 ] && [ "$have" -ge $(( want * 9 / 10 )) ]
}
score_set() {  # set, name, dest: the scorers of the training runs, without the GATE (the yardstick does not use it)
  local s="$1" n="$2" d="$3"
  [ -f "$d/garment.json" ] && [ -f "$d/face_skin.json" ] && [ -f "$d/stability.json" ] && return 0
  mkdir -p "$d"
  ( cd "$T" && "$MP" -m eval.face_skin_report "$R/$s" --models "$FAV" "$n" --baseline "$FAV" --model "$SEG" --out "$d/face_skin.md" ) \
    > "$d/face_skin.log" 2>&1 && cp "$R/$s/face_skin.json" "$d/face_skin.json"
  ( cd "$T" && "$MP" -m eval.stability_report "$R/$s" --models "$FAV" "$n" --model "$SEG" --out "$d/stability.md" ) > "$d/stability.log" 2>&1
  ( cd "$T" && "$MP" -m eval.garment_report seg "$R/$s" --models "$FAV" "$n" --model "$SEG" ) > "$d/garment_seg.log" 2>&1
  ( cd "$T" && "$PY" -m eval.garment_report score "$R/$s" --models "$FAV" "$n" --baseline "$FAV" --out "$d/garment.md" \
      --json "$d/garment.json" ) > "$d/garment.log" 2>&1
  [ -f "$d/garment.json" ] && [ -f "$d/face_skin.json" ]
}
reuse_scores() {  # name: the scores an earlier run made for this very render on the old clips are copied, not made again
  "$PY" -c "
import json, shutil, sys
from pathlib import Path
sweep, yard, name, sets = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4].split()
n = 0
for d in sorted(sweep.iterdir()) if sweep.is_dir() else []:
    try:
        meta = json.loads((d / 'meta.json').read_text())
    except (OSError, ValueError):
        continue
    if meta.get('name') != name or not (d / '.scored').exists():
        continue
    for s in sets:
        src, dst = d / s, yard / name / s
        if all((src / f).exists() for f in ('garment.json', 'face_skin.json', 'stability.json')) and not (dst / 'garment.json').exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in ('garment.json', 'face_skin.json', 'stability.json'):
                shutil.copy2(src / f, dst / f)
            n += 1
    break
print(n)" "$SWEEP" "$Y" "$1" "$OLD" 2>/dev/null || echo 0
}
JOBS=()         # name<TAB>weights<TAB>seed
job() {  # name, model, seed, weights, label
  mkdir -p "$Y/$1"
  [ -f "$Y/$1/meta.json" ] || "$PY" -c "import json, sys; json.dump({'name': sys.argv[1], 'model': sys.argv[2], 'seed': int(sys.argv[3]), 'weights': sys.argv[4], 'label': sys.argv[5]}, open(sys.argv[6], 'w'))" \
    "$1" "$2" "$3" "$4" "$5" "$Y/$1/meta.json"
  JOBS+=("$1"$'\t'"$4"$'\t'"$3")
  local got; got=$(reuse_scores "$1")
  [ "${got:-0}" -gt 0 ] && say "   $5: the scores of $got old set(s) reused"
  return 0
}

# ---- 2. the seeds: the best of the sweep on the old clips, one from the middle, the worst
if [ -z "$SEEDS" ]; then
  SEEDS=$("$PY" -c "
import json, re, sys
d = json.load(open(sys.argv[1]))
xs = []
for s in d.get('saves', []):
    m = re.match(r'fav2seed(\d+)_', s.get('name') or '')
    if s.get('kind') == 'aa' and m and s.get('score') is not None:
        xs.append((s['score'], int(m.group(1))))
xs.sort(reverse=True)
top = [x[1] for x in xs[:int(sys.argv[2])]]
rest = [x[1] for x in xs[int(sys.argv[2]):]]
extra = []
if len(rest) >= 2:
    extra = [rest[len(rest) // 2 - (1 if len(rest) % 2 == 0 else 0)], rest[-1]]
elif rest:
    extra = [rest[-1]]
print(' '.join(map(str, top)) + '|' + ' '.join(map(str, extra)))" "$SWEEP/best.json" "$TOP_N" 2>/dev/null)
  TOP_SEEDS="${SEEDS%%|*}"; EXTRA_SEEDS="${SEEDS##*|}"
  [ -n "$TOP_SEEDS" ] || fail "NO_SWEEP: no favourite #2 seed runs in $SWEEP/best.json (run the seed sweep first, or name the seeds: AZ_YARD_SEEDS)"
  say "2. seeds from the sweep: best on the old clips $TOP_SEEDS${EXTRA_SEEDS:+; for comparison (middle, worst) $EXTRA_SEEDS}"
  SEEDS="$TOP_SEEDS${EXTRA_SEEDS:+ $EXTRA_SEEDS}"
else
  TOP_SEEDS="$SEEDS"
  say "2. seeds as named: $SEEDS"
fi
for sd in $SEEDS; do
  case "$sd" in [1-9]|[1-9][0-9]|[1-9][0-9][0-9]) ;; *) fail "BAD_SEED: $sd";; esac
  [ "$sd" = "$LIVE_SEED" ] && continue
  job "fav2seed${sd}_$TAG" fav2 "$sd" "$INIT_W" "favourite #2 at seed $sd"
done
MODEL_SEEDS="$LIVE_SEED $(echo $TOP_SEEDS | tr ' ' '\n' | grep -vx "$LIVE_SEED" | head -n "$MODEL_SEEDS_N" | tr '\n' ' ')"
for m in $MODELS; do
  run="${m%%:*}"; step="${m##*:}"; pfx="fx${run#fxreal}"
  f="$AZ_ROOT/outputs/fluxrt_lora/$run/checkpoint-$step/pytorch_lora_weights.safetensors"
  [ -f "$f" ] || { say "NOTE: no save $run step $step ($f): left out"; continue; }
  sum=$(sha256sum "$f" | cut -c1-8)
  w="$AZ_ROOT/outputs/fluxrt_lora/$run/saves/step${step}_${sum}_s65/pytorch_lora_weights.safetensors"
  [ -f "$w" ] || { say "NOTE: $run step $step has no file turned down to 0.65 ($w): left out"; continue; }
  for sd in $MODEL_SEEDS; do
    if [ "$sd" = "$LIVE_SEED" ]; then n="${pfx}${step}x${sum}s65_$TAG"; else n="${pfx}${step}x${sum}s65seed${sd}_$TAG"; fi
    job "$n" "$run step $step" "$sd" "$w" "$run step $step at seed $sd"
  done
done
say "3. ${#JOBS[@]} renders to check on: $SETS (models at seeds $MODEL_SEEDS)"

# ---- 3. render (GPU) and score (CPU, beside it, one at a time: the scorers share files per set)
score_loop() {
  local row n s ok
  while :; do
    local waiting=0
    for row in "${JOBS[@]}"; do
      n="${row%%$'\t'*}"
      [ -f "$Y/$n/.scored" ] || [ -f "$Y/$n/.failed" ] && continue
      if [ ! -f "$Y/$n/.rendered" ]; then
        [ -f "$Y/.renders_done" ] && continue                             # it will not come any more: left out
        waiting=1; break                                                  # in order: wait for this one's videos
      fi
      ok=1
      # the sets of one render side by side (each scorer writes into its own set folder); wait for THESE only
      local pids="" todo=""
      for s in $SETS; do
        [ -n "$(ls "$R/$s"/*/out_"$n".mp4 2>/dev/null | head -1)" ] || continue
        todo="$todo $s"
        score_set "$s" "$n" "$Y/$n/$s" & pids="$pids $!"
      done
      [ -n "$pids" ] && { wait $pids || true; }
      for s in $todo; do [ -f "$Y/$n/$s/garment.json" ] && [ -f "$Y/$n/$s/face_skin.json" ] || ok=0; done
      if [ "$ok" = 1 ]; then touch "$Y/$n/.scored"; say "scored: $("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['label'])" "$Y/$n/meta.json")"
      else
        echo x >> "$Y/$n/.score_tries"
        [ "$(wc -l < "$Y/$n/.score_tries")" -ge 2 ] && { say "NO_SCORES for $n after 2 tries: left out (see $Y/$n/*/*.log)"; touch "$Y/$n/.failed"; }
      fi
    done
    [ "$waiting" = 0 ] && [ -f "$Y/.renders_done" ] && break
    out_of_money && break
    sleep 30
  done
}
rm -f "$Y/.renders_done"
score_loop & SCORE_PID=$!
if [ -n "$FRESH" ]; then
  render "$FRESH" "$FAV" "$INIT_W" "" || fail "NO_BASELINE: favourite #2 could not be rendered on $FRESH (see $Y/render_${FRESH}_$FAV.log)"
fi
for row in "${JOBS[@]}"; do
  IFS=$'\t' read -r n w sd <<< "$row"
  [ -f "$Y/$n/.rendered" ] || [ -f "$Y/$n/.failed" ] && continue
  out_of_money && { say "money limit reached: the remaining renders are left out"; break; }
  x=""; [ "$sd" != "$LIVE_SEED" ] && x="--seed $sd"
  ok=1
  for s in $SETS; do
    out_of_money && { ok=0; break; }
    render "$s" "$n" "$w" "$x" || ok=0
  done
  if [ "$ok" = 1 ]; then touch "$Y/$n/.rendered"
  else
    echo x >> "$Y/$n/.render_tries"
    [ "$(wc -l < "$Y/$n/.render_tries")" -ge 2 ] && { say "NO_RENDERS for $n after 2 tries: left out (see $Y/render_*_$n.log)"; touch "$Y/$n/.failed"; }
  fi
done
touch "$Y/.renders_done"
say "4. renders done; finishing the scores (the money limit leaves $(( $(left_min) - END_MIN )) min)"
while kill -0 "$SCORE_PID" 2>/dev/null; do
  out_of_money && { say "money limit reached during the scores: stopping them"; kill "$SCORE_PID" 2>/dev/null; break; }
  sleep 20
done
SCORE_PID=""

# ---- 4. the report
( cd "$T" && "$PY" -m eval.yardstick "$Y" --baseline "$FAV" --old $OLD ${FRESH:+--fresh $FRESH} --weights "$WEIGHTS" \
    --clips "$E" --json "$Y/yardstick.json" ) > "$Y/yardstick.log" 2>&1 || say "NOTE: the report failed (see $Y/yardstick.log)"
grep -E "^YARD_" "$Y/yardstick.log" | sed 's/^/YARD /'
LEFT=$(for row in "${JOBS[@]}"; do n="${row%%$'\t'*}"; [ -f "$Y/$n/.scored" ] || echo "$n"; done | tr '\n' ' ')
[ -n "$LEFT" ] && say "NOT CHECKED: $LEFT- run the same line again (after a top-up if the money ran out)"
# two videos on the fresh clips (the old ones when there is no fresh set): the seeds side by side, the models side by side
VSET="${FRESH:-$(echo $OLD | cut -d' ' -f1)}"
video() {  # file name, panels...
  local out="$1"; shift
  [ "$#" -ge 2 ] || return 0
  ( cd "$T" && timeout 15m "$PY" -m eval.step_a video "$AZ_ROOT" --set "$VSET" --cols $(( $# + 3 )) --repeats 1 --panels "$@" ) \
    > "$Y/video_$out.log" 2>&1 || true
  [ -f "$E/${VSET}_labelled.mp4" ] && mv -f "$E/${VSET}_labelled.mp4" "$Y/$out.mp4" && say "video: $Y/$out.mp4"
}
if [ "$(left_min)" -gt 4 ]; then
  P=("$FAV=favourite #2, seed $LIVE_SEED (live)")
  for sd in $TOP_SEEDS; do [ -f "$Y/fav2seed${sd}_$TAG/.rendered" ] && P+=("fav2seed${sd}_$TAG=favourite #2, seed $sd"); done
  video yard_seeds "${P[@]}"
  P=("$FAV=favourite #2 (today)")
  for row in "${JOBS[@]}"; do
    IFS=$'\t' read -r n w sd <<< "$row"
    case "$n" in fav2seed*) continue;; esac
    [ "$sd" = "$LIVE_SEED" ] && [ -f "$Y/$n/.rendered" ] && P+=("$n=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['model'])" "$Y/$n/meta.json")")
  done
  video yard_models "${P[@]}"
fi
( cd "$Y" && tar cf "$AZ_ROOT/eval/yard_review.tar" yardstick.json yardstick.log yard_*.mp4 2>/dev/null )
say "YARD_DONE: $AZ_ROOT/eval/yard_review.tar"

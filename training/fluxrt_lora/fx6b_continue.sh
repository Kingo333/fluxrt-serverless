#!/usr/bin/env bash
# fxreal6b (owner, 2026-11-16): carry on training fxreal6 from its step-5800 weights at half the learning rate, check
# every save against favourite #2 while the training goes on, and keep the BEST save, not the last one.
#
#   AZ_TRAIN_FROM=best_blend (owner, 2026-11-17): the blends are scored BEFORE the training, and the training
#      starts from the best blend (exactly: at the blend file's own rank, 128, with alpha = rank) into a new run,
#      AZ_EXP (default fxreal6c; saves named fx6c...). It is skipped when no blend could be made, or when the best
#      blend scores below step 5800 itself (AZ_TRAIN_ALWAYS=1 trains anyway). The videos and the review file of
#      an earlier run are copied to EVAL/fx6b/backup_before_<run>/ first; this run's own get the run's prefix.
#   0. blend test FIRST (owner, 2026-11-16): step 5800 mixed with favourite #2 at AZ_BLENDS percent of the new
#      weights (default "75 50"), each merged into one LoRA file (fluxrt_lora.merge_lora: exact, loads like any
#      LoRA), rendered on the three test sets before the training starts and scored like the saves. If the money
#      runs short, the blend test still finishes first; the training then gets what is left.
#
#   1. start: the step-5800 weights (sha 0ab25649...) are copied to outputs/fluxrt_lora/fxreal6_s5800 and the new
#      run fxreal6b starts from them (fxreal6 itself is never touched); same data as fxreal6 (linked, not copied),
#      same recipe, learning rate 2.5e-5 (fxreal6: 5e-5), AZ_MAX_STEPS more steps (default 2000), a save every 500.
#   2. checks: each save is rendered on the 24 test clips (fx6test), the owner's 5 real cases (cases5) and the 10
#      dresses (dress10) on the same GPU while the training goes on (favourite #2's and step 5800's videos are
#      reused), then scored on the CPU: garment colour, detail, fabric, movement, steadiness, dress LENGTH, face
#      and skin, and the GATE. No hijab set (owner, 2026-11-16).
#   3. pick: fluxrt_lora.pick_best_save ranks every save and step 5800 itself (GATE first, then the weighted score;
#      the 5 real cases count double) -> outputs/fluxrt_lora/fxreal6b/best/ (+ best.json) and labelled videos.
#   4. money: it never goes past the cap (AZ_CAP, default 55, counting AZ_SPENT from earlier runs) nor past the
#      RunPod balance (AZ_BALANCE, the balance when started, minus AZ_BALANCE_KEEP, default 1.50). The training is
#      stopped early enough to check its last save, then the Pod stops itself.
#      AZ_CHECK_DURING_TRAINING=0 makes the check renders wait until the training has ended (slower, less GPU memory).
#   5. safe to stop and start again: every save is kept; checked saves are never checked again. Run the same line
#      again (with the new AZ_SPENT and AZ_BALANCE): unchecked saves are checked first, then the training carries on
#      from the BEST save (later, worse saves are moved to set_aside/, never deleted; AZ_RESUME_FROM=newest keeps
#      them). When no save has beaten step 5800 yet, it carries on from the newest save.
#
#   6. memory (2026-11-16: a 94 GB-RAM Pod was killed while caching): the trainer's text rows (~8 MB each, up to
#      ~60 GB) go to a file on disk (AZ_TEXT_CACHE_DIR; the container disk when it has room, else the volume;
#      deleted at the end), and a check render only starts while the GPU and the CPU RAM have room for it
#      (AZ_RENDER_MIN_GPU_GB, default 22; AZ_RENDER_MIN_RAM_GB, default 30), otherwise it waits for the training.
#
# GPU: an RTX PRO 6000 (96 GB) or an RTX 5090 (32 GB: the checks then wait until the training ends). Needs the
# azyah-vton-train volume at /workspace. Never touches an endpoint.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fx6b_continue.sh -o /workspace/fx6b_continue.sh
#   AZ_REF=<SHA> AZ_SPENT=37.08 AZ_CAP=55 AZ_PRICE=2.79 AZ_BALANCE=11.35 nohup bash /workspace/fx6b_continue.sh > /workspace/fx6b.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
SPENT0="${AZ_SPENT:?set AZ_SPENT to the dollars already spent on training (e.g. 37.08)}"
AZ_ROOT="${AZ_ROOT:-/workspace/azyah-vton}"   # the volume; another root only for tests
export AZ_REF AZ_ROOT
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1
CAP="${AZ_CAP:-55}"
PRICE="${AZ_PRICE:-2.79}"
BALANCE="${AZ_BALANCE:-}"
KEEP="${AZ_BALANCE_KEEP:-1.50}"
STEPS="${AZ_MAX_STEPS:-2000}"
EVERY="${AZ_SAVE_EVERY:-500}"
LR="${AZ_LR:-2.5e-5}"
RESERVE_MIN="${AZ_RESERVE_MIN:-45}"   # kept after the training to check its last save
END_MIN="${AZ_END_MIN:-8}"            # kept at the very end: pick the best, videos, stop
RESUME_FROM="${AZ_RESUME_FROM:-best}"
START_SHA="${AZ_START_SHA:-0ab25649}"
TRAIN_FROM="${AZ_TRAIN_FROM:-s5800}"
case "$TRAIN_FROM" in s5800) EXP="${AZ_EXP:-fxreal6b}";; best_blend) EXP="${AZ_EXP:-fxreal6c}";;
  *) echo "BAD_AZ_TRAIN_FROM: $TRAIN_FROM (s5800 or best_blend)"; exit 1;; esac
PFX="fx6${EXP#fxreal6}"          # fxreal6b -> fx6b (saves, videos), fxreal6c -> fx6c
REF_SIZE=528x704
REF_SIZES=528x704:3,384x512:1
TAG=r528
FAV="fav2_$TAG"
S5800="fx6s5800_$TAG"
SETS="${AZ_EVAL_SETS:-fx6test cases5 dress10}"

C="$AZ_ROOT/checkpoints"; D="$AZ_ROOT/data"; S="$D/staging"; E="$AZ_ROOT/eval/fxa"; R="$E/results"
PY="$AZ_ROOT/venv-fluxlora/bin/python"; MP="$AZ_ROOT/venv-mp/bin/python"
SEG="$AZ_ROOT/selfie_multiclass_256x256.tflite"
INIT="$AZ_ROOT/outputs/fluxrt_lora/blend05s065"          # favourite #2 (its videos are reused)
SRC_OUT="$AZ_ROOT/outputs/fluxrt_lora/fxreal6"
START_DIR="$AZ_ROOT/outputs/fluxrt_lora/fxreal6_s5800"
OUT="$AZ_ROOT/outputs/fluxrt_lora/$EXP"
SAVES="$OUT/saves"
EV="$E/fx6b"
DATA_SRC="$D/fluxrt_lora/fxreal6"; DATA="$D/fluxrt_lora/$EXP"
START=$(date +%s)
mins() { echo "$(( ($(date +%s) - START) / 60 ))"; }
spent_run() { awk -v s="$START" -v n="$(date +%s)" -v p="$PRICE" 'BEGIN { printf "%.2f", (n - s) / 3600 * p }'; }
say() { echo "FX6B $(date -u +%H:%M) [$(mins) min, \$$(awk -v a="$SPENT0" -v b="$(spent_run)" 'BEGIN { printf "%.2f", a + b }') total] $*"; }

# Minutes this run may last: the smaller of what the cap and what the balance allow.
allowed_minutes() {  # cap spent0 price balance keep -> minutes (0 if nothing is left)
  awk -v c="$1" -v s="$2" -v p="$3" -v b="$4" -v k="$5" 'BEGIN {
    m = c - s; if (b != "") { bb = b - k; if (bb < m) m = bb }
    t = m / p * 60; if (t < 0) t = 0; printf "%d", t }'
}
# Spend record on the volume (2026-11-16, Pods started by "Deploy when available" run this line again after any
# restart): every start of the SAME line (same AZ_SPENT and AZ_BALANCE) adds what the earlier starts spent, so a
# restart never gets a fresh budget. A line with a new AZ_SPENT/AZ_BALANCE (already counting those) starts clean.
SPENT_GIVEN="$SPENT0"; BALANCE_GIVEN="${BALANCE:-none}"
LEDGER="$AZ_ROOT/eval/fxa/fx6b/spend"
mkdir -p "$LEDGER" 2>/dev/null || true
PRIOR=$(cat "$LEDGER"/*.run 2>/dev/null | awk -v a="$SPENT_GIVEN" -v b="$BALANCE_GIVEN" '$1 == a && $2 == b { s += $3 } END { printf "%.2f", s + 0 }')
SPENT0=$(awk -v a="$SPENT_GIVEN" -v p="$PRIOR" 'BEGIN { printf "%.2f", a + p }')
[ -n "$BALANCE" ] && BALANCE=$(awk -v a="$BALANCE" -v p="$PRIOR" 'BEGIN { printf "%.2f", a - p }')
RUN_FILE="$LEDGER/$START.run"
record_spend() { echo "$SPENT_GIVEN $BALANCE_GIVEN $(spent_run)" > "$RUN_FILE.tmp" 2>/dev/null && mv -f "$RUN_FILE.tmp" "$RUN_FILE"; }
ALLOWED=$(allowed_minutes "$CAP" "$SPENT0" "$PRICE" "$BALANCE" "$KEEP")
left_min() { echo $(( ALLOWED - $(mins) )); }
( while :; do record_spend; sleep 60; done ) >/dev/null 2>&1 &
SPEND_PID=$!

stop_pod() {
  for p in ${LOOP_PIDS:-}; do kill "$p" 2>/dev/null; done
  kill "$SPEND_PID" 2>/dev/null; record_spend
  say "end: \$$(spent_run) this run (cap \$$CAP${BALANCE:+, balance at start \$$BALANCE})"
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: Pod left RUNNING (billing) - stop it by hand"; return; fi
  echo "== stopping the Pod"; runpodctl stop pod "$RUNPOD_POD_ID" 2>/dev/null || true
}
trap stop_pod EXIT
fail() { say "STOPPED: $*"; exit 1; }
ram_free_gb() {  # free CPU RAM for this container: the smaller of the host's and the container limit's
  local avail lim="" used=0
  avail=$(awk '/MemAvailable/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
  # the working set: what the container uses minus file pages the kernel can drop (disk cache, dataset)
  if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != "max" ]; then
    lim=$(cat /sys/fs/cgroup/memory.max)
    used=$(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1 == "inactive_file" { print $2 }' /sys/fs/cgroup/memory.stat) ))
  elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    lim=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    used=$(( $(cat /sys/fs/cgroup/memory/memory.usage_in_bytes) - $(awk '$1 == "total_inactive_file" { print $2 }' /sys/fs/cgroup/memory/memory.stat) ))
  fi
  if [ -n "$lim" ] && [ "${#lim}" -lt 19 ]; then
    local c=$(( (lim - used) / 1073741824 )); [ "$c" -lt "$avail" ] && avail=$c
  fi
  echo "${avail:-0}"
}
gpu_free_gb() {  # 0 when unknown: a render then waits for the training
  local v; v=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{ printf "%d", $1 / 1024 }')
  echo "${v:-0}"
}

[ "$PRIOR" != "0.00" ] && say "earlier starts of this same line spent \$$PRIOR (counted)"
say "start: cap \$$CAP, already spent \$$SPENT0, price \$$PRICE/h${BALANCE:+, RunPod balance \$$BALANCE (keeping \$$KEEP)}: $ALLOWED min allowed"
[ -n "$BALANCE" ] || say "NOTE: AZ_BALANCE not given: only the cap limits this run; if the RunPod balance runs out first, the Pod stops without picking the best save (run the same line again after a top-up)"
[ "$ALLOWED" -ge $(( RESERVE_MIN + END_MIN + 30 )) ] || fail "only $ALLOWED minutes of money left (need at least $(( RESERVE_MIN + END_MIN + 30 ))): top up and/or raise AZ_CAP"

for i in $(seq 60); do curl -fsS -m 5 https://raw.githubusercontent.com >/dev/null 2>&1 && break; echo "waiting for network"; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || fail NO_GPU
git -C "$AZ_ROOT/azyah-code" fetch -q origin "$AZ_REF" && git -C "$AZ_ROOT/azyah-code" checkout -q "$AZ_REF" || fail CODE_FAILED
T="$AZ_ROOT/azyah-code/training"
export PYTHONPATH="$T"
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
[ -x "$PY" ] && "$PY" -c "import torch, diffusers, pyarrow" 2>/dev/null || fail "NO_RENDER_VENV: $PY"
mp_ok() { [ -x "$MP" ] && "$MP" -c "import mediapipe, av, numpy, pyarrow, PIL" 2>/dev/null; }
if ! mp_ok; then
  say "rebuilding the MediaPipe venv"
  [ -e "$AZ_ROOT/venv-mp" ] && mv "$AZ_ROOT/venv-mp" "$AZ_ROOT/venv-mp-stale-$(date -u +%Y%m%dT%H%M%SZ)"
  export UV_PYTHON_INSTALL_DIR="$AZ_ROOT/uv-python"
  python3 -m pip install -q uv 2>/dev/null || pip install -q uv 2>/dev/null || true
  if python3 -m uv venv --python 3.11 "$AZ_ROOT/venv-mp" >/dev/null 2>&1; then
    python3 -m uv pip install -q --python "$MP" mediapipe==0.10.21 av pillow "numpy<2" "pyarrow==17.0.0" || true
  else
    python3 -m venv "$AZ_ROOT/venv-mp" && "$AZ_ROOT/venv-mp/bin/pip" install -q mediapipe==0.10.21 av pillow "numpy<2" "pyarrow==17.0.0" || true
  fi
  mp_ok || { apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null 2>&1; }
fi
mp_ok || fail "NO_MEDIAPIPE_VENV: $MP"

# ---- 1. the start weights, the data, the test sets
[ -f "$INIT/pytorch_lora_weights.safetensors" ] || fail "NO_FAVOURITE2: $INIT"
mkdir -p "$START_DIR" "$OUT" "$SAVES" "$EV"
if [ ! -f "$START_DIR/pytorch_lora_weights.safetensors" ]; then
  [ -f "$SRC_OUT/pytorch_lora_weights.safetensors" ] || fail "NO_FXREAL6_WEIGHTS: $SRC_OUT"
  cp "$SRC_OUT/pytorch_lora_weights.safetensors" "$START_DIR/pytorch_lora_weights.safetensors"
fi
START_SUM=$(sha256sum "$START_DIR/pytorch_lora_weights.safetensors" | cut -c1-64)
case "$START_SUM" in "$START_SHA"*) ;; *) fail "WRONG_START_WEIGHTS: $START_DIR is ${START_SUM:0:8}..., expected $START_SHA... (the step-5800 weights)";; esac
say "1. start weights: step 5800 of fxreal6 (${START_SUM:0:8}...)"
[ -f "$DATA_SRC/dataset_summary.json" ] && [ -f "$DATA_SRC/region_weight.json" ] || fail "NO_DATASET: $DATA_SRC"
[ -e "$DATA" ] || ln -s "$DATA_SRC" "$DATA"
ROWS=$("$PY" -c "import json; print(json.load(open('$DATA_SRC/region_weight.json')).get('rows_written', 0))" 2>/dev/null || echo 0)
# The trainer's text rows (~8.1 MB each) go to disk. A row is kept once per caption, so the file is about one row per
# distinct caption; the worst case is one per row. The container disk is used when it has room, else the volume.
# A network volume's df can show the whole storage pool, not its quota: AZ_VOLUME_FREE_GB (from the RunPod
# console) overrides it.
DISTINCT=$("$PY" -c "
import glob, pyarrow.parquet as pq
s = set()
for f in glob.glob('$DATA_SRC/data/*.parquet'):
    s.update(pq.read_table(f, columns=['caption']).column('caption').to_pylist())
print(len(s))" 2>/dev/null || echo 0)
[ "${DISTINCT:-0}" -gt 0 ] || DISTINCT=$ROWS
NEED_MAX=$(( ROWS * 9 / 1000 + 5 )); NEED_LIKELY=$(( DISTINCT * 9 / 1000 + 5 ))
free_gb_of() {  # dir -> free GB (the volume: AZ_VOLUME_FREE_GB when given)
  case "$1" in "$AZ_ROOT"/*) [ -n "${AZ_VOLUME_FREE_GB:-}" ] && { echo "$AZ_VOLUME_FREE_GB"; return; };; esac
  df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'
}
if [ -n "${AZ_TEXT_CACHE_DIR:-}" ]; then TEXT_DIR="$AZ_TEXT_CACHE_DIR"
else
  TEXT_DIR=""
  for need in "$NEED_MAX" "$NEED_LIKELY"; do
    for dir in /root/az_text_cache "$AZ_ROOT/tmp_text_cache"; do
      mkdir -p "$dir" 2>/dev/null || continue
      [ "$(free_gb_of "$dir")" -ge "$need" ] && { TEXT_DIR="$dir"; break 2; }
    done
  done
  [ -n "$TEXT_DIR" ] || fail "NO_DISK_FOR_TEXT_CACHE: need about $NEED_LIKELY GB (at most $NEED_MAX GB) free on the container disk (/root: $(free_gb_of /root/az_text_cache) GB) or the volume ($(free_gb_of "$AZ_ROOT/tmp_text_cache") GB)"
fi
say "memory: $(ram_free_gb) GB CPU RAM free, $(gpu_free_gb) GB GPU free; $ROWS rows, $DISTINCT distinct captions: text cache about $NEED_LIKELY GB (at most $NEED_MAX GB) on disk at $TEXT_DIR ($(free_gb_of "$TEXT_DIR") GB free)"
[ "$(readlink -f "$DATA")" = "$(readlink -f "$DATA_SRC")" ] || fail "DATA_MISMATCH: $DATA is not fxreal6's data"
SETS_OK=""
for s in $SETS; do
  if [ -f "$E/$s.json" ] && ls "$R/$s"/*/out_"$FAV".mp4 >/dev/null 2>&1; then SETS_OK="$SETS_OK $s"
  else say "NOTE: test set $s missing (no $E/$s.json or no favourite #2 videos): left out"; fi
done
SETS="${SETS_OK# }"
[ -n "$SETS" ] || fail "NO_TEST_SETS"
say "test sets: $SETS (favourite #2's videos reused)"
# The scorers write fixed file names in each set folder: keep the fxreal6 evaluation's copies once.
for s in $SETS; do
  mkdir -p "$EV/fx6_eval_backup/$s"
  for f in preserve_check.json face_skin.json garment_report_fx6.json; do
    [ -f "$R/$s/$f" ] && [ ! -f "$EV/fx6_eval_backup/$s/$f" ] && cp "$R/$s/$f" "$EV/fx6_eval_backup/$s/$f"
  done
done

# ---- 2. the checks: render (GPU) and score (CPU), one save at a time, in step order
G="$PY -m eval.generate_fluxrt --model-dir $C/FLUX.2-klein-4B --size 288x512 --steps 2 --prompt-style live --ref-size $REF_SIZE"
render() {  # set, name, weights -> 0 when the set has videos of that name
  local s="$1" n="$2" w="$3"
  ( cd "$T" && timeout -s INT 40m $G --lora "$w" --name "$n" --clip-list "$E/$s.json" --staging "$S/vivid" --out "$R/$s" --max-items 30 ) \
    > "$EV/render_${s}_$n.log" 2>&1
  local have; have=$(ls "$R/$s"/*/out_"$n".mp4 2>/dev/null | wc -l)
  say "render $s/$n: $(grep -c '"model"' "$EV/render_${s}_$n.log") new clips, $have in total"
  [ "$have" -gt 0 ]
}
score_set() {  # set, name, dest
  local s="$1" n="$2" d="$3"
  mkdir -p "$d"
  ( cd "$T" && "$MP" -m eval.face_skin_report "$R/$s" --models "$FAV" "$n" --baseline "$FAV" --model "$SEG" --out "$d/face_skin.md" ) \
    > "$d/face_skin.log" 2>&1 && cp "$R/$s/face_skin.json" "$d/face_skin.json"
  ( cd "$T" && "$MP" -m eval.preserve_check "$R/$s" --base "$FAV" --candidates "$n" --out "$d/preserve.md" --model "$SEG" ) \
    > "$d/preserve.log" 2>&1; cp "$R/$s/preserve_check.json" "$d/preserve_check.json" 2>/dev/null
  ( cd "$T" && "$MP" -m eval.stability_report "$R/$s" --models "$FAV" "$n" --model "$SEG" --out "$d/stability.md" ) > "$d/stability.log" 2>&1
  ( cd "$T" && "$MP" -m eval.garment_report seg "$R/$s" --models "$FAV" "$n" --model "$SEG" ) > "$d/garment_seg.log" 2>&1
  ( cd "$T" && "$PY" -m eval.garment_report score "$R/$s" --models "$FAV" "$n" --baseline "$FAV" --out "$d/garment.md" \
      --json "$d/garment.json" ) > "$d/garment.log" 2>&1
  [ -f "$d/garment.json" ] && [ -f "$d/face_skin.json" ] && [ -f "$d/preserve_check.json" ]
}
# A save to check: EV/<id>/meta.json. Step 5800 itself is "s0" (its videos exist already).
register() {  # id, step, name, weights, kind (start/blend/save), label
  local id="$1"
  [ -f "$EV/$id/meta.json" ] && return 0
  mkdir -p "$EV/$id"
  "$PY" -c "import json, sys; json.dump({'step': int(sys.argv[1]), 'name': sys.argv[2], 'weights': sys.argv[3], 'sha256': sys.argv[4], 'kind': sys.argv[6], 'label': sys.argv[7]}, open(sys.argv[5], 'w'))" \
    "$2" "$3" "$4" "$(sha256sum "$4" | cut -c1-64)" "$EV/$id/meta.json" "${5:-save}" "${6:-step $2}"
}
meta() { "$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$EV/$1/meta.json" "$2"; }
register_saves() {  # every finished checkpoint of this run (its file unchanged for a minute)
  local n f sum
  for n in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n); do
    f="$OUT/checkpoint-$n/pytorch_lora_weights.safetensors"
    [ -f "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge 60 ] || continue
    sum=$(sha256sum "$f" | cut -c1-8)
    [ -f "$EV/s${n}x$sum/meta.json" ] && continue
    cp "$f" "$SAVES/step${n}_$sum.safetensors"
    register "${PFX}s${n}x$sum" "$n" "${PFX}${n}x${sum}_$TAG" "$SAVES/step${n}_$sum.safetensors" save "$EXP step $n"
    say "save at step $n registered for checking (s${n}x$sum)"
  done
}
pending() {  # ids in step order whose $1 marker (.rendered/.scored) is missing
  local id
  for id in $(ls "$EV" 2>/dev/null); do
    [ -f "$EV/$id/meta.json" ] && [ ! -f "$EV/$id/$1" ] && [ ! -f "$EV/$id/.failed" ] && echo "$(meta "$id" step) $id"
  done | sort -n | awk '{ print $2 }'
}
eval_deadline_passed() { [ "$(left_min)" -le "$END_MIN" ]; }
render_loop() {
  local id n w s ok
  while ! eval_deadline_passed; do
    # AZ_CHECK_DURING_TRAINING=0: the check renders wait for the training to end (if the shared GPU runs short of memory)
    if [ "${AZ_CHECK_DURING_TRAINING:-1}" = "0" ] && [ ! -f "$EV/.training_done" ]; then sleep 60; continue; fi
    register_saves
    for id in $(pending .rendered); do
      eval_deadline_passed && break
      # while the training runs, a render starts only when the GPU and the CPU RAM have room for it
      if [ ! -f "$EV/.training_done" ] && [ "$(meta "$id" step)" != 0 ]; then
        if [ "$(gpu_free_gb)" -lt "${AZ_RENDER_MIN_GPU_GB:-22}" ] || [ "$(ram_free_gb)" -lt "${AZ_RENDER_MIN_RAM_GB:-30}" ]; then
          [ -f "$EV/.waiting_note" ] || { say "check renders wait for the training to end ($(gpu_free_gb) GB GPU, $(ram_free_gb) GB RAM free)"; touch "$EV/.waiting_note"; }
          break
        fi
      fi
      n=$(meta "$id" name); w=$(meta "$id" weights); ok=1
      for s in $SETS; do render "$s" "$n" "$w" || ok=0; done
      if [ "$ok" = 1 ]; then touch "$EV/$id/.rendered"
      else
        echo x >> "$EV/$id/.render_tries"
        [ "$(wc -l < "$EV/$id/.render_tries")" -ge 3 ] && { say "NO_RENDERS for $id after 3 tries: left out (see $EV/render_*.log)"; touch "$EV/$id/.failed"; }
      fi
    done
    [ -f "$EV/.training_done" ] && [ -z "$(pending .rendered)" ] && { register_saves; [ -z "$(pending .rendered)" ] && break; }
    sleep 60
  done
}
score_loop() {
  local id n s ok
  while ! eval_deadline_passed; do
    for id in $(pending .scored); do
      [ -f "$EV/$id/.rendered" ] || break               # in step order: wait for this one's videos
      eval_deadline_passed && break
      n=$(meta "$id" name); ok=1
      for s in $SETS; do score_set "$s" "$n" "$EV/$id/$s" & done
      wait || true
      for s in $SETS; do [ -f "$EV/$id/$s/garment.json" ] && [ -f "$EV/$id/$s/preserve_check.json" ] || ok=0; done
      if [ "$ok" = 1 ]; then
        touch "$EV/$id/.scored"
        ( cd "$T" && "$PY" -m fluxrt_lora.pick_best_save "$EV" --baseline "$FAV" ) > "$EV/pick.log" 2>&1
        grep -F "($(meta "$id" name))" "$EV/pick.log" | grep "^SAVE" | sed 's/^/FX6B check: /'
      else
        say "SCORING_FAILED for $id (see $EV/$id/*/*.log): will try again"
        sleep 60
      fi
    done
    [ -f "$EV/.training_done" ] && [ -z "$(pending .scored)" ] && [ -z "$(pending .rendered)" ] && break
    sleep 60
  done
}
best_step() {  # the best checked save's step (0 = step 5800 itself), empty when nothing is checked
  ( cd "$T" && "$PY" -m fluxrt_lora.pick_best_save "$EV" --baseline "$FAV" ) > "$EV/pick.log" 2>&1
  "$PY" -c "import json, sys; b = json.load(open(sys.argv[1])).get('best'); print('' if not b else b['step'])" "$EV/best.json" 2>/dev/null
}
best_of() {  # kind [name prefix] -> "score gate_sets sets weights name" of the best checked entry of that kind
  ( cd "$T" && "$PY" -m fluxrt_lora.pick_best_save "$EV" --baseline "$FAV" ) > "$EV/pick.log" 2>&1
  "$PY" -c "
import json, sys
d = json.load(open(sys.argv[1]))
xs = [s for s in d['saves'] if s.get('kind') == sys.argv[2] and s['name'].startswith(sys.argv[3]) and s.get('score') is not None]
xs.sort(key=lambda s: (s['gate_sets'], s['score']), reverse=True)
print(*(xs[0][k] for k in ('score', 'gate_sets', 'sets', 'weights', 'name', 'step')) if xs else '')" "$EV/best.json" "$1" "${2:-}" 2>/dev/null
}

# A run that would overwrite an earlier run's videos and review file keeps a copy first.
BACKUP="$EV/backup_before_$EXP"
if [ "$PFX" != "fx6b" ] && [ ! -d "$BACKUP" ] && { ls "$EV"/fx6b_*_best.mp4 >/dev/null 2>&1 || [ -f "$AZ_ROOT/eval/fx6b_review.tar" ]; }; then
  mkdir -p "$BACKUP"
  cp -p "$EV"/fx6b_*_best.mp4 "$EV/best.json" "$EV/pick.log" "$AZ_ROOT/eval/fx6b_review.tar" "$BACKUP/" 2>/dev/null
  say "1. earlier videos, picks and review file copied to $BACKUP"
fi

register "s0" 0 "$S5800" "$START_DIR/pytorch_lora_weights.safetensors" start "fxreal6 step 5800"
rm -f "$EV/.training_done" "$EV/.waiting_note"

# ---- 2. the blend test first: step 5800 mixed with favourite #2, one merged file per percentage
BLENDS="${AZ_BLENDS-75 50}"
for b in $BLENDS; do
  case "$b" in [1-9][0-9]) ;; *) say "BAD_AZ_BLENDS: $b (two-digit percentages, 10 to 99): skipped"; continue;; esac
  bdir="$OUT/blends/b$b"
  if [ ! -f "$bdir/pytorch_lora_weights.safetensors" ]; then
    ( cd "$T" && "$PY" -m fluxrt_lora.merge_lora --inputs "$START_DIR/pytorch_lora_weights.safetensors" \
        "$INIT/pytorch_lora_weights.safetensors" --weights "0.$b" "0.$(( 100 - b ))" --scale 1.0 --out "$bdir" \
        --selection-from "$SRC_OUT" --note "fxreal6 step 5800 at $b% + favourite #2 at $(( 100 - b ))%" ) > "$EV/merge_b$b.log" 2>&1 \
      || { say "BLEND_MERGE_FAILED for $b% (see $EV/merge_b$b.log): left out"; continue; }
  fi
  bsum=$(sha256sum "$bdir/pytorch_lora_weights.safetensors" | cut -c1-8)
  register "b${b}x$bsum" "-$b" "fx6blend${b}x${bsum}_$TAG" "$bdir/pytorch_lora_weights.safetensors" blend \
    "blend: step 5800 at $b% + favourite #2"
done
if [ -n "$(pending .rendered)" ]; then
  say "2. blend test: rendering $(pending .rendered | tr '\n' ' ')before the training"
  touch "$EV/.training_done"; render_loop; rm -f "$EV/.training_done"
  say "2. blend renders done"
fi
TRAIN_RANK=64
TRAIN_INIT="$START_DIR"
if [ "$TRAIN_FROM" = "best_blend" ]; then
  # the blends are scored now: the training starts from the best one
  if [ -n "$(pending .scored)" ]; then
    say "2. scoring the blends before the training"
    touch "$EV/.training_done"; score_loop; rm -f "$EV/.training_done"
  fi
  grep -E "^SAVE (blend|fxreal6 step 5800)" "$EV/pick.log" 2>/dev/null | sed 's/^/FX6B check: /'
  read -r BB_SCORE BB_GATE BB_SETS BB_W BB_NAME BB_STEP < <(best_of blend)
  read -r S0_SCORE _ < <(best_of start)
  if [ -z "${BB_W:-}" ]; then
    say "2. NO_BLEND: no blend could be made and checked: NOT TRAINING"; SKIP_TRAIN=1
  elif [ "${AZ_TRAIN_ALWAYS:-0}" != "1" ] && awk -v a="$BB_SCORE" -v b="${S0_SCORE:--1}" 'BEGIN { exit !(a < b) }'; then
    say "2. the best blend ($BB_NAME, score $BB_SCORE) is below step 5800 itself ($S0_SCORE): NOT TRAINING (AZ_TRAIN_ALWAYS=1 trains anyway)"; SKIP_TRAIN=1
  else
    TRAIN_INIT=$(dirname "$BB_W")
    TRAIN_RANK=$("$PY" -c "
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], 'pt') as f:
    k = next(k for k in f.keys() if '.lora_A.' in k)
    print(f.get_slice(k).get_shape()[0])" "$BB_W" 2>/dev/null)
    [ -n "$TRAIN_RANK" ] || fail "BAD_BLEND_FILE: no rank in $BB_W"
    say "2. training starts from the best blend: $BB_NAME (score $BB_SCORE vs favourite #2, GATE $BB_GATE/$BB_SETS), rank $TRAIN_RANK"
  fi
fi

# ---- 3. a restart: check what an earlier run left unchecked, then carry on from the best save
if ls -d "$OUT"/checkpoint-* >/dev/null 2>&1; then
  register_saves
  if [ "$RESUME_FROM" = "best" ] && [ -n "$(pending .scored)" ]; then
    say "3. restart: checking the saves an earlier run left unchecked before training on"
    touch "$EV/.training_done"
    render_loop; score_loop
    rm -f "$EV/.training_done"
  fi
  read -r _ _ _ _ _ B < <(best_of save "$PFX")      # the best save of THIS run
  if [ "$RESUME_FROM" = "best" ] && [ -n "$B" ] && [ "$B" -gt 0 ]; then
    ASIDE="$OUT/set_aside/$(date -u +%Y%m%dT%H%M%SZ)"
    for n in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n); do
      [ "$n" -gt "$B" ] && { mkdir -p "$ASIDE"; mv "$OUT/checkpoint-$n" "$ASIDE/"; }
    done
    [ -d "${ASIDE:-/nonexistent}" ] && say "3. carrying on from the best save, step $B (later saves moved to $ASIDE, kept)" \
      || say "3. carrying on from the best save, step $B (the newest)"
  else
    say "3. carrying on from the newest save (best so far: ${B:-none checked}${B:+ - step 5800 itself when 0})"
  fi
fi

# ---- 4. training, with the checks running beside it
LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
TRAIN_MIN=$(( $(left_min) - RESERVE_MIN - END_MIN ))
render_loop & LOOP_PIDS="$!"
score_loop & LOOP_PIDS="$LOOP_PIDS $!"
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  say "4. no training (see above): only the checks and the final pick"
elif [ "${LAST:-0}" -ge "$STEPS" ]; then
  say "4. training already at step $LAST of $STEPS: only the checks"
elif [ "$TRAIN_MIN" -lt 20 ]; then
  say "4. NOT TRAINING: only $TRAIN_MIN minutes of money left after the checks; top up to carry on"
else
  say "4. training $EXP from step ${LAST:-0} to $STEPS at learning rate $LR, rank $TRAIN_RANK, from $TRAIN_INIT, a save every $EVERY, time limit $TRAIN_MIN min"
  timeout -s INT "${TRAIN_MIN}m" bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/fluxrt_lora/run_train.sh | \
    AZ_REF=$AZ_REF AZ_MANIFEST_DIR=$AZ_ROOT/manifests/research-train AZ_EXP=$EXP \
    KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B AZ_TIME_SHIFT_MU=2.01 AZ_REF_SIZE=$REF_SIZE AZ_REF_SIZES=$REF_SIZES \
    AZ_MAX_STEPS=$STEPS AZ_BATCH=4 AZ_LORA_RANK=$TRAIN_RANK AZ_LR=$LR \
    AZ_CKPT_INTERVAL=$EVERY AZ_CKPT_LIMIT=40 AZ_MASK_COLUMN=mask AZ_MASK_WEIGHT=${AZ_MASK_WEIGHT:-3.0} \
    AZ_PAIR_COLUMN=pair AZ_PAIR_WEIGHT=${AZ_PAIR_WEIGHT:-1.0} AZ_DETAIL_WEIGHT=${AZ_DETAIL_WEIGHT:-0.05} \
    AZ_COLOUR_WEIGHT=${AZ_COLOUR_WEIGHT:-0.5} AZ_TEXTURE_WEIGHT=${AZ_TEXTURE_WEIGHT:-0.3} \
    AZ_ANCHOR_COLUMN=${AZ_ANCHOR_COLUMN:-anchor} AZ_ANCHOR_PROB=${AZ_ANCHOR_PROB:-1.0} \
    AZ_TEXT_CACHE_DIR=$TEXT_DIR AZ_INIT_LORA=$TRAIN_INIT bash" > "$AZ_ROOT/reports/fx6b_train_run.log" 2>&1
  TRAIN_EXIT=$?
  TLOG=$(ls -t "$AZ_ROOT"/reports/fluxlora_${EXP}_*.log 2>/dev/null | head -1)
  grep -hE "LoRA initialised from|^TEXT_CACHE_DISK|^CACHE_RAM|^CACHE_CPU|NOT_ENOUGH_CPU_RAM|OutOfMemoryError|CUDA out of memory|Traceback" \
    "$TLOG" "$AZ_ROOT/reports/fx6b_train_run.log" 2>/dev/null | tail -8 | sed 's/^/FX6B trainer: /'
  rm -f "$TEXT_DIR/text_cache.bin"   # the trainer's own scratch file (made again at every start)
  grep -oE "(loss|detail|pair|colour|texture)=[0-9.e+-]+" "$TLOG" 2>/dev/null | tail -5 | tr '\n' ' '; echo
  LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  [ "$TRAIN_EXIT" = 124 ] && say "training stopped by the money limit at step ${LAST:-none} (exit 124); the saves are kept" \
    || say "training ended (exit $TRAIN_EXIT), newest save: step ${LAST:-none} of $STEPS"
fi
touch "$EV/.training_done"
say "5. finishing the checks of the remaining saves (the money limit leaves $(( $(left_min) - END_MIN )) min)"
# The loops stop between saves at the money limit; a render or scoring already running is cut off there too.
while :; do
  alive=0; for p in $LOOP_PIDS; do kill -0 "$p" 2>/dev/null && alive=1; done
  [ "$alive" = 0 ] && break
  if [ "$(left_min)" -le "$END_MIN" ]; then
    say "money limit reached during the checks: stopping them"
    for p in $LOOP_PIDS; do kill "$p" 2>/dev/null; done
    pkill -INT -f "eval.generate_fluxrt" 2>/dev/null
    pkill -f "eval.(face_skin_report|preserve_check|stability_report|garment_report)" 2>/dev/null
    sleep 5; break
  fi
  sleep 30
done
LOOP_PIDS=""

# ---- 6. the best save
B=$(best_step)
grep -E "^(SAVE|BEST)" "$EV/pick.log" | sed 's/^/FX6B /'
UNCHECKED=$(pending .scored | tr '\n' ' ')
[ -n "$UNCHECKED" ] && say "NOT CHECKED (money ran out): $UNCHECKED- run the same line again after a top-up"
if [ -z "$B" ]; then
  say "NO_BEST: nothing was checked"
else
  BW=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['best']['weights'])" "$EV/best.json")
  BN=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['best']['name'])" "$EV/best.json")
  mkdir -p "$OUT/best"
  cp "$BW" "$OUT/best/pytorch_lora_weights.safetensors" && cp "$EV/best.json" "$OUT/best/best.json"
  cp -f "$DATA_SRC/selection.json" "$OUT/best/selection.json" 2>/dev/null || true
  BL=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['best'].get('label') or 'step ' + str(json.load(open(sys.argv[1]))['best']['step']))" "$EV/best.json")
  say "BEST_SAVE $BL -> $OUT/best/pytorch_lora_weights.safetensors $(sha256sum "$OUT/best/pytorch_lora_weights.safetensors" | cut -c1-16)..."
  grep -E "^VERDICT" "$EV/pick.log" | sed 's/^/FX6B /'
  # the best of each kind, for the videos
  read -r BB_N BB_L BS_N BS_L < <("$PY" -c "
import json, sys
d = json.load(open(sys.argv[1]))
def top(kind):
    xs = [s for s in d['saves'] if s.get('kind') == kind and s.get('score') is not None]
    xs.sort(key=lambda s: (s['gate_sets'], s['score']), reverse=True)
    return (xs[0]['name'], (xs[0].get('label') or '').replace(' ', '_')) if xs else ('-', '-')
print(*top('blend'), *top('save'))" "$EV/best.json")
  if [ "$(left_min)" -gt 4 ]; then
    for s in cases5 dress10; do
      case " $SETS " in *" $s "*) ;; *) continue;; esac
      PANELS=("$FAV=favourite #2 (today)" "$S5800=fxreal6 step 5800")
      [ "$BB_N" != "-" ] && PANELS+=("$BB_N=${BB_L//_/ } (best blend)")
      [ "$BS_N" != "-" ] && PANELS+=("$BS_N=${BS_L//_/ } (best save)")
      ( cd "$T" && timeout 10m "$PY" -m eval.step_a video "$AZ_ROOT" --set "$s" --cols $(( ${#PANELS[@]} + 2 )) --no-legend \
          --panels "${PANELS[@]}" ) > "$EV/video_$s.log" 2>&1 || true
      [ -f "$E/${s}_labelled.mp4" ] && mv -f "$E/${s}_labelled.mp4" "$EV/${PFX}_${s}_best.mp4" && say "video: $EV/${PFX}_${s}_best.mp4"
    done
  fi
fi
( cd "$EV" && tar cf "$AZ_ROOT/eval/${PFX}_review.tar" best.json pick.log */*/*.md "${PFX}"_*_best.mp4 2>/dev/null )
say "FX6B_DONE: reports $AZ_ROOT/eval/${PFX}_review.tar"

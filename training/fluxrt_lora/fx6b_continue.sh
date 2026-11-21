#!/usr/bin/env bash
# fxreal6b (owner, 2026-11-16): carry on training fxreal6 from its step-5800 weights at half the learning rate, check
# every save against favourite #2 while the training goes on, and keep the BEST save, not the last one.
#
#   AZ_TRAIN_FROM=fav2_addon (owner, 2026-11-18) = fxreal8: favourite #2 stays as it is and a small add-on learns
#      on top. fxreal7 lost 14-19% on flicker and 7-9% on fabric at every save: nothing in the training rewarded
#      steadiness where the live model works (the frame-pair term only compared the garment's average colour and
#      was turned off at the live noise levels).
#        a. favourite #2 un-scaled (as fxreal7, sha checked, strength measured) is merged into the FROZEN model;
#           a rank-16 LoRA add-on starts as an exact zero on top (training/patches/flux2-klein-fx8.patch).
#        b. steadiness term: the change between the two predicted pictures of a frame pair must match the change
#           between the two real frames, counted where the real frames barely change, at every noise level;
#           its weight is set from the first steps to AZ_STEADY_RATIO (0.25) of the loss.
#        c. reference term: the distance from favourite #2's own prediction for the same input (AZ_REF_WEIGHT,
#           1.0), so the add-on only moves where the real frames pull harder (garment and face count 3x there).
#        d. a 20-step safety check (AZ_SAFETY_STEPS): the add-on starts exactly at favourite #2, every term is
#           finite, frame pairs are present; otherwise the trainer stops with FX8_SAFETY_FAILED.
#        e. 1,000 steps at 1e-4, a save every 250; each save = favourite #2 + the add-on in ONE file, turned down
#           to 0.65 (it loads at scale 1.0 like favourite #2).
#        f. before the training: favourite #2 with the run-time steadiness filter (eval.steady_filter) at
#           AZ_RUNTIME percent (default "30 50"), kind "runtime": reported (RUNTIME lines), never the pick.
#   AZ_TRAIN_FROM=fav2 (owner, 2026-11-17) = fxreal7: train favourite #2 ITSELF, at the strength it was made at.
#      Favourite #2 is 0.5 x fxreal3 + 0.5 x fxreal4 turned DOWN to strength 0.65 on the live model. fxreal5,
#      fxreal6, 6b and 6c all started from that file, trained on the base model (which pulls the strength back
#      up) and were judged at full strength: all about 7% behind from the first save on, whatever the steps.
#        a. the file is checked (sha 8a2f5635..., the one in docs/fluxrt-favourite-2.md) and un-scaled (x 1/0.65):
#           that is the training start; turned back down to 0.65 it is favourite #2 again (to rounding).
#        b. no GPU: fluxrt_lora.lora_geometry prints how strong step 5800 and the other files are against it,
#           and the training start is checked from that measurement (favourite #2 x 1.54 and nothing else,
#           or no training: UNSCALE_WRONG).
#        c. before the training: favourite #2 with another random seed (the size of chance differences, kind
#           "aa"), and step 5800 turned down to AZ_SCALED percent (default "65 80"). Scored as they are ready.
#        d. training: same data and recipe as fxreal6 but WITHOUT the earlier-frame input (AZ_USE_ANCHOR=0: the
#           live model gets [camera, garment] only), learning rate 5e-5, AZ_MAX_STEPS (2000), a save every 500.
#        e. every save is turned down to AZ_EVAL_SCALE percent (65) before it is rendered, scored and kept:
#           outputs/fluxrt_lora/fxreal7/best/ is a file the live worker loads at scale 1.0, like favourite #2.
#      All modes: the pick is weighted for the kiosk (AZ_SET_WEIGHTS, default fx6test=1,dress10=1,cases5=0.5)
#      and prints AREAS lines (colour, detail, fabric, movement, flicker, face and skin, length).
#      By garment type (owner, 2026-11-21: dresses first, shirts and trousers too): every checked save also gets a
#      GARMENT line (dresses | tops | bottoms: the same score for the test clips of that type), the best one a
#      GARMENT_VERDICT against chance per type and its GARMENT_AREAS lines; at the start the saves already on the
#      volume are listed that way (step 1b, no GPU). At the end one more video per type is cut from the same
#      clips: <run>_dresses_best.mp4, <run>_tops_best.mp4, <run>_bottoms_best.mp4 (AZ_GARMENT_CLIPS per type,
#      default 8; AZ_GARMENT_VIDEOS=0 leaves them out).
#      AZ_AA_SEEDS (default "53"): the random seeds of the chance runs. More than one ("53 54 55") gives a
#      steadier measure of chance: the largest of theirs, overall, per area and per garment type.
#      Longer runs: AZ_MAX_STEPS and AZ_SAVE_EVERY with AZ_RESUME_FROM=newest carry a finished run on from its
#      newest save (fxreal8 from step 1000 to 3000, a save every 500).
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
#      The checks always end by themselves (practice runs, tests/training/test_fluxrt_fx7_dryrun.py): the last
#      save is registered as soon as the training has ended, the scorer only ends after the last render, a model
#      that cannot be rendered (3 tries) or scored (2 tries) is left out, and nothing waits for the money limit.
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
RESERVE_MIN="${AZ_RESERVE_MIN:-45}"   # kept after the training to check its last save
END_MIN="${AZ_END_MIN:-8}"            # kept at the very end: pick the best, videos, stop
RESUME_FROM="${AZ_RESUME_FROM:-best}"
START_SHA="${AZ_START_SHA:-0ab25649}"
TRAIN_FROM="${AZ_TRAIN_FROM:-s5800}"
case "$TRAIN_FROM" in s5800) EXP="${AZ_EXP:-fxreal6b}";; best_blend) EXP="${AZ_EXP:-fxreal6c}";; fav2) EXP="${AZ_EXP:-fxreal7}";;
  fav2_addon) EXP="${AZ_EXP:-fxreal8}";;
  *) echo "BAD_AZ_TRAIN_FROM: $TRAIN_FROM (s5800, best_blend, fav2 or fav2_addon)"; exit 1;; esac
PFX="fx${EXP#fxreal}"            # fxreal6b -> fx6b (saves, videos), fxreal6c -> fx6c, fxreal7 -> fx7, fxreal8 -> fx8
FAV2_MODE=0; case "$TRAIN_FROM" in fav2|fav2_addon) FAV2_MODE=1;; esac   # favourite #2 is the start
case "$TRAIN_FROM" in
  fav2_addon) D_LR=1e-4; D_SCALE=65; D_ANCHOR=0; D_SCALED=""; D_AA=1; D_BLENDS=""; D_STEPS=1000; D_EVERY=250; D_RANK=16; D_RUNTIME="30 50";;
  fav2) D_LR=5e-5; D_SCALE=65; D_ANCHOR=0; D_SCALED="65 80"; D_AA=1; D_BLENDS=""; D_STEPS=2000; D_EVERY=500; D_RANK=64; D_RUNTIME="";;
  *) D_LR=2.5e-5; D_SCALE=100; D_ANCHOR=1; D_SCALED=""; D_AA=0; D_BLENDS="75 50"; D_STEPS=2000; D_EVERY=500; D_RANK=64; D_RUNTIME="";;
esac
STEPS="${AZ_MAX_STEPS:-$D_STEPS}"
EVERY="${AZ_SAVE_EVERY:-$D_EVERY}"
RUNTIME="${AZ_RUNTIME-$D_RUNTIME}"         # favourite #2 with the run-time steadiness filter at these percentages
LR="${AZ_LR:-$D_LR}"
EVAL_SCALE="${AZ_EVAL_SCALE:-$D_SCALE}"   # saves are judged and kept turned down to this percentage (100 = as trained)
case "$EVAL_SCALE" in 100|[1-9][0-9]) ;; *) echo "BAD_AZ_EVAL_SCALE: $EVAL_SCALE (10 to 100)"; exit 1;; esac
USE_ANCHOR="${AZ_USE_ANCHOR:-$D_ANCHOR}"  # 0: no earlier-frame input in training (the live model gets none)
ANCHOR_PROB=0; [ "$USE_ANCHOR" = "1" ] && ANCHOR_PROB="${AZ_ANCHOR_PROB:-1.0}"
SCALED="${AZ_SCALED-$D_SCALED}"           # step 5800 turned down to these percentages, checked before the training
AA="${AZ_AA:-$D_AA}"                      # 1: favourite #2 with another random seed (the size of chance)
AA_SEEDS="${AZ_AA_SEEDS:-53}"             # those seeds; more than one = a steadier measure of chance (53 54 55)
BRANCH_FROM="${AZ_BRANCH_FROM:-}"         # <run>:<step> (fxreal8:1000): this run starts from that run's save
BRANCH_STEP=0
WEIGHTS="${AZ_SET_WEIGHTS:-fx6test=1,dress10=1,cases5=0.5}"   # the kiosk: studio-like sets first
FAV2_SHA="${AZ_FAV2_SHA:-8a2f5635}"       # favourite #2's file (docs/fluxrt-favourite-2.md)
FAV2_STRENGTH="${AZ_FAV2_STRENGTH:-0.65}"
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
UNSCALED_DIR="$AZ_ROOT/outputs/fluxrt_lora/blend05_unscaled"   # favourite #2 / 0.65: fxreal7's training start
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
  for p in ${LOOP_PIDS:-} ${SCORE_PID:-}; do kill "$p" 2>/dev/null; done
  kill "$SPEND_PID" 2>/dev/null; record_spend
  say "end: \$$(spent_run) this run (cap \$$CAP${BALANCE:+, balance at start \$$BALANCE})"
  if [ "${AZ_NO_STOP:-0}" = "1" ]; then echo "== AZ_NO_STOP=1: ${AZ_NO_STOP_NOTE:-Pod left RUNNING (billing) - stop it by hand}"; return; fi
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
# fxreal9 (owner, 2026-11-21): a run that starts from another run's save, with that save's optimiser state, so two
# recipes can be compared from the same point at equal steps. The other run's folder is only read.
if [ -n "$BRANCH_FROM" ]; then
  BRUN="${BRANCH_FROM%%:*}"; BRANCH_STEP="${BRANCH_FROM##*:}"
  case "$BRANCH_STEP" in ''|*[!0-9]*) fail "BAD_AZ_BRANCH_FROM: $BRANCH_FROM (run:step, e.g. fxreal8:1000)";; esac
  [ "$TRAIN_FROM" = "fav2_addon" ] || fail "BAD_AZ_BRANCH_FROM: only with AZ_TRAIN_FROM=fav2_addon"
  [ -n "$BRUN" ] && [ "$BRUN" != "$EXP" ] || fail "BAD_AZ_BRANCH_FROM: $BRANCH_FROM (set AZ_EXP to a new run name)"
  BSRC="$AZ_ROOT/outputs/fluxrt_lora/$BRUN"
  if ls -d "$OUT"/checkpoint-* >/dev/null 2>&1; then
    [ "$(cat "$OUT/branch_from.txt" 2>/dev/null)" = "$BRANCH_FROM" ] \
      || fail "BRANCH_MISMATCH: $OUT was started from '$(cat "$OUT/branch_from.txt" 2>/dev/null)', not from $BRANCH_FROM"
    say "1. $EXP carries on (it started from $BRUN step $BRANCH_STEP)"
  else
    [ -f "$BSRC/checkpoint-$BRANCH_STEP/pytorch_lora_weights.safetensors" ] || fail "NO_BRANCH_POINT: $BSRC/checkpoint-$BRANCH_STEP"
    # copied under another name first, so a copy that was cut off is never taken for a save (nothing is deleted:
    # a cut-off copy stays as a small hidden folder)
    BTMP="$OUT/.branch_tmp_$START"
    cp -r "$BSRC/checkpoint-$BRANCH_STEP" "$BTMP" && mv "$BTMP" "$OUT/checkpoint-$BRANCH_STEP" \
      || fail "BRANCH_COPY_FAILED: $BSRC/checkpoint-$BRANCH_STEP"
    # the steadiness weight measured at that run's first start stays the same
    [ -f "$BSRC/fx8_steady_weight.json" ] && cp "$BSRC/fx8_steady_weight.json" "$OUT/fx8_steady_weight.json"
    echo "$BRANCH_FROM" > "$OUT/branch_from.txt"
    say "1. $EXP starts from $BRUN step $BRANCH_STEP (its save and optimiser state copied; $BRUN itself is not changed)"
  fi
  RESUME_FROM=newest          # nothing of this run is set aside
fi
if [ ! -f "$START_DIR/pytorch_lora_weights.safetensors" ]; then
  [ -f "$SRC_OUT/pytorch_lora_weights.safetensors" ] || fail "NO_FXREAL6_WEIGHTS: $SRC_OUT"
  cp "$SRC_OUT/pytorch_lora_weights.safetensors" "$START_DIR/pytorch_lora_weights.safetensors"
fi
START_SUM=$(sha256sum "$START_DIR/pytorch_lora_weights.safetensors" | cut -c1-64)
case "$START_SUM" in "$START_SHA"*) ;; *) fail "WRONG_START_WEIGHTS: $START_DIR is ${START_SUM:0:8}..., expected $START_SHA... (the step-5800 weights)";; esac
if [ "$FAV2_MODE" = 1 ]; then
  FAV_SUM=$(sha256sum "$INIT/pytorch_lora_weights.safetensors" | cut -c1-64)
  case "$FAV_SUM" in "$FAV2_SHA"*) ;; *) fail "WRONG_FAVOURITE2: $INIT is ${FAV_SUM:0:8}..., expected $FAV2_SHA... (favourite #2's file)";; esac
  say "1. favourite #2 confirmed (${FAV_SUM:0:8}..., blend05s065); step 5800 of fxreal6 (${START_SUM:0:8}...) is used for the strength test"
else
  say "1. start weights: step 5800 of fxreal6 (${START_SUM:0:8}...)"
fi
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
render() {  # set, name, weights, extra renderer arguments -> 0 when the set has videos of that name
  local s="$1" n="$2" w="$3" x="${4:-}"
  ( cd "$T" && timeout -s INT 40m $G --lora "$w" --name "$n" $x --clip-list "$E/$s.json" --staging "$S/vivid" --out "$R/$s" --max-items 30 ) \
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
register() {  # id, step, name, weights, kind (start/blend/save/scaled/aa), label, extra renderer arguments
  local id="$1"
  [ -f "$EV/$id/meta.json" ] && return 0
  mkdir -p "$EV/$id"
  "$PY" -c "import json, sys; json.dump({'step': int(sys.argv[1]), 'name': sys.argv[2], 'weights': sys.argv[3], 'sha256': sys.argv[4], 'kind': sys.argv[6], 'label': sys.argv[7], 'args': sys.argv[8]}, open(sys.argv[5], 'w'))" \
    "$2" "$3" "$4" "$(sha256sum "$4" | cut -c1-64)" "$EV/$id/meta.json" "${5:-save}" "${6:-step $2}" "${7:-}"
}
meta() { "$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$EV/$1/meta.json" "$2"; }
meta_get() { "$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" "$EV/$1/meta.json" "$2" 2>/dev/null; }
IDP="${PFX}s"; [ "$PFX" = "fx6b" ] && IDP="s"   # fxreal6b's saves were registered as s<step>x<sum>
register_saves() {  # every finished checkpoint of this run
  local n f sum id w name label
  for n in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n); do
    f="$OUT/checkpoint-$n/pytorch_lora_weights.safetensors"
    [ -f "$f" ] || continue
    [ "$n" -le "$BRANCH_STEP" ] && continue    # the save this run started from: the other run's, checked there
    # while the trainer runs, a file must be unchanged for a minute (it may still be written). Once the training
    # has ended nothing writes any more: without this the LAST save, less than a minute old when the loops look,
    # was never registered and the run ended without it (found by the practice run, 2026-11-17).
    [ -f "$EV/.training_done" ] || [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge 60 ] || continue
    sum=$(sha256sum "$f" | cut -c1-8)
    id="${IDP}${n}x$sum"
    [ -f "$EV/$id/meta.json" ] && continue
    [ -f "$EV/$id.scale_failed" ] && continue
    w="$SAVES/step${n}_$sum.safetensors"; name="${PFX}${n}x${sum}_$TAG"; label="$EXP step $n"
    cp "$f" "$w"
    if [ "$EVAL_SCALE" != "100" ]; then
      # judged, and kept, turned down to EVAL_SCALE: the way favourite #2 itself was made. fxreal8's saves hold
      # the add-on only: favourite #2 un-scaled + the add-on are merged into one file first.
      local min=("$w") mw=(1.0) note="$EXP step $n turned down to 0.$EVAL_SCALE"
      if [ "$TRAIN_FROM" = "fav2_addon" ]; then
        min=("$UNSCALED_DIR/pytorch_lora_weights.safetensors" "$w"); mw=(1.0 1.0)
        note="favourite #2 + the $EXP add-on at step $n, turned down to 0.$EVAL_SCALE"
      fi
      ( cd "$T" && "$PY" -m fluxrt_lora.merge_lora --inputs "${min[@]}" --weights "${mw[@]}" --scale "0.$EVAL_SCALE" \
          --out "$SAVES/step${n}_${sum}_s$EVAL_SCALE" --selection-from "$DATA_SRC" \
          --note "$note" ) > "$EV/merge_$id.log" 2>&1 \
        || { say "SCALE_FAILED for step $n (see $EV/merge_$id.log): left out"; touch "$EV/$id.scale_failed"; continue; }
      w="$SAVES/step${n}_${sum}_s$EVAL_SCALE/pytorch_lora_weights.safetensors"
      name="${PFX}${n}x${sum}s${EVAL_SCALE}_$TAG"; label="$EXP step $n at 0.$EVAL_SCALE"
    fi
    register "$id" "$n" "$name" "$w" save "$label"
    say "save at step $n registered for checking ($id)"
  done
}
pending() {  # ids in step order whose $1 marker (.rendered/.scored) is missing
  local id
  for id in $(ls "$EV" 2>/dev/null); do
    [ -f "$EV/$id/meta.json" ] && [ ! -f "$EV/$id/$1" ] && [ ! -f "$EV/$id/.failed" ] && echo "$(meta "$id" step) $id"
  done | sort -n | awk '{ print $2 }'
}
eval_deadline_passed() { [ "$(left_min)" -le "$END_MIN" ]; }
render_loop() {  # ends when the training has ended and every save is rendered: then it leaves .renders_done
  local id n w s ok
  rm -f "$EV/.renders_done"
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
      for s in $SETS; do render "$s" "$n" "$w" "$(meta_get "$id" args)" || ok=0; done
      if [ "$ok" = 1 ]; then touch "$EV/$id/.rendered"
      else
        echo x >> "$EV/$id/.render_tries"
        [ "$(wc -l < "$EV/$id/.render_tries")" -ge 3 ] && { say "NO_RENDERS for $id after 3 tries: left out (see $EV/render_*.log)"; touch "$EV/$id/.failed"; }
      fi
    done
    if [ -f "$EV/.training_done" ] && [ -z "$(pending .rendered)" ]; then
      register_saves
      [ -z "$(pending .rendered)" ] && { touch "$EV/.renders_done"; break; }
    fi
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
      # wait for THESE scorers only: a bare "wait" also waits for the spend recorder, which never ends when this
      # loop runs in the main script (2026-11-17: the blend scoring in best_blend mode hung there for 40 minutes)
      local pids=""
      for s in $SETS; do score_set "$s" "$n" "$EV/$id/$s" & pids="$pids $!"; done
      wait $pids || true
      for s in $SETS; do [ -f "$EV/$id/$s/garment.json" ] && [ -f "$EV/$id/$s/preserve_check.json" ] || ok=0; done
      if [ "$ok" = 1 ]; then
        touch "$EV/$id/.scored"
        ( cd "$T" && "$PY" -m fluxrt_lora.pick_best_save "$EV" --baseline "$FAV" --weights "$WEIGHTS" ) > "$EV/pick.log" 2>&1
        grep -F "($(meta "$id" name))" "$EV/pick.log" | grep "^SAVE" | sed 's/^/FX6B check: /'
        grep -F "AREAS $(meta "$id" label):" "$EV/pick.log" | grep "^AREAS" | sed 's/^/FX6B check: /'
        grep -F "GARMENT $(meta "$id" label):" "$EV/pick.log" | grep "^GARMENT " | sed 's/^/FX6B check: /'
      else
        echo x >> "$EV/$id/.score_tries"
        if [ "$(wc -l < "$EV/$id/.score_tries")" -ge 2 ]; then
          # never "until the money runs out": a model that cannot be scored twice is left out, the run goes on
          say "NO_SCORES for $id after 2 tries: left out (see $EV/$id/*/*.log)"; touch "$EV/$id/.failed"
        else
          say "SCORING_FAILED for $id (see $EV/$id/*/*.log): will try once more"; sleep 60
        fi
      fi
    done
    # only after the render loop has finished: it is the one that registers the last save. Ending on "nothing
    # to score right now" left a save rendered but never scored (found by the practice run, 2026-11-17).
    [ -f "$EV/.training_done" ] && [ -f "$EV/.renders_done" ] && [ -z "$(pending .scored)" ] && [ -z "$(pending .rendered)" ] && break
    sleep 60
  done
}
scorer_alive() { [ -n "${SCORE_PID:-}" ] && kill -0 "$SCORE_PID" 2>/dev/null; }
start_scorer() { scorer_alive || { score_loop & SCORE_PID=$!; }; }   # ONE scorer at a time: the scorers share files per set
score_now() { touch "$EV/.renders_done"; score_loop; }               # in the main script, while no render loop runs
scorable() {  # ids that have their videos and no scores yet
  local id; for id in $(pending .scored); do [ -f "$EV/$id/.rendered" ] && echo "$id"; done
}
best_step() {  # the best checked save's step (0 = step 5800 itself), empty when nothing is checked
  ( cd "$T" && "$PY" -m fluxrt_lora.pick_best_save "$EV" --baseline "$FAV" --weights "$WEIGHTS" ) > "$EV/pick.log" 2>&1
  "$PY" -c "import json, sys; b = json.load(open(sys.argv[1])).get('best'); print('' if not b else b['step'])" "$EV/best.json" 2>/dev/null
}
best_of() {  # kind [name prefix] -> "score gate_sets sets weights name" of the best checked entry of that kind
  ( cd "$T" && "$PY" -m fluxrt_lora.pick_best_save "$EV" --baseline "$FAV" --weights "$WEIGHTS" ) > "$EV/pick.log" 2>&1
  "$PY" -c "
import json, sys
d = json.load(open(sys.argv[1]))
xs = [s for s in d['saves'] if s.get('kind') == sys.argv[2] and s['name'].startswith(sys.argv[3]) and s.get('score') is not None]
xs.sort(key=lambda s: (s['gate_sets'], s['score']), reverse=True)
print(*(xs[0][k] for k in ('score', 'gate_sets', 'sets', 'weights', 'name', 'step')) if xs else '')" "$EV/best.json" "$1" "${2:-}" 2>/dev/null
}

# A run that would overwrite an earlier run's videos and review file keeps a copy first.
BACKUP="$EV/backup_before_$EXP"
if [ "$PFX" != "fx6b" ] && [ ! -d "$BACKUP" ] && { ls "$EV"/fx*_best.mp4 >/dev/null 2>&1 || ls "$AZ_ROOT"/eval/fx*_review.tar >/dev/null 2>&1; }; then
  mkdir -p "$BACKUP"
  cp -p "$EV"/fx*_best.mp4 "$EV/best.json" "$EV/pick.log" "$AZ_ROOT"/eval/fx*_review.tar "$BACKUP/" 2>/dev/null
  say "1. earlier videos, picks and review file copied to $BACKUP"
fi

register "s0" 0 "$S5800" "$START_DIR/pytorch_lora_weights.safetensors" start "fxreal6 step 5800"
rm -f "$EV/.training_done" "$EV/.waiting_note" "$EV/.renders_done"

# ---- 1b. by garment type (owner, 2026-11-21): the checks already on the volume, split into dresses, tops and
# bottoms. No GPU and nothing new is rendered: the same per-clip numbers, grouped by what each test clip is.
( cd "$T" && "$PY" -m fluxrt_lora.pick_best_save "$EV" --baseline "$FAV" --weights "$WEIGHTS" ) > "$EV/pick.log" 2>&1 || true
if grep -q "^GARMENT " "$EV/pick.log" 2>/dev/null; then
  say "1b. by garment type, from the checks already on the volume (score vs favourite #2; + = better):"
  grep -E "^GARMENT ($EXP |${BRANCH_FROM:+${BRANCH_FROM%%:*} |}favourite #2)" "$EV/pick.log" | sed 's/^/FX6B /'
  grep -E "^GARMENT_CHANCE" "$EV/pick.log" | sed 's/^/FX6B /'
else
  say "1b. NOTE: no split by garment type (the clip lists in $E do not say what their clips are)"
fi

# ---- 2a. fxreal7/8: favourite #2 un-scaled (the training start), the strength check, the checks without training
if [ "$FAV2_MODE" = 1 ]; then
  if [ ! -f "$UNSCALED_DIR/pytorch_lora_weights.safetensors" ]; then
    UP=$(awk -v s="$FAV2_STRENGTH" 'BEGIN { printf "%.6f", 1 / s }')
    ( cd "$T" && "$PY" -m fluxrt_lora.merge_lora --inputs "$INIT/pytorch_lora_weights.safetensors" --weights 1.0 --scale "$UP" \
        --out "$UNSCALED_DIR" --selection-from "$INIT" \
        --note "favourite #2 before its strength $FAV2_STRENGTH (x $UP): the training start of $EXP, not for live" ) \
      > "$EV/merge_unscaled.log" 2>&1 || { say "UNSCALE_FAILED (see $EV/merge_unscaled.log): NOT TRAINING, only the checks"; SKIP_TRAIN=1; }
  fi
  # how strong the trained files are against favourite #2 (CPU, about a minute; never stops the run)
  GEO=("fxreal6 step 5800=$START_DIR/pytorch_lora_weights.safetensors")
  [ -f "$UNSCALED_DIR/pytorch_lora_weights.safetensors" ] && GEO+=("$EXP start (favourite #2 un-scaled)=$UNSCALED_DIR/pytorch_lora_weights.safetensors")
  for f in "$AZ_ROOT/outputs/fluxrt_lora/fxreal3/pytorch_lora_weights.safetensors" "$AZ_ROOT/outputs/fluxrt_lora/fxreal4/pytorch_lora_weights.safetensors" \
           "$(ls "$AZ_ROOT"/outputs/fluxrt_lora/fxreal6b/saves/step1500_*.safetensors 2>/dev/null | head -1)" \
           "$(ls "$AZ_ROOT"/outputs/fluxrt_lora/fxreal6c/saves/step1500_*.safetensors 2>/dev/null | head -1)"; do
    [ -n "$f" ] && [ -f "$f" ] || continue
    case "$f" in */saves/*) GEO+=("$(basename "$(dirname "$(dirname "$f")")") step 1500=$f");; *) GEO+=("$(basename "$(dirname "$f")") (a parent of favourite #2)=$f");; esac
  done
  ( cd "$T" && timeout 10m "$PY" -m fluxrt_lora.lora_geometry --ref "$INIT/pytorch_lora_weights.safetensors" "${GEO[@]}" \
      --json "$EV/geometry.json" ) 2> "$EV/geometry.err" | grep "^GEOMETRY" | sed 's/^/FX6B /' || true
  say "2. strength check above (no GPU): x1.00 = favourite #2 as it runs live, x1.54 = full training strength"
  # the training start must be favourite #2 x 1/0.65 and nothing else: read from that measurement, before any GPU time
  if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    "$PY" -c "
import json, sys
g = json.load(open(sys.argv[1])).get(sys.argv[2])
if g is None:
    sys.exit(3)
want = 1 / float(sys.argv[3])
sys.exit(0 if abs(g['strength'] / want - 1) < 0.01 and g['new_part'] < 0.02 else 1)" \
      "$EV/geometry.json" "$EXP start (favourite #2 un-scaled)" "$FAV2_STRENGTH" 2>/dev/null
    case "$?" in
      0) say "2. the training start is favourite #2 at full strength (x $(awk -v s="$FAV2_STRENGTH" 'BEGIN { printf "%.2f", 1 / s }')) and nothing else: confirmed";;
      1) say "UNSCALE_WRONG: the training start is not favourite #2 x 1/$FAV2_STRENGTH (see its GEOMETRY line): NOT TRAINING, only the checks"; SKIP_TRAIN=1;;
      *) say "NOTE: STRENGTH_CHECK_MISSING (see $EV/geometry.err): the training start could not be measured; training anyway";;
    esac
  fi
  # chance: favourite #2 with another random seed. Never a candidate; it sets the bar for the others.
  if [ "$AA" = "1" ]; then
    AAN=0
    for sd in $AA_SEEDS; do
      case "$sd" in [1-9]|[1-9][0-9]|[1-9][0-9][0-9]) ;; *) say "BAD_AZ_AA_SEEDS: $sd (whole numbers 1 to 999): skipped"; continue;; esac
      albl="favourite #2, another random seed"; [ "$sd" = "53" ] || albl="favourite #2, random seed $sd"
      register "aa$sd" "-$(( 300 + AAN ))" "fav2seed${sd}_$TAG" "$INIT/pytorch_lora_weights.safetensors" aa "$albl" "--seed $sd"
      AAN=$(( AAN + 1 ))
    done
  fi
  # step 5800 turned down, as favourite #2 was
  for sc in $SCALED; do
    case "$sc" in [1-9][0-9]) ;; *) say "BAD_AZ_SCALED: $sc (two-digit percentages, 10 to 99): skipped"; continue;; esac
    sdir="$AZ_ROOT/outputs/fluxrt_lora/fxreal6_s5800_s$sc"
    if [ ! -f "$sdir/pytorch_lora_weights.safetensors" ]; then
      ( cd "$T" && "$PY" -m fluxrt_lora.merge_lora --inputs "$START_DIR/pytorch_lora_weights.safetensors" --weights 1.0 --scale "0.$sc" \
          --out "$sdir" --selection-from "$SRC_OUT" --note "fxreal6 step 5800 turned down to 0.$sc" ) > "$EV/merge_s5800_s$sc.log" 2>&1 \
        || { say "SCALE_FAILED for step 5800 at 0.$sc (see $EV/merge_s5800_s$sc.log): left out"; continue; }
    fi
    register "s5800s$sc" "-$(( 300 - sc ))" "fx6s5800s${sc}_$TAG" "$sdir/pytorch_lora_weights.safetensors" scaled "fxreal6 step 5800 at 0.$sc"
  done
  # favourite #2 with the run-time steadiness filter (fxreal8): favourite #2's own file, another renderer setting
  for rt in $RUNTIME; do
    case "$rt" in [1-9][0-9]) ;; *) say "BAD_AZ_RUNTIME: $rt (two-digit percentages, 10 to 99): skipped"; continue;; esac
    register "rt$rt" "-$(( 500 - rt ))" "fav2rt${rt}_$TAG" "$INIT/pytorch_lora_weights.safetensors" runtime \
      "favourite #2 + run-time steadiness 0.$rt" "--steady-blend 0.$rt"
  done
  # a blend an earlier run rendered but never scored (its marker was set by hand) is scored now: CPU only
  for d in "$EV"/b[0-9][0-9]x*; do
    [ -f "$d/.rendered" ] && [ -f "$d/.failed" ] && [ ! -f "$d/.scored" ] && [ ! -f "$d/.score_tries" ] || continue
    rm -f "$d/.failed"; say "2. the blend $(basename "$d") was rendered but never scored: scored in this run"
  done
fi

# ---- 2. the blend test first: step 5800 mixed with favourite #2, one merged file per percentage
BLENDS="${AZ_BLENDS-$D_BLENDS}"
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
  say "2. before the training: rendering $(pending .rendered | tr '\n' ' ')"
  [ "$FAV2_MODE" = 1 ] && start_scorer    # scored as each one is ready (best_blend scores them itself, below)
  touch "$EV/.training_done"; render_loop; rm -f "$EV/.training_done"
  say "2. renders before the training done"
fi
TRAIN_RANK="${AZ_LORA_RANK:-$D_RANK}"
TRAIN_INIT="$START_DIR"
[ "$TRAIN_FROM" = "fav2" ] && TRAIN_INIT="$UNSCALED_DIR"
FX8_ENV=""
if [ "$TRAIN_FROM" = "fav2_addon" ]; then
  # favourite #2 un-scaled is merged into the frozen model; the add-on starts at zero (no starting LoRA)
  TRAIN_INIT=""
  FX8_ENV="AZ_FROZEN_LORA=$UNSCALED_DIR AZ_STEADY_RATIO=${AZ_STEADY_RATIO:-0.25} AZ_REF_WEIGHT=${AZ_REF_WEIGHT:-1.0} AZ_SAFETY_STEPS=${AZ_SAFETY_STEPS:-20}"
  # fxreal9: garment terms counted at the live model's noise levels, favourite #2's fabric contrast kept
  FX9_TXT=""
  for v in AZ_PAIR_FLOOR AZ_DETAIL_FLOOR AZ_TEXTURE_FLOOR AZ_KEEP_SPREAD; do
    x="${!v:-}"; [ -n "$x" ] || continue
    case "$x" in *[!0-9.]*|'') fail "BAD_$v: $x (a number)";; esac
    FX8_ENV="$FX8_ENV $v=$x"
  done
  [ -n "${AZ_PAIR_FLOOR:-}" ] && FX9_TXT="$FX9_TXT, frame-to-frame garment colour counted from ${AZ_PAIR_FLOOR} at every noise level"
  [ -n "${AZ_KEEP_SPREAD:-}" ] && FX9_TXT="$FX9_TXT, favourite #2's fabric contrast kept (x${AZ_KEEP_SPREAD})"
  [ -n "${AZ_TEXTURE_FLOOR:-}" ] && FX9_TXT="$FX9_TXT, fabric contrast towards the real frame counted from ${AZ_TEXTURE_FLOOR}"
  [ -n "${AZ_DETAIL_FLOOR:-}" ] && FX9_TXT="$FX9_TXT, detail counted from ${AZ_DETAIL_FLOOR}"
fi
if [ "$TRAIN_FROM" = "best_blend" ]; then
  # the blends are scored now: the training starts from the best one
  if [ -n "$(pending .scored)" ]; then
    say "2. scoring the blends before the training"
    touch "$EV/.training_done"; score_now; rm -f "$EV/.training_done"
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
    render_loop
    while scorer_alive && [ -n "$(pending .scored)" ] && ! eval_deadline_passed; do sleep 30; done
    scorer_alive || score_now      # never two scorers at once; one that ended early is replaced here
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
rm -f "$EV/.renders_done"
render_loop & RENDER_PID=$!; LOOP_PIDS="$RENDER_PID"
start_scorer
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  say "4. no training (see above): only the checks and the final pick"
elif [ "${LAST:-0}" -ge "$STEPS" ]; then
  say "4. training already at step $LAST of $STEPS: only the checks"
elif [ "$TRAIN_MIN" -lt 20 ]; then
  say "4. NOT TRAINING: only $TRAIN_MIN minutes of money left after the checks; top up to carry on"
else
  FROM_TXT="from $TRAIN_INIT"
  [ "$TRAIN_FROM" = "fav2_addon" ] && FROM_TXT="as an add-on over favourite #2 (frozen, un-scaled), steadiness ${AZ_STEADY_RATIO:-0.25} of the loss, reference to favourite #2 x${AZ_REF_WEIGHT:-1.0}, a ${AZ_SAFETY_STEPS:-20}-step safety check${FX9_TXT:-}"
  say "4. training $EXP from step ${LAST:-0} to $STEPS at learning rate $LR, rank $TRAIN_RANK, $FROM_TXT, earlier-frame input $( [ "$USE_ANCHOR" = "1" ] && echo on || echo off ), a save every $EVERY, time limit $TRAIN_MIN min"
  timeout -s INT "${TRAIN_MIN}m" bash -c "curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/fluxrt_lora/run_train.sh | \
    AZ_REF=$AZ_REF AZ_MANIFEST_DIR=$AZ_ROOT/manifests/research-train AZ_EXP=$EXP \
    KLEIN_REPO=black-forest-labs/FLUX.2-klein-base-4B AZ_TIME_SHIFT_MU=2.01 AZ_REF_SIZE=$REF_SIZE AZ_REF_SIZES=$REF_SIZES \
    AZ_MAX_STEPS=$STEPS AZ_BATCH=4 AZ_LORA_RANK=$TRAIN_RANK AZ_LR=$LR \
    AZ_CKPT_INTERVAL=$EVERY AZ_CKPT_LIMIT=40 AZ_MASK_COLUMN=mask AZ_MASK_WEIGHT=${AZ_MASK_WEIGHT:-3.0} \
    AZ_PAIR_COLUMN=pair AZ_PAIR_WEIGHT=${AZ_PAIR_WEIGHT:-1.0} AZ_DETAIL_WEIGHT=${AZ_DETAIL_WEIGHT:-0.05} \
    AZ_COLOUR_WEIGHT=${AZ_COLOUR_WEIGHT:-0.5} AZ_TEXTURE_WEIGHT=${AZ_TEXTURE_WEIGHT:-0.3} \
    AZ_ANCHOR_COLUMN=${AZ_ANCHOR_COLUMN:-anchor} AZ_ANCHOR_PROB=$ANCHOR_PROB \
    AZ_TEXT_CACHE_DIR=$TEXT_DIR AZ_INIT_LORA=$TRAIN_INIT $FX8_ENV bash" > "$AZ_ROOT/reports/fx6b_train_run.log" 2>&1
  TRAIN_EXIT=$?
  TLOG=$(ls -t "$AZ_ROOT"/reports/fluxlora_${EXP}_*.log 2>/dev/null | head -1)
  grep -hE "LoRA initialised from|^FROZEN_LORA|^FX8_|^FX9_|^STEADY_WEIGHT|^ANCHOR_BATCHES|^TEXT_CACHE_DISK|^CACHE_RAM|^CACHE_CPU|NOT_ENOUGH_CPU_RAM|OutOfMemoryError|CUDA out of memory|Traceback" \
    "$TLOG" "$AZ_ROOT/reports/fx6b_train_run.log" 2>/dev/null | tail -12 | sed 's/^/FX6B trainer: /'
  grep -qh "^FX8_SAFETY_FAILED" "$TLOG" "$AZ_ROOT/reports/fx6b_train_run.log" 2>/dev/null \
    && say "4. the safety check stopped the training before it could spend more (FX8_SAFETY_FAILED above): only the checks"
  rm -f "$TEXT_DIR/text_cache.bin"   # the trainer's own scratch file (made again at every start)
  grep -oE "(loss|detail|pair|colour|texture|steady|ref|keep)=[0-9.e+-]+" "$TLOG" 2>/dev/null | tail -8 | tr '\n' ' '; echo
  LAST=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  [ "$TRAIN_EXIT" = 124 ] && say "training stopped by the money limit at step ${LAST:-none} (exit 124); the saves are kept" \
    || say "training ended (exit $TRAIN_EXIT), newest save: step ${LAST:-none} of $STEPS"
fi
touch "$EV/.training_done"
say "5. finishing the checks of the remaining saves (the money limit leaves $(( $(left_min) - END_MIN )) min)"
# The loops stop between saves at the money limit; a render or scoring already running is cut off there too.
# This wait must end by itself: it ends when both loops have ended, and it never waits for something that cannot
# come (a render loop that broke, a scorer that ended early or waits for videos that will not be made).
IDLE=0; RESTARTS=0
while :; do
  if kill -0 "$RENDER_PID" 2>/dev/null; then RENDERING=1; else RENDERING=0; touch "$EV/.renders_done"; fi
  if [ -n "$(scorable)" ]; then
    IDLE=0
    if ! scorer_alive && [ "$RESTARTS" -lt 3 ]; then
      RESTARTS=$(( RESTARTS + 1 )); say "5. saves with videos and no scores, and no scorer running: scorer started again"
      start_scorer
    elif ! scorer_alive && [ "$RENDERING" = 0 ]; then
      say "5. the scorer keeps ending: giving up on $(scorable | tr '\n' ' ')"; break
    fi
  elif [ "$RENDERING" = 0 ]; then
    scorer_alive || break                         # everything that can be checked is checked
    IDLE=$(( IDLE + 1 ))
    if [ "$IDLE" -ge 6 ]; then say "5. nothing left that can be scored: ending the checks"; kill "$SCORE_PID" 2>/dev/null; break; fi
  fi
  if [ "$(left_min)" -le "$END_MIN" ]; then
    say "money limit reached during the checks: stopping them"
    for p in $LOOP_PIDS ${SCORE_PID:-}; do kill "$p" 2>/dev/null; done
    pkill -INT -f "eval.generate_fluxrt" 2>/dev/null
    pkill -f "eval.(face_skin_report|preserve_check|stability_report|garment_report)" 2>/dev/null
    sleep 5; break
  fi
  sleep 30
done
LOOP_PIDS=""

# ---- 6. the best save
B=$(best_step)
grep -E "^(SAVE|AREAS|CHANCE|RUNTIME|BEST)" "$EV/pick.log" | sed 's/^/FX6B /'
grep -E "^GARMENT( |_CHANCE)" "$EV/pick.log" | sed 's/^/FX6B /'      # per garment type: dresses | tops | bottoms
grep -E "^(FABRIC|COMPARE)" "$EV/pick.log" | sed 's/^/FX6B /'        # fabric direction; runs at equal steps
UNCHECKED=$(pending .scored | tr '\n' ' ')
[ -n "$UNCHECKED" ] && say "NOT CHECKED: $UNCHECKED- run the same line again (after a top-up if the money ran out)"
if [ -z "$B" ]; then
  say "NO_BEST: nothing was checked"
else
  BW=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['best']['weights'])" "$EV/best.json")
  BN=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['best']['name'])" "$EV/best.json")
  mkdir -p "$OUT/best"
  cp "$BW" "$OUT/best/pytorch_lora_weights.safetensors" && cp "$EV/best.json" "$OUT/best/best.json"
  # the provenance file the live worker asks for: the one written next to a merged or turned-down file, else the data's
  cp -f "$(dirname "$BW")/selection.json" "$OUT/best/selection.json" 2>/dev/null \
    || cp -f "$DATA_SRC/selection.json" "$OUT/best/selection.json" 2>/dev/null || true
  BL=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['best'].get('label') or 'step ' + str(json.load(open(sys.argv[1]))['best']['step']))" "$EV/best.json")
  say "BEST_SAVE $BL -> $OUT/best/pytorch_lora_weights.safetensors $(sha256sum "$OUT/best/pytorch_lora_weights.safetensors" | cut -c1-16)..."
  grep -E "^VERDICT" "$EV/pick.log" | sed 's/^/FX6B /'
  grep -E "^GARMENT_VERDICT" "$EV/pick.log" | sed 's/^/FX6B /'
  grep -F "GARMENT_AREAS $BL," "$EV/pick.log" | grep "^GARMENT_AREAS" | sed 's/^/FX6B /'
  # the video panels: favourite #2, then the best of each kind (step 5800 turned down, or as it is; the best
  # blend; this run's best save) and the overall pick
  PANELS=("$FAV=favourite #2 (today)")
  while IFS=$'\t' read -r pn pl; do [ -n "$pn" ] && PANELS+=("$pn=$pl"); done < <("$PY" -c "
import json, sys
d = json.load(open(sys.argv[1])); pfx, s5800, mode = sys.argv[2], sys.argv[3], sys.argv[4]
rows = [s for s in d['saves'] if s.get('score') is not None and s.get('kind') != 'aa']
def top(pred):
    xs = sorted((s for s in rows if pred(s)), key=lambda s: (s['gate_sets'], s['score']), reverse=True)
    return xs[0] if xs else None
if mode == 'fav2_addon':
    picks = [top(lambda s: s.get('kind') == 'runtime')]
else:
    picks = [top(lambda s: s.get('kind') == 'scaled') or next((s for s in rows if s['name'] == s5800), None)]
if mode not in ('fav2', 'fav2_addon'):
    picks.append(top(lambda s: s.get('kind') == 'blend'))
picks += [top(lambda s: s.get('kind') == 'save' and s['name'].startswith(pfx)), d.get('best')]
seen = set()
for s in picks:
    if s and s['name'] not in seen:
        seen.add(s['name'])
        print(s['name'] + '\t' + (s.get('label') or s['name']))" "$EV/best.json" "$PFX" "$S5800" "$TRAIN_FROM")
  if [ "$(left_min)" -gt 4 ]; then
    for s in dress10 cases5 fx6test; do
      case " $SETS " in *" $s "*) ;; *) continue;; esac
      [ "$(left_min)" -gt 4 ] || break
      ( cd "$T" && timeout 10m "$PY" -m eval.step_a video "$AZ_ROOT" --set "$s" --cols $(( ${#PANELS[@]} + 2 )) --no-legend \
          --panels "${PANELS[@]}" ) > "$EV/video_$s.log" 2>&1 || true
      [ -f "$E/${s}_labelled.mp4" ] && mv -f "$E/${s}_labelled.mp4" "$EV/${PFX}_${s}_best.mp4" && say "video: $EV/${PFX}_${s}_best.mp4"
    done
  fi
  # one video per garment type (owner, 2026-11-21): dresses, tops, bottoms, cut from the same clips (nothing is
  # rendered again). Each has the item's type and number on screen.
  if [ "${AZ_GARMENT_VIDEOS:-1}" = "1" ] && [ "$(left_min)" -gt 4 ]; then
    GSETS=""; for s in dress10 fx6test cases5; do case " $SETS " in *" $s "*) GSETS="$GSETS $s";; esac; done
    ( cd "$T" && "$PY" -m eval.garment_mix "$AZ_ROOT" --sets $GSETS --per-type "${AZ_GARMENT_CLIPS:-8}" ) > "$EV/garment_mix.log" 2>&1 || true
    grep "^GARMENT_SETS" "$EV/garment_mix.log" | sed 's/^/FX6B /'
    for t in dresses tops bottoms; do
      [ -f "$E/g_$t.json" ] || continue
      [ "$(left_min)" -gt 4 ] || break
      ( cd "$T" && timeout 10m "$PY" -m eval.step_a video "$AZ_ROOT" --set "g_$t" --cols $(( ${#PANELS[@]} + 3 )) \
          --panels "${PANELS[@]}" ) > "$EV/video_g_$t.log" 2>&1 || true
      [ -f "$E/g_${t}_labelled.mp4" ] && mv -f "$E/g_${t}_labelled.mp4" "$EV/${PFX}_${t}_best.mp4" && say "video: $EV/${PFX}_${t}_best.mp4"
    done
  fi
fi
( cd "$EV" && tar cf "$AZ_ROOT/eval/${PFX}_review.tar" best.json pick.log */*/*.md "${PFX}"_*_best.mp4 2>/dev/null )
say "FX6B_DONE: reports $AZ_ROOT/eval/${PFX}_review.tar"

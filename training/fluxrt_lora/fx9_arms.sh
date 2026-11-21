#!/usr/bin/env bash
# fxreal9 (owner, 2026-11-21): two short runs in one Pod, one after the other, both started from fxreal8's step
# 1000 (its save and optimiser state), so each is compared with fxreal8's own continuation at equal steps
# (1500 and 2000), per area and per garment type. Same data, same order, same test clips.
#
#   fxreal9a  "protect":  the frame-to-frame garment colour term counted at every noise level (it was off at the
#                          two noise levels the live model runs at), and favourite #2's fabric contrast kept
#                          (one-sided: the add-on may not go flatter than favourite #2).
#   fxreal9b  "reach":    the same, plus the fabric-contrast and detail terms towards the real frame counted from
#                          0.5 at every noise level (they gave the live noise levels almost no weight).
#
# Each arm is one start of fx6b_continue.sh (AZ_TRAIN_FROM=fav2_addon, AZ_BRANCH_FROM=fxreal8:1000). Both get the
# same AZ_SPENT and AZ_BALANCE: the spend record on the volume counts the first arm for the second, so the two
# together never go past the cap or the balance. The Pod stops itself after the second arm, or at once when the
# first cannot even start. Safe to start again: finished saves and checks are kept.
#
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/fluxrt_lora/fx9_arms.sh -o /workspace/fx9_arms.sh
#   AZ_REF=<SHA> AZ_SPENT=79.50 AZ_CAP=100 AZ_PRICE=2.80 AZ_BALANCE=43.00 nohup bash /workspace/fx9_arms.sh >> /workspace/fx9.log 2>&1 &
set -uo pipefail
AZ_REF="${AZ_REF:?set AZ_REF to the pinned commit}"
: "${AZ_SPENT:?set AZ_SPENT to the dollars already spent on training}"
AZ_ROOT="${AZ_ROOT:-/workspace/azyah-vton}"
ARMS="${AZ_ARMS:-a b}"
STEPS="${AZ_MAX_STEPS:-2000}"
EVERY="${AZ_SAVE_EVERY:-500}"
BRANCH="${AZ_BRANCH_FROM:-fxreal8:1000}"
F="${AZ_CONTINUE_SCRIPT:-/workspace/fx6b_continue_${AZ_REF:0:7}.sh}"
stop_pod() { [ "${AZ_NO_STOP:-0}" = "1" ] && return; echo "== stopping the Pod"; runpodctl stop pod "${RUNPOD_POD_ID:-}" 2>/dev/null || true; }
if [ ! -s "$F" ]; then
  for i in $(seq 30); do
    curl -fsSL "https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/fluxrt_lora/fx6b_continue.sh" -o "$F" && break
    sleep 10
  done
fi
[ -s "$F" ] || { echo "FX9 STOPPED: the run script could not be downloaded"; stop_pod; exit 1; }
N=$(echo $ARMS | wc -w); K=0
for arm in $ARMS; do
  K=$(( K + 1 ))
  case "$arm" in
    a) NAME="protect"; TERMS=(AZ_PAIR_FLOOR=1.0 AZ_KEEP_SPREAD=1.0);;
    b) NAME="reach";   TERMS=(AZ_PAIR_FLOOR=1.0 AZ_KEEP_SPREAD=1.0 AZ_TEXTURE_FLOOR=0.5 AZ_DETAIL_FLOOR=0.5);;
    *) echo "FX9 BAD_AZ_ARMS: $arm (a or b)"; continue;;
  esac
  LAST=0; [ "$K" -lt "$N" ] && LAST=1          # 1 = another arm follows: this one leaves the Pod running
  echo "FX9 $(date -u +%H:%M) arm $K of $N: fxreal9$arm ($NAME) from $BRANCH to step $STEPS: ${TERMS[*]}"
  env "${TERMS[@]}" AZ_REF="$AZ_REF" AZ_ROOT="$AZ_ROOT" AZ_TRAIN_FROM=fav2_addon AZ_EXP="fxreal9$arm" AZ_BRANCH_FROM="$BRANCH" \
    AZ_MAX_STEPS="$STEPS" AZ_SAVE_EVERY="$EVERY" AZ_NO_STOP="$( [ "$LAST" = 1 ] && echo 1 || echo "${AZ_NO_STOP:-0}" )" \
    AZ_NO_STOP_NOTE="the Pod keeps running: the next arm starts now" bash "$F"
  echo "FX9 $(date -u +%H:%M) arm fxreal9$arm ended (exit $?)"
done
echo "FX9_DONE"
# the last arm stops the Pod itself; this covers a list that ended on a bad name
[ "${LAST:-1}" = 1 ] && stop_pod
exit 0

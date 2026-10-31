#!/usr/bin/env bash
# One paid GPU session for the first result (cost cut C2): setup re-check, the ToyData
# smoke test (20 steps + resume to 30, LoRA mode, proves the patch and measures memory
# and step time), then -- only if that passed -- fashion exp1 on our data, all without
# reloading the Pod. Auto-resumes after a spot interruption: just run it again.
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/scripts/run_first_session.sh | \
#     AZ_REF=<SHA> AZ_MANIFEST_DIR=/workspace/azyah-vton/manifests/research-train AZ_MAX_STEPS=500 bash
set -euo pipefail
: "${AZ_REF:?set AZ_REF to the commit SHA Claude Code gave you}"
: "${AZ_MANIFEST_DIR:?set AZ_MANIFEST_DIR (output of data_tools.manifests)}"
SCRIPTS_URL="https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/scripts"
export AZ_REF SCRIPTS_URL
fetch() { curl -fsSL "$SCRIPTS_URL/$1"; }
source <(fetch common.sh)
DONE="$AZ_REPORTS/smoke_passed"

fetch setup_training_pod.sh | bash
if [ ! -f "$DONE" ]; then
  fetch run_smoke.sh | GPUS="${GPUS:-1}" HSDP="${HSDP:-${GPUS:-1}}" bash
  last=$(ls -t "$AZ_REPORTS"/smoke_*.log | grep -v _gpu | head -1)
  if grep -qE "Traceback|CUDA out of memory|RuntimeError|Killed" "$last" || ! find "$AZ_OUT/stage1_smoke" -name "step_30.pth" | grep -q .; then
    echo "SMOKE FAILED: not starting exp1. Paste the summary above to Claude Code, then stop the Pod." >&2
    exit 1
  fi
  touch "$DONE"
  echo "== smoke passed (see $last); starting exp1"
fi
fetch run_fashion.sh | GPUS="${GPUS:-1}" HSDP="${HSDP:-${GPUS:-1}}" bash

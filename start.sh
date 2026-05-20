#!/usr/bin/env bash
# Don't use set -e here: we want diagnostics to ALWAYS run, even if a probe fails.
set -uxo pipefail

LOG=/tmp/fluxrt-startup.log
mkdir -p /tmp
: > "$LOG"

log() { echo "$*" | tee -a "$LOG"; }

log "=== FluxRT serverless start.sh ==="
log "date=$(date -u +%FT%TZ)"
log "whoami=$(whoami)"
log "PWD=$(pwd)"
log "PATH=$PATH"

log "--- env var names (values redacted) ---"
env | awk -F= '{print $1}' | sort | tee -a "$LOG"

log "--- ls /workspace ---"
ls -la /workspace 2>&1 | tee -a "$LOG" || true
log "--- ls /workspace/app ---"
ls -la /workspace/app 2>&1 | tee -a "$LOG" || true
log "--- ls /workspace/FluxRT ---"
ls -la /workspace/FluxRT 2>&1 | tee -a "$LOG" || true

log "--- command -v probes ---"
for c in bash python python3 python3.12 /usr/bin/python3.12; do
  log "command -v $c -> $(command -v "$c" 2>&1 || echo MISSING)"
done

log "--- /usr/bin/python3.12 --version ---"
/usr/bin/python3.12 --version 2>&1 | tee -a "$LOG" || true

log "--- test -f /workspace/app/server.py ---"
if [ -f /workspace/app/server.py ]; then
  log "server.py: FOUND"
else
  log "server.py: MISSING"
fi

log "=== launching server.py (FLUXRT_SKIP_WARMUP=${FLUXRT_SKIP_WARMUP:-unset}) ==="
exec /usr/bin/python3.12 -u /workspace/app/server.py

#!/usr/bin/env bash
set -euxo pipefail

# --- Diagnostics (visible in RunPod worker logs) ---
echo "=== FluxRT serverless start.sh ==="
echo "PWD=$(pwd)"
echo "PATH=$PATH"
echo "--- ls /workspace ---"
ls -la /workspace || true
echo "--- ls /workspace/app ---"
ls -la /workspace/app || true
echo "--- ls /workspace/FluxRT ---"
ls -la /workspace/FluxRT || true
echo "--- which/command -v ---"
command -v bash || true
command -v python || true
command -v python3 || true
command -v python3.12 || true
command -v /usr/bin/python3.12 || true
echo "--- /usr/bin/python3.12 --version ---"
/usr/bin/python3.12 --version || true
echo "=== launching server.py ==="
exec /usr/bin/python3.12 -u /workspace/app/server.py

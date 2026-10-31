#!/usr/bin/env bash
# Phase B: build the training Python env on the volume. Run after download_assets.sh,
# with the azyah-vton-train volume attached at /workspace. Works on the cheap CPU Pod
# (installs only; saves paid GPU time) and again on the GPU Pod (quick re-check):
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/main/training/scripts/setup_training_pod.sh | bash
# The Python env lives on the volume (/workspace/azyah-vton/venv) so later Pods reuse it.
set -euo pipefail
SCRIPTS_URL="${SCRIPTS_URL:-https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/main/training/scripts}"
source <(curl -fsSL "$SCRIPTS_URL/common.sh")

echo "== GPU"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || echo "no GPU here (CPU Pod): installing the environment only"
free -g | head -2
df -h "$AZ_ROOT" | tail -1

[ -d "$AZ_UPSTREAM/.git" ] || { echo "Run download_assets.sh first (upstream code missing)"; exit 1; }
git -C "$AZ_UPSTREAM" checkout -q "$JOYAI_COMMIT"

echo "== Python env (python3.10 venv on the volume)"
command -v python3.10 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv python3.10-dev >/dev/null; }
VENV="$AZ_ROOT/venv"
[ -x "$VENV/bin/python" ] || python3.10 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel setuptools
# Pinned serving requirements (torch 2.9.1 cu128 etc.) plus what training/ imports.
"$VENV/bin/pip" install -q -r "$AZ_UPSTREAM/deploy/requirements.txt"
"$VENV/bin/pip" install -q datasets decord torchdata ftfy peft pyarrow regex safetensors tensorboard "huggingface_hub[cli]>=0.34"
# flash-attn is optional: training falls back to PyTorch SDPA when it is missing.

echo "== smoke config"
curl -fsSL "$SCRIPTS_URL/make_smoke_config.py" -o "$AZ_ROOT/make_smoke_config.py"
"$VENV/bin/python" "$AZ_ROOT/make_smoke_config.py" "$AZ_UPSTREAM/training"

echo "== checks"
"$VENV/bin/python" - <<'PY'
import torch, datasets, peft, decord, torchdata
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu{i}: {p.name} {p.total_memory / 2**30:.0f} GiB sm_{p.major}{p.minor}")
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)
else:
    print("no GPU: skipped GPU checks")
PY
echo "Setup done. Next: run_smoke.sh"

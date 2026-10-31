#!/usr/bin/env bash
# Phase B: build the training Python env on the volume. Run after download_assets.sh,
# with the azyah-vton-train volume attached at /workspace. Works on the cheap CPU Pod
# (installs only; saves paid GPU time) and again on the GPU Pod (quick re-check):
#   curl -fsSL https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/<SHA>/training/scripts/setup_training_pod.sh | bash
# The Python env lives on the volume (/workspace/azyah-vton/venv) so later Pods reuse it.
set -euo pipefail
# Pin every download to one commit (review R4). AZ_REF = a commit SHA; if unset, main's
# current SHA is looked up once and logged, so each run records exactly which code it used.
AZ_REF="${AZ_REF:-$(curl -fsSL https://api.github.com/repos/Kingo333/fluxrt-serverless/commits/main | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')}"
export AZ_REF
SCRIPTS_URL="${SCRIPTS_URL:-https://raw.githubusercontent.com/Kingo333/fluxrt-serverless/$AZ_REF/training/scripts}"
echo "[azyah] scripts ref: $AZ_REF"
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
# One resolver call that repeats upstream's pins, so pip picks extras that fit them.
# A separate unpinned call upgraded huggingface_hub to 2.x, which breaks
# transformers 4.57 and diffusers 0.36 (both need <1.0 / <2.0) (P5, 2026-10-31).
"$VENV/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.9.1+cu128" "torchvision==0.24.1+cu128" "transformers>=4.57.1,<4.58" "diffusers==0.36.0" "accelerate==1.10.1" \
  "huggingface_hub>=0.34,<1.0" datasets decord torchdata ftfy peft pyarrow regex safetensors tensorboard
"$VENV/bin/pip" check || echo "WARNING: pip check reported conflicts (see above)"
# flash-attn is optional: training falls back to PyTorch SDPA when it is missing.

echo "== Azyah training patches (Stage 1 LoRA)"
git -C "$AZ_UPSTREAM" checkout -q -- training
for p in stage1-lora.patch; do
  curl -fsSL "${SCRIPTS_URL%/scripts}/patches/$p" -o "$AZ_ROOT/$p"
  git -C "$AZ_UPSTREAM" apply "$AZ_ROOT/$p" && echo "applied $p"
done

echo "== smoke config"
curl -fsSL "$SCRIPTS_URL/make_smoke_config.py" -o "$AZ_ROOT/make_smoke_config.py"
"$VENV/bin/python" "$AZ_ROOT/make_smoke_config.py" "$AZ_UPSTREAM/training"

echo "== checks"
"$VENV/bin/python" - <<'PY'
import torch, datasets, peft, decord, torchdata, transformers, diffusers, huggingface_hub
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "diffusers", diffusers.__version__,
      "huggingface_hub", huggingface_hub.__version__, "peft", peft.__version__, "datasets", datasets.__version__)
assert int(huggingface_hub.__version__.split(".")[0]) < 1, "huggingface_hub must be <1.0 for transformers 4.57"
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

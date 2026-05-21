FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8765 \
    PORT_HEALTH=8765 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/workspace/.cache/huggingface \
    FLUXRT_ROOT=/workspace/FluxRT \
    MODEL_ROOT=/workspace/FluxRT

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
        git git-lfs bash ffmpeg \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && git lfs install \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && /usr/bin/python3.12 --version \
    && /usr/local/bin/python --version

WORKDIR /workspace

ARG FLUXRT_REF=main
RUN git clone https://github.com/tensorforger/FluxRT.git /workspace/FluxRT \
    && cd /workspace/FluxRT \
    && git checkout ${FLUXRT_REF}

WORKDIR /workspace/FluxRT

RUN /usr/bin/python3.12 -m pip install --upgrade pip setuptools wheel \
    && /usr/bin/python3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && /usr/bin/python3.12 -m pip install -r requirements.txt \
    && /usr/bin/python3.12 -m pip install -e . \
    && /usr/bin/python3.12 -m pip install fastapi "uvicorn[standard]" websockets pillow numpy "huggingface_hub[hf_transfer]"

RUN /usr/bin/python3.12 - <<'PY'
import sys
print("python", sys.version)
import fastapi, uvicorn, numpy, PIL
print("basic imports ok")
import cv2
print("cv2 ok", cv2.__version__)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available_build_check", torch.cuda.is_available())
import fluxrt
print("fluxrt import ok")
PY

# ------------------------------------------------------------------------------
# Hugging Face model bake -- split into separate RUN layers so each upload to
# the RunPod registry is smaller and a transient I/O error only fails one layer.
#
# HF_TOKEN is passed as a build arg; it is consumed inside each RUN layer and
# is NOT persisted as an env var in any subsequent layer.
#
# ENABLE_INT8 defaults to false at BUILD time on purpose: baking the INT8 weights
# pushed us over RunPod's 30-minute build limit. server.py keeps its runtime
# ENABLE_INT8 behavior unchanged, so /warmup can lazy-fetch INT8 at runtime if
# the endpoint env var ENABLE_INT8=true is set.
# ------------------------------------------------------------------------------
ARG HF_TOKEN=""
ARG ENABLE_INT8=false
ARG FLUX_REPO_ID=black-forest-labs/FLUX.2-klein-4B
ARG RIFE_REPO_ID=TensorForger/RIFE-safetensors
ARG INT8_REPO_ID=aydin99/FLUX.2-klein-4B-int8

# Bake 1: RIFE (small, ~hundreds of MB) -- its own layer.
RUN HF_TOKEN="${HF_TOKEN}" \
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    RIFE_REPO_ID="${RIFE_REPO_ID}" \
    /usr/bin/python3.12 - <<'PY'
import os, time
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path("/workspace/FluxRT")
ROOT.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
if token in ("", "None"):
    token = None

repo_id = os.environ["RIFE_REPO_ID"]
target = ROOT / "RIFE-safetensors"
target.mkdir(parents=True, exist_ok=True)
print(f"[bake-rife] downloading {repo_id} -> {target}", flush=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=str(target),
    local_dir_use_symlinks=False,
    token=token,
    max_workers=8,
)
(target / ".downloaded").write_text(str(time.time()))
print(f"[bake-rife] done {repo_id}", flush=True)
PY

# Bake 2: FLUX.2-klein-4B (the big one, ~12-16 GB) -- its own layer.
RUN HF_TOKEN="${HF_TOKEN}" \
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    FLUX_REPO_ID="${FLUX_REPO_ID}" \
    /usr/bin/python3.12 - <<'PY'
import os, time
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path("/workspace/FluxRT")
ROOT.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
if token in ("", "None"):
    token = None

repo_id = os.environ["FLUX_REPO_ID"]
target = ROOT / "FLUX.2-klein-4B"
target.mkdir(parents=True, exist_ok=True)
print(f"[bake-flux] downloading {repo_id} -> {target}", flush=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=str(target),
    local_dir_use_symlinks=False,
    token=token,
    max_workers=8,
)
(target / ".downloaded").write_text(str(time.time()))
print(f"[bake-flux] done {repo_id}", flush=True)
PY

# Bake 3: INT8 -- intentionally skipped at build time (ENABLE_INT8 build-arg
# defaults to false). server.py /warmup can lazy-fetch this at runtime if
# the endpoint env var ENABLE_INT8=true is set. To re-enable baking,
# pass --build-arg ENABLE_INT8=true.
RUN HF_TOKEN="${HF_TOKEN}" \
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    ENABLE_INT8="${ENABLE_INT8}" \
    INT8_REPO_ID="${INT8_REPO_ID}" \
    /usr/bin/python3.12 - <<'PY'
import os, time
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path("/workspace/FluxRT")
ROOT.mkdir(parents=True, exist_ok=True)

if os.environ.get("ENABLE_INT8", "false").strip().lower() not in ("1", "true", "yes", "on"):
    print("[bake-int8] ENABLE_INT8 build-arg is false -- skipping INT8 bake (server.py will lazy-fetch at warmup if needed)", flush=True)
else:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    if token in ("", "None"):
        token = None
    repo_id = os.environ["INT8_REPO_ID"]
    target = ROOT / "FLUX.2-klein-4B-int8"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[bake-int8] downloading {repo_id} -> {target}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=token,
        max_workers=8,
    )
    (target / ".downloaded").write_text(str(time.time()))
    print(f"[bake-int8] done {repo_id}", flush=True)
PY

COPY server.py /app/server.py
COPY bootstrap.py /app/bootstrap.py

# Build-time smoke check: fail the build early if either app file has a syntax error.
RUN /usr/bin/python3.12 -m py_compile /app/server.py /app/bootstrap.py

EXPOSE 8765

WORKDIR /workspace/FluxRT

# Use bootstrap.py: it tries to import & run server.app under uvicorn, and if
# that fails it starts a minimal FastAPI fallback on the same port so the
# worker stays reachable and /debug/startup can surface the real traceback.
CMD ["/usr/bin/python3.12", "-u", "/app/bootstrap.py"]

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
    && git lfs install

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

# Bake Hugging Face model weights into the image so workers start lighter.
# HF_TOKEN is passed as a build arg; it is consumed inside the RUN layer
# and is NOT persisted as an env var in any subsequent layer.
ARG HF_TOKEN=""
ARG ENABLE_INT8=true
ARG FLUX_REPO_ID=black-forest-labs/FLUX.2-klein-4B
ARG RIFE_REPO_ID=TensorForger/RIFE-safetensors
ARG INT8_REPO_ID=aydin99/FLUX.2-klein-4B-int8

RUN HF_TOKEN="${HF_TOKEN}" \
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    ENABLE_INT8="${ENABLE_INT8}" \
    FLUX_REPO_ID="${FLUX_REPO_ID}" \
    RIFE_REPO_ID="${RIFE_REPO_ID}" \
    INT8_REPO_ID="${INT8_REPO_ID}" \
    /usr/bin/python3.12 - <<'PY'
import os, time
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path("/workspace/FluxRT")
ROOT.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
if token in ("", "None"):
    token = None

def fetch(repo_id, subdir):
    target = ROOT / subdir
    target.mkdir(parents=True, exist_ok=True)
    print(f"[bake] downloading {repo_id} -> {target}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        token=token,
        max_workers=8,
    )
    (target / ".downloaded").write_text(str(time.time()))
    print(f"[bake] done {repo_id}", flush=True)

fetch(os.environ["RIFE_REPO_ID"], "RIFE-safetensors")
fetch(os.environ["FLUX_REPO_ID"], "FLUX.2-klein-4B")

if os.environ.get("ENABLE_INT8", "true").strip().lower() in ("1", "true", "yes", "on"):
    fetch(os.environ["INT8_REPO_ID"], "FLUX.2-klein-4B-int8")
else:
    print("[bake] ENABLE_INT8 disabled, skipping int8 weights", flush=True)

print("[bake] all models present under", ROOT, flush=True)
PY

COPY server.py /app/server.py

EXPOSE 8765

WORKDIR /workspace/FluxRT

CMD ["/usr/bin/python3.12", "-u", "/app/server.py"]

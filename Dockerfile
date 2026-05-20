# Base: CUDA 12.4 runtime + cuDNN, Ubuntu 22.04 (downgraded from 12.8 to match RunPod host driver)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8765 \
    PORT_HEALTH=8765 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/workspace/.cache/huggingface \
    FLUXRT_MODELS_DIR=/workspace/models

# Bootstrap: add deadsnakes PPA (FluxRT requires Python >=3.12, Ubuntu 22.04 ships 3.10).
# Also install OpenCV / FluxRT runtime libs: libgl1 (libGL.so.1 for cv2), libglib2.0-0
# (for GIO used by cv2), libgomp1 (OpenMP runtime), ffmpeg (video codecs used by cv2.VideoCapture).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv git git-lfs bash \
        libgl1 libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && git lfs install

WORKDIR /workspace

# Clone FluxRT source (small repo, no model weights)
ARG FLUXRT_REF=main
RUN git clone https://github.com/tensorforger/FluxRT.git \
    && cd FluxRT && git checkout ${FLUXRT_REF}

WORKDIR /workspace/FluxRT

# PyTorch (CUDA 12.4) + FluxRT deps + server deps + huggingface_hub for runtime model fetch.
# Use absolute /usr/bin/python3.12 -m pip (avoid relying on a 'python' or 'pip' symlink).
RUN /usr/bin/python3.12 -m pip install --upgrade pip \
    && /usr/bin/python3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && /usr/bin/python3.12 -m pip install -r requirements.txt \
    && /usr/bin/python3.12 -m pip install -e . \
    && /usr/bin/python3.12 -m pip install fastapi "uvicorn[standard]" websockets pillow numpy \
        "huggingface_hub[hf_transfer]"

# Build-time smoke check: fail the build (instead of producing a crashing image) if any of
# the required imports are broken. cv2 in particular needs libgl1/libglib2.0-0/libgomp1.
RUN /usr/bin/python3.12 - <<'PY'
import sys
print("python", sys.version)
import fastapi, uvicorn, numpy, PIL
print("basic imports OK:", fastapi.__version__, uvicorn.__version__, numpy.__version__, PIL.__version__)
import cv2
print("cv2 import OK:", cv2.__version__)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
import fluxrt
print("fluxrt import OK")
PY

# Our app code lives at /workspace/app/server.py (separate from upstream FluxRT to avoid file conflicts).
RUN mkdir -p /workspace/app
COPY server.py /workspace/app/server.py
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 8765

# Diagnostic entrypoint: prints PATH, ls, command -v checks, then execs python3.12 -u server.py.
CMD ["bash", "/workspace/start.sh"]

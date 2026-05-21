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

# Clone FluxRT source only. Model weights are NOT baked into this image --
# they are downloaded at runtime via /warmup or /api/load (Owen-style HTTP-first
# pattern). This keeps the image small enough to actually start on RunPod
# serverless workers.
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
# Slim image: NO model baking.
# - No FLUX.2-klein-4B bake
# - No RIFE-safetensors bake
# - No INT8 bake
# - No build-time ARG HF_TOKEN
# - No snapshot_download at build time
#
# server.py already supports lazy model download at warmup via ensure_models().
# /warmup and the new /api/load endpoint trigger that lazy download at runtime.
# A RunPod Network Volume mounted at /runpod-volume (or HF_HOME) can be used
# later to persist weights across worker spin-ups.
# ------------------------------------------------------------------------------

COPY server.py /app/server.py
COPY bootstrap.py /app/bootstrap.py

# Build-time smoke check: fail the build early if either app file has a syntax error.
RUN /usr/bin/python3.12 -m py_compile /app/server.py /app/bootstrap.py

# Diagnostic: install netcat-openbsd for shell-health-server CMD below.
RUN apt-get update && apt-get install -y --no-install-recommends netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*

EXPOSE 8765

WORKDIR /workspace/FluxRT

# Diagnostic shell-health-server: prove RunPod can execute a basic shell CMD
# and answer /ping on PORT 8765 without Python, Uvicorn, FluxRT, Torch, or model
# code. Workers were exiting with code 127 even with a bash wrapper, so this
# pulls the entrypoint below the Python layer entirely.
CMD ["/bin/sh", "-c", "echo '[shell-health] started'; while true; do printf 'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{\"ok\":true}' | nc -l -p ${PORT:-8765} -q 1; done"]

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

COPY server.py /app/server.py

EXPOSE 8765

WORKDIR /workspace/FluxRT

CMD ["/usr/bin/python3.12", "-u", "/app/server.py"]

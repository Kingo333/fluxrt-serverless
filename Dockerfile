# Base: CUDA 12.8 runtime + cuDNN, Ubuntu 22.04 (matches FluxRT README requirements)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8765 \
    PORT_HEALTH=8765 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/workspace/.cache/huggingface \
    FLUXRT_MODELS_DIR=/workspace/models

# Bootstrap: add deadsnakes PPA (FluxRT requires Python >=3.12, Ubuntu 22.04 ships 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv git git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && git lfs install

WORKDIR /workspace

# Clone FluxRT source (small repo, no model weights)
ARG FLUXRT_REF=main
RUN git clone https://github.com/tensorforger/FluxRT.git \
    && cd FluxRT && git checkout ${FLUXRT_REF}

WORKDIR /workspace/FluxRT

# PyTorch (CUDA 12.8) + FluxRT deps + server deps + huggingface_hub for runtime model fetch
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt \
    && pip install -e . \
    && pip install fastapi "uvicorn[standard]" websockets pillow numpy \
        "huggingface_hub[hf_transfer]"

# NOTE: model weights (FLUX.2-klein-4B ~14 GB + RIFE ~2 GB) are NOT baked into the image.
# They are downloaded lazily at first worker boot by server.py warmup() using
# huggingface_hub.snapshot_download into /workspace/models (persists on the worker for
# the lifetime of the container). This keeps the build under the 30-minute RunPod limit.

COPY server.py /workspace/FluxRT/server.py

EXPOSE 8765

CMD ["python", "-u", "server.py"]

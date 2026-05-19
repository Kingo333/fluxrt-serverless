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
        python3.12 python3.12-dev python3.12-venv git git-lfs bash \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && git lfs install

WORKDIR /workspace

# Clone FluxRT source (small repo, no model weights)
ARG FLUXRT_REF=main
RUN git clone https://github.com/tensorforger/FluxRT.git \
    && cd FluxRT && git checkout ${FLUXRT_REF}

WORKDIR /workspace/FluxRT

# PyTorch (CUDA 12.8) + FluxRT deps + server deps + huggingface_hub for runtime model fetch.
# Use absolute /usr/bin/python3.12 -m pip (avoid relying on a 'python' or 'pip' symlink).
RUN /usr/bin/python3.12 -m pip install --upgrade pip \
    && /usr/bin/python3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    && /usr/bin/python3.12 -m pip install -r requirements.txt \
    && /usr/bin/python3.12 -m pip install -e . \
    && /usr/bin/python3.12 -m pip install fastapi "uvicorn[standard]" websockets pillow numpy \
        "huggingface_hub[hf_transfer]"

# Our app code lives at /workspace/app/server.py (separate from upstream FluxRT to avoid file conflicts).
RUN mkdir -p /workspace/app
COPY server.py /workspace/app/server.py
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 8765

# Diagnostic entrypoint: prints PATH, ls, command -v checks, then execs python3.12 -u server.py.
CMD ["bash", "/workspace/start.sh"]

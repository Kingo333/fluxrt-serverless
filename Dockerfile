# Base: CUDA 12.8 runtime + cuDNN, Ubuntu 22.04 (matches FluxRT README requirements)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8765 \
    PORT_HEALTH=8765 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs python3.12 python3.12-dev python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /workspace

# Clone FluxRT (replace FLUXRT_REF with a pinned SHA when ready)
ARG FLUXRT_REF=main
RUN git clone https://github.com/tensorforger/FluxRT.git \
 && cd FluxRT && git checkout ${FLUXRT_REF}

WORKDIR /workspace/FluxRT

# PyTorch (CUDA 12.8) + FluxRT deps + server deps
RUN pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
 && pip install -r requirements.txt \
 && pip install -e . \
 && pip install fastapi "uvicorn[standard]" websockets pillow numpy hf_transfer

# Bake model weights so cold-start is short (~16 GB total)
RUN git clone https://huggingface.co/TensorForger/RIFE-safetensors \
 && git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B

COPY server.py /workspace/FluxRT/server.py

EXPOSE 8765

CMD ["python", "-u", "server.py"]

# ============================================================
# ACE-Step 1.5 — RunPod Serverless
# ============================================================
#
# Image size: ~8GB (without model weights)
# Models are downloaded at worker cold-start time
#
# Deployment flow:
#   1. Push this repo to GitHub
#   2. GitHub Actions builds this Dockerfile, pushes to GHCR
#   3. RunPod creates template from the published image
# ============================================================

ARG CUDA_VERSION=12.8.1

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==================== System packages ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        git curl wget ca-certificates \
        libsndfile1 libsndfile1-dev \
        ffmpeg \
        libffi-dev libssl-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# ==================== Install uv (fast Python package manager) ====================
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_CONCURRENT_DOWNLOADS=32
ENV UV_HTTP_TIMEOUT=600
ENV UV_CONNECT_TIMEOUT=120

# ==================== Clone ACE-Step 1.5 ====================
WORKDIR /app
RUN git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git /app \
    && rm -rf /app/.git

# ==================== Install Python dependencies ====================
# Try uv sync first (if lock exists), fallback to pip
RUN uv sync --frozen --no-dev --python python3.11 || \
    (uv venv --python python3.11 && \
     uv pip install -e . --python python3.11)

# Verify torch
RUN /app/.venv/bin/python -c "import torch; print('[torch]', torch.__version__, 'cuda:', torch.version.cuda)"

# ==================== RunPod SDK + hf_transfer (fast HF downloads) ====================
RUN /app/.venv/bin/pip install --no-cache-dir \
        runpod \
        hf_transfer \
        'huggingface_hub[hf_transfer]' \
        boto3

# ==================== Copy handler.py ====================
COPY handler.py /app/handler.py

# ==================== Runtime environment ====================
ENV ACESTEP_PROJECT_ROOT=/app
ENV ACESTEP_MODEL_DIR=/app/models
ENV ACESTEP_CHECKPOINTS_DIR=/app/models
ENV HF_HOME=/app/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TOKENIZERS_PARALLELISM=false

ARG ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
ARG ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B

ENV ACESTEP_CONFIG_PATH=${ACESTEP_CONFIG_PATH}
ENV ACESTEP_LM_MODEL_PATH=${ACESTEP_LM_MODEL_PATH}
ENV ACESTEP_LM_BACKEND=vllm
ENV ACESTEP_DEVICE=cuda
ENV ACESTEP_AUDIO_FORMAT=mp3

# ==================== Entry point ====================
CMD ["/app/.venv/bin/python", "-u", "/app/handler.py"]

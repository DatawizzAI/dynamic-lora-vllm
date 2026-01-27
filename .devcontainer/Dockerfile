# Development Dockerfile - same base as production but with dev tools
# CUDA base image - vLLM will install its required torch version

FROM nvidia/cuda:12.4-cudnn-devel-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, build essentials and development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    vim \
    htop \
    tmux \
    sudo \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements and install dependencies (same as production)
COPY requirements.txt /tmp/requirements.txt

# Install vLLM first (this installs torch as dependency)
RUN pip install --no-cache-dir --break-system-packages vllm==0.12.0

# Install flash-attn (torch now available from vLLM)
RUN pip install --no-cache-dir --break-system-packages flash-attn==2.7.4.post1 --no-build-isolation

# Install flashinfer (same as production)
RUN pip install --no-cache-dir --break-system-packages flashinfer-python==0.5.3

# Install remaining production dependencies (torch already installed by vLLM)
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Install additional dev tools
RUN pip install --no-cache-dir --break-system-packages \
    ipython \
    jupyter \
    jupyterlab \
    pytest \
    black \
    flake8 \
    ipdb

# Set environment for development
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"
ENV HF_HOME="/workspace/.cache/huggingface"
ENV CACHE_DIR="/workspace/.cache/huggingface"

# Create cache directories
RUN mkdir -p /workspace/.cache/huggingface

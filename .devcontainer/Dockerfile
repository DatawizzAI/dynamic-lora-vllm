# Development Dockerfile - same base as production but with dev tools
# Uses NGC PyTorch 25.10 (torch 2.9.0) to match production environment

FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    vim \
    htop \
    tmux \
    sudo \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install dependencies (same as production)
COPY requirements.txt /tmp/requirements.txt

# Install flash-attn (same as production)
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Install flashinfer (same as production)
RUN pip install --no-cache-dir flashinfer-python==0.5.3

# Install production dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional dev tools
RUN pip install --no-cache-dir \
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

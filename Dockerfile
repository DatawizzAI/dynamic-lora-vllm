FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/miniconda3/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda3 && \
    rm miniconda.sh

# Create conda environment with Python 3.11
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
RUN conda create -n vllm python=3.11 -y

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in the conda environment with cleanup
RUN /opt/miniconda3/envs/vllm/bin/pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy && \
    rm -rf /opt/miniconda3/pkgs/* && \
    rm -rf /tmp/*

# Copy application code
COPY src/ ./src/
COPY *.md *.yml *.yaml ./

# Set default environment variables (can be overridden at runtime)
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
ENV CACHE_DIR="/app/.cache/huggingface"
ENV MAX_LORAS=10
ENV MAX_LORA_RANK=16
ENV MAX_CPU_LORAS=5
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Optional environment variables (empty by default)
ENV API_KEY=""
ENV HF_TOKEN=""

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /tmp/.cache/huggingface

# Note: Port is configurable via PORT environment variable (default: 8000)
# Use -p HOST_PORT:CONTAINER_PORT when running to map the correct port

# Run the server
CMD ["/opt/miniconda3/envs/vllm/bin/python", "src/server.py"]
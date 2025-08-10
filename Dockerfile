FROM nvidia/cuda:12.1-devel-ubuntu22.04

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
RUN conda create -n vllm python=3.11 -y
SHELL ["/opt/miniconda3/bin/conda", "run", "-n", "vllm", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY *.md *.yml *.yaml ./

# Set default environment variables
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
ENV CACHE_DIR="/tmp/.cache/huggingface"
ENV MAX_LORAS=10
ENV MAX_LORA_RANK=16
ENV MAX_CPU_LORAS=5
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create cache directory
RUN mkdir -p /tmp/.cache/huggingface

# Expose the port
EXPOSE 8000

# Run the server
CMD ["/opt/miniconda3/envs/vllm/bin/python", "src/server.py"]
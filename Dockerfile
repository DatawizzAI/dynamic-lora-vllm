FROM nvcr.io/nvidia/pytorch:24.10-py3

# Build arguments for optional model pre-download and HF authentication
ARG HF_TOKEN=""
ARG MODEL_ID=""

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with NumPy compatibility fix
RUN pip install --no-cache-dir "numpy==1.26.4"

# Install deps from requirements.txt (flash-attn is NOT in requirements.txt; we install it separately below).
RUN pip install --no-cache-dir -r requirements.txt

# Base 24.10 has PyTorch 2.5.x; use official flash-attn wheel built for torch2.5 to avoid undefined symbol at runtime.
# Python 3.10 → cp310. If base image is 3.11, use the cp311 wheel from the same release.
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Add flashinfer to disable warning
RUN pip install --no-cache-dir flashinfer-cubin

# Pre-download model if MODEL_ID is provided as build arg
RUN if [ -n "$MODEL_ID" ] && [ "$MODEL_ID" != "" ]; then \
        echo "Pre-downloading model: $MODEL_ID"; \
        mkdir -p /app/.cache/huggingface; \
        if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "" ]; then \
            export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; \
        fi; \
        python -c "import os; from huggingface_hub import snapshot_download; \
                   cache_dir = '/app/.cache/huggingface'; \
                   token = os.environ.get('HUGGING_FACE_HUB_TOKEN', None); \
                   model_id = '$MODEL_ID'; \
                   print(f'Downloading {model_id} to {cache_dir}'); \
                   snapshot_download(repo_id=model_id, \
                                   cache_dir=cache_dir, \
                                   token=token, \
                                   ignore_patterns=['*.bin'])" && \
        echo "Model pre-download completed"; \
    else \
        echo "No MODEL_ID build arg provided, skipping model pre-download"; \
    fi

# Copy application code
COPY src/ ./src/
COPY *.md *.yml *.yaml ./

# Set default environment variables (can be overridden at runtime)
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MODEL_ID=${MODEL_ID:-"meta-llama/Llama-3.2-1B-Instruct"}
ENV CACHE_DIR="/app/.cache/huggingface"
ENV HF_HOME="/app/.cache/huggingface"
ENV MAX_LORAS=10
ENV MAX_LORA_RANK=16
ENV MAX_CPU_LORAS=5
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Optional environment variables (empty by default)
ENV API_KEY=""
ENV HF_TOKEN=${HF_TOKEN:-""}

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /tmp/.cache/huggingface

# Note: Port is configurable via PORT environment variable (default: 8000)
# Use -p HOST_PORT:CONTAINER_PORT when running to map the correct port

# Run the server
CMD ["python", "src/server.py"]
# NGC PyTorch 25.10 = torch 2.9.0, matching pip-freeze-runpod-worker.txt
# See: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html
FROM nvcr.io/nvidia/pytorch:25.10-py3

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

# Install flash-attn separately with --no-build-isolation since it requires torch at build time
# and torch is already available in the base image
# Version pinned to match RunPod worker (pip-freeze-runpod-worker.txt)
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Add flashinfer for vLLM top-p & top-k sampling performance
# Version pinned to match RunPod worker
RUN pip install --no-cache-dir flashinfer-python==0.5.3

# Install remaining dependencies
# torch 2.9.0 is provided by base image (NGC 25.10), versions pinned to match RunPod worker
RUN pip install --no-cache-dir -r requirements.txt

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
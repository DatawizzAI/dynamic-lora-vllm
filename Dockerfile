FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with NumPy compatibility fix
RUN pip install --no-cache-dir "numpy<2.0,>=1.24.0" && \
    pip install --no-cache-dir -r requirements.txt

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
CMD ["python", "src/server.py"]
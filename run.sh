#!/bin/bash
set -e

# Dynamic LoRA vLLM Service Runner
echo "Dynamic LoRA vLLM Service"
echo "========================"

# Load environment variables from .env if it exists
if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  set -a  # automatically export all variables
  source .env
  set +a  # turn off automatic export
fi

# Set default values if not provided
export PORT=${PORT:-8000}
export PORT_HEALTH=${PORT_HEALTH:-8001}
export HOST=${HOST:-0.0.0.0}
export MODEL_ID=${MODEL_ID:-"meta-llama/Llama-3.2-1B-Instruct"}
export CACHE_DIR=${CACHE_DIR:-"$(pwd)/.cache/huggingface"}
export MAX_LORAS=${MAX_LORAS:-10}
export MAX_LORA_RANK=${MAX_LORA_RANK:-16}
export MAX_CPU_LORAS=${MAX_CPU_LORAS:-5}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

echo "Configuration:"
echo "  Model: $MODEL_ID"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Health Port: $PORT_HEALTH"
echo "  Cache Directory: $CACHE_DIR"
echo "  Max LoRAs: $MAX_LORAS"
echo "  Max LoRA Rank: $MAX_LORA_RANK"
echo "  Max CPU LoRAs: $MAX_CPU_LORAS"

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"

# Check API key configuration
if [ -z "$API_KEY" ]; then
    echo "Info: API_KEY not set. Server will run without authentication."
else
    echo "API key provided for server authentication."
fi

# Check if HF_TOKEN is provided (optional)
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Private models will not be accessible."
else
    echo "HuggingFace token provided for private model access."
fi

# Set PYTHONPATH to include src directory
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

echo ""
echo "Starting server..."
python src/server.py
# Dynamic LoRA vLLM Service

A vLLM-based LLM serving service that supports dynamically loading LoRA adapters from Hugging Face. It exposes an OpenAI compatible API for easy integration.

## Features

- **Dynamic LoRA Loading**: Automatically downloads and loads LoRA adapters from Hugging Face Hub on demand
- **OpenAI Compatible API**: Uses vLLM's built-in OpenAI compatible API server
- **Containerized Deployment**: Docker-based deployment for easy scaling and management
- **Flexible Configuration**: Environment variable based configuration
- **Caching**: Intelligent caching of downloaded models and adapters

## Quick Start

### Using Docker Compose

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration:
   ```bash
   # Optional: Set API key for server authentication (leave empty for no auth)
   API_KEY=your_secret_api_key_here
   
   # Optional: Set your Hugging Face token for private models
   HF_TOKEN=your_huggingface_token_here
   
   # Optional: Customize other settings
   MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
   PORT=8000
   ```

3. Start the service:
   ```bash
   docker-compose up -d
   ```

### Using Docker

```bash
docker build -t dynamic-lora-vllm .

docker run -d \
  -p 8000:8000 \
  -e API_KEY=your_secret_api_key \
  -e HF_TOKEN=your_token_here \
  -e MODEL_ID=meta-llama/Llama-3.2-1B-Instruct \
  -v ./cache:/tmp/.cache/huggingface \
  --gpus all \
  dynamic-lora-vllm
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port the server listens on |
| `HOST` | `0.0.0.0` | Host the server listens on |
| `MODEL_ID` | `meta-llama/Llama-3.2-1B-Instruct` | Base model to load |
| `API_KEY` | - | API key for server authentication (optional - leave empty for no auth) |
| `HF_TOKEN` | - | Hugging Face token for private model access (optional) |
| `CACHE_DIR` | `/tmp/.cache/huggingface` | Directory to cache models and adapters |
| `MAX_LORAS` | `10` | Maximum number of LoRA adapters to load |
| `MAX_LORA_RANK` | `16` | Maximum rank of LoRA adapters |
| `MAX_CPU_LORAS` | `5` | Maximum LoRA adapters on CPU |

## Usage

### Basic Request

Once the service is running, you can make OpenAI-compatible requests:

**Without Authentication (default):**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

**With Authentication (if API_KEY is set):**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_secret_api_key" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

### Using LoRA Adapters

To use a LoRA adapter, simply specify the adapter's Hugging Face model ID as the model:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "username/my-lora-adapter",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

The service will automatically:
1. Download the LoRA adapter from Hugging Face Hub
2. Load it into memory
3. Use it for inference
4. Cache it for future requests

### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_secret_api_key" \
  -d '{
    "model": "username/my-lora-adapter",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'
```

### Using with OpenAI Python Client

```python
from openai import OpenAI

# Without authentication
client = OpenAI(
    api_key="not-needed",  # Required by client but ignored by server
    base_url="http://localhost:8000/v1"
)

# With authentication
client = OpenAI(
    api_key="your_secret_api_key",
    base_url="http://localhost:8000/v1"
)

response = client.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    prompt="Hello, how are you?",
    max_tokens=100
)
```

## Development

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export API_KEY=your_secret_key  # Optional - for authentication
   export HF_TOKEN=your_token_here  # Optional - for private models
   export MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
   ```

3. Run the server:
   ```bash
   ./run.sh
   ```

   Or directly:
   ```bash
   export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
   python src/server.py
   ```

## Requirements

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- At least 8GB GPU memory (depends on model size)
- Sufficient disk space for model and adapter caching

## Architecture

The service consists of:

1. **HuggingFace LoRA Resolver** (`src/hf_lora_resolver.py`): Downloads and manages LoRA adapters from HF Hub
2. **Main Server** (`src/server.py`): Integrates the resolver with vLLM's OpenAI compatible API
3. **Docker Environment**: Provides consistent deployment with GPU support

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**: Reduce `MAX_LORAS` or use a smaller base model
2. **Download Failures**: Check your `HF_TOKEN` and network connectivity
3. **Adapter Loading Errors**: Ensure the LoRA adapter is compatible with the base model

### Logs

Check Docker logs for debugging:
```bash
docker-compose logs -f dynamic-lora-vllm
```
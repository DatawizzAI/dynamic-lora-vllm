# Dynamic LoRA vLLM Service - Claude Context

## Project Overview
This project implements a vLLM-based LLM serving service that supports dynamically loading LoRA adapters from Hugging Face. The service exposes an OpenAI compatible API and automatically downloads/caches LoRA adapters on demand.

## Key Architecture Components

### Core Files
- **`src/server.py`** - Main server implementation using vLLM's OpenAI compatible API
- **`src/hf_lora_resolver.py`** - Custom LoRA resolver that downloads adapters from HuggingFace Hub
- **`requirements.txt`** - Python dependencies including vLLM, transformers, huggingface-hub
- **`Dockerfile`** - Container setup with CUDA support and conda environment
- **`docker-compose.yml`** - Easy deployment configuration with GPU access

### Configuration Files
- **`.env.example`** - Environment variable template
- **`run.sh`** - Local development runner script
- **`test_client.py`** - API validation and testing client

### Development Container
- **`.devcontainer/`** - VS Code dev container configuration with CUDA support
  - `devcontainer.json` - Dev container settings, extensions, and port forwarding
  - `Dockerfile` - Development environment with Python, CUDA, and dev tools
  - `post-create.sh` - Post-creation setup script with Claude Code installation
- **`.vscode/`** - VS Code workspace configuration
  - `settings.json` - Python interpreter, formatting, and linting settings
  - `launch.json` - Debug configurations for server and test client
  - `tasks.json` - Common development tasks (start server, test, format, lint)

## Technical Implementation Details

### LoRA Resolution Process
1. When a request comes in with a model field containing a HuggingFace model ID
2. If it's not the base model, the `HuggingFaceLoRAResolver` is triggered
3. Downloads adapter from HF Hub to local cache directory
4. Creates `LoRARequest` object for vLLM to load
5. Subsequent requests use cached version

### Environment Variables
- `MODEL_ID`: Base model (default: "meta-llama/Llama-3.2-1B-Instruct")
- `API_KEY`: API key for server authentication (optional - leave empty for no auth)
- `HF_TOKEN`: HuggingFace token for private model access (optional)
- `CACHE_DIR`: Model/adapter cache location (default: ".cache/huggingface")
- `MAX_LORAS`: Maximum concurrent LoRA adapters (default: 10)
- `MAX_LORA_RANK`: Maximum LoRA rank (default: 16)
- `MAX_CPU_LORAS`: CPU LoRA limit (default: 5)
- `PORT`/`HOST`: Server configuration

### API Usage
The service maintains full OpenAI API compatibility:
- Use base model ID in `model` field for base model inference
- Use LoRA adapter HF model ID in `model` field for adapter inference
- Supports both `/v1/completions` and `/v1/chat/completions` endpoints

## Development Guidelines

### When Working on This Project
1. **Dev Container Setup**: Use VS Code dev container for consistent CUDA-enabled development environment
2. **Dependencies**: Always use the versions specified in `requirements.txt`
3. **Testing**: Use `test_client.py` to validate API functionality
4. **Local Development**: Use `./run.sh` with `.env` file for configuration
5. **Containerization**: Test with `docker-compose up` to ensure GPU access works
6. **LoRA Compatibility**: Ensure LoRA adapters are compatible with the base model architecture

### Common Commands

#### Dev Container Development
```bash
# Open in VS Code dev container (recommended)
code .
# Then: "Reopen in Container" or Ctrl+Shift+P > "Dev Containers: Reopen in Container"

# Inside dev container:
claude auth                 # Authenticate Claude Code (first time only)
./run.sh                    # Start server
python test_client.py       # Test API
jupyter lab --ip=0.0.0.0    # Start Jupyter Lab

# Claude Code usage:
claude                      # CLI mode
# Or in VS Code: Cmd+Shift+P > "Claude Code: Chat"
```

#### Local Development
```bash
# Local development (requires CUDA setup)
./run.sh

# Test API
python test_client.py --base-url http://localhost:8000

# Docker deployment
docker-compose up -d

# Check logs
docker-compose logs -f dynamic-lora-vllm
```

#### VS Code Tasks (Ctrl+Shift+P > "Tasks: Run Task")
- **Start vLLM Server** - Launch the development server
- **Test API** - Run API validation tests
- **Docker Compose Up/Down** - Container management
- **Format Code** - Apply Black formatting
- **Lint Code** - Run Flake8 linting

### Key Considerations
- **Dev Container Requirements**: 
  - VS Code with Dev Containers extension
  - Docker with NVIDIA Container Toolkit for GPU access
  - Sufficient disk space for container image (~8GB) and model cache
- **GPU Requirements**: Needs NVIDIA GPU with sufficient VRAM for base model + LoRA adapters
- **Memory Management**: Monitor `MAX_LORAS` setting to prevent OOM errors
- **Network**: HuggingFace Hub downloads require internet access
- **Security**: Never commit HF tokens to repository

### Extension Points
- Add new LoRA resolvers for different sources (S3, local filesystem, etc.)
- Implement model warming/preloading for frequently used adapters
- Add metrics and monitoring endpoints
- Implement adapter version management

## Testing Strategy
1. **Health Check**: Verify service startup and basic health endpoint
2. **Base Model**: Test completions and chat with base model
3. **LoRA Loading**: Test with actual LoRA adapter from HuggingFace
4. **Caching**: Verify subsequent requests use cached adapters
5. **Error Handling**: Test with invalid model IDs and network issues

## Troubleshooting
- **GPU Memory Issues**: Reduce `MAX_LORAS` or use smaller base model
- **Download Failures**: Check `HF_TOKEN` and network connectivity
- **Adapter Loading Errors**: Verify LoRA compatibility with base model
- **Container Issues**: Ensure NVIDIA Container Toolkit is installed

## Future Enhancements
- Support for multiple base models simultaneously
- Advanced caching strategies (LRU, size-based eviction)
- Authentication and rate limiting
- Integration with model registries beyond HuggingFace
- Streaming responses for better user experience

## Important: Server Startup

**ALWAYS use `./run.sh` to start the server**, not `python src/server.py` directly. The run script:
- Loads environment variables from `.env` file
- Sets proper defaults for all configuration
- Creates necessary cache directories
- Sets PYTHONPATH correctly
- Provides better error handling and logging

Direct python execution bypasses important initialization steps and environment setup.
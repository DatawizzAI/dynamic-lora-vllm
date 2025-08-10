import os
import asyncio
import uvicorn
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.entrypoints.openai.api_server import app, init_app_state
from vllm.lora.resolver import LoRAResolverRegistry
from hf_lora_resolver import HuggingFaceLoRAResolver


def get_env_var(name: str, default_value: str = None, var_type: type = str):
    """Get environment variable with type conversion and default value."""
    value = os.getenv(name, default_value)
    if value is None:
        return None
    if var_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    return var_type(value)


async def create_engine():
    """Create and configure the vLLM async engine with LoRA support."""
    
    # Get configuration from environment variables
    model_id = get_env_var("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
    max_loras = get_env_var("MAX_LORAS", "10", int)
    max_lora_rank = get_env_var("MAX_LORA_RANK", "16", int)
    max_cpu_loras = get_env_var("MAX_CPU_LORAS", "5", int)
    cache_dir = get_env_var("CACHE_DIR", "/tmp/.cache/huggingface")
    hf_token = get_env_var("HF_TOKEN")
    
    print(f"Initializing vLLM engine with model: {model_id}")
    print(f"LoRA configuration: max_loras={max_loras}, max_lora_rank={max_lora_rank}, max_cpu_loras={max_cpu_loras}")
    
    # Configure engine arguments with LoRA support
    engine_args = AsyncEngineArgs(
        model=model_id,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        max_cpu_loras=max_cpu_loras,
        trust_remote_code=True,
        # Add GPU memory utilization to prevent OOM
        gpu_memory_utilization=0.8,
    )
    
    # Create the async engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Register HuggingFace LoRA resolver
    hf_resolver = HuggingFaceLoRAResolver(cache_dir=cache_dir, hf_token=hf_token)
    LoRAResolverRegistry.register_resolver("hf_resolver", hf_resolver)
    
    print("Registered HuggingFace LoRA resolver")
    
    return engine


async def run_server():
    """Initialize and run the vLLM OpenAI compatible API server."""
    
    # Get server configuration from environment variables
    host = get_env_var("HOST", "0.0.0.0")
    port = get_env_var("PORT", "8000", int)
    
    print("Creating vLLM engine...")
    engine = await create_engine()
    
    print("Initializing OpenAI compatible API server...")
    # Initialize the FastAPI app state with the engine
    await init_app_state(engine)
    
    print(f"Starting server on {host}:{port}")
    
    # Configure and start the server
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
    
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """Main entry point."""
    print("Dynamic LoRA vLLM Server")
    print("========================")
    
    # Enable runtime LoRA updating
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        raise


if __name__ == "__main__":
    main()
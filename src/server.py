import os
import sys
import uvloop
from vllm.entrypoints.openai.api_server import run_server, cli_env_setup
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser
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


def register_custom_lora_resolver():
    """Register our custom HuggingFace LoRA resolver."""
    cache_dir = get_env_var("CACHE_DIR", ".cache/huggingface")
    hf_token = get_env_var("HF_TOKEN")
    
    # Register HuggingFace LoRA resolver
    hf_resolver = HuggingFaceLoRAResolver(cache_dir=cache_dir, hf_token=hf_token)
    LoRAResolverRegistry.register_resolver("hf_resolver", hf_resolver)
    print("Registered HuggingFace LoRA resolver")


def main():
    """Main entry point - Use the exact same pattern as vLLM's official server."""
    print("Dynamic LoRA vLLM Server")
    print("========================")
    
    # Enable runtime LoRA updating
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    
    # Register our custom LoRA resolver before starting the server
    register_custom_lora_resolver()
    
    # Use the exact same pattern as vLLM's official server
    # NOTE: This section is in sync with vllm/entrypoints/openai/api_server.py
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    
    # Build CLI args from environment variables
    model_id = get_env_var("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
    host = get_env_var("HOST", "0.0.0.0")
    port = get_env_var("PORT", "8000", int)
    max_loras = get_env_var("MAX_LORAS", "10", int)
    max_lora_rank = get_env_var("MAX_LORA_RANK", "16", int)
    max_cpu_loras = get_env_var("MAX_CPU_LORAS", "5", int)
    api_key = get_env_var("API_KEY")  # Separate API key for server authentication
    hf_token = get_env_var("HF_TOKEN")  # HuggingFace token for model access
    
    # Create CLI args list
    cli_args = [
        "--model", model_id,
        "--host", host,
        "--port", str(port),
        "--enable-lora",
        "--max-loras", str(max_loras),
        "--max-lora-rank", str(max_lora_rank),
        "--max-cpu-loras", str(max_cpu_loras),
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.8"
    ]
    
    if api_key:
        cli_args.extend(["--api-key", api_key])
    
    print(f"Starting vLLM server with model: {model_id}")
    print(f"LoRA configuration: max_loras={max_loras}, max_lora_rank={max_lora_rank}, max_cpu_loras={max_cpu_loras}")
    
    # Parse arguments using vLLM's official parser
    args = parser.parse_args(cli_args)
    validate_parsed_serve_args(args)
    
    # Run the server using vLLM's official run_server function
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
"""Dynamic LoRA vLLM Server - Main entry point."""

import asyncio
import json
import threading

import uvloop
from vllm.entrypoints.openai.api_server import run_server, cli_env_setup
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.engine.arg_utils import FlexibleArgumentParser
from vllm.lora.resolver import LoRAResolverRegistry

from config import ServerConfig
from health import HealthServer
from tool_parsers import infer_tool_call_parser
from hf_lora_resolver import HuggingFaceLoRAResolver


# Global health server instance
health_server = HealthServer()


def register_custom_lora_resolver(config: ServerConfig):
    """Register our custom HuggingFace LoRA resolver."""
    hf_resolver = HuggingFaceLoRAResolver(
        cache_dir=config.cache_dir, 
        hf_token=config.hf_token, 
        copy_chat_template=config.copy_chat_template
    )
    LoRAResolverRegistry.register_resolver("hf_resolver", hf_resolver)
    print(f"Registered HuggingFace LoRA resolver (copy_chat_template={config.copy_chat_template})")


def build_cli_args(config: ServerConfig) -> list:
    """Build CLI arguments for vLLM server."""
    cli_args = [
        "--model", config.model_id,
        "--host", config.host,
        "--port", str(config.port),
        "--enable-lora",
        "--max-loras", str(config.max_loras),
        "--max-lora-rank", str(config.max_lora_rank),
        "--max-cpu-loras", str(config.max_cpu_loras),
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.8",
        "--download-dir", config.cache_dir
    ]
    
    if config.api_key:
        cli_args.extend(["--api-key", config.api_key])
    
    # Add multimodal limits if configured
    mm_limits = {}
    if config.max_images_per_prompt > 0:
        mm_limits["image"] = config.max_images_per_prompt
    if config.max_videos_per_prompt > 0:
        mm_limits["video"] = config.max_videos_per_prompt
    if config.max_audios_per_prompt > 0:
        mm_limits["audio"] = config.max_audios_per_prompt
    
    if mm_limits:
        cli_args.extend(["--limit-mm-per-prompt", json.dumps(mm_limits)])
    
    # Add auto tool choice configuration
    if config.enable_auto_tool_choice:
        cli_args.append("--enable-auto-tool-choice")
        
        parser_to_use = config.tool_call_parser or infer_tool_call_parser(config.model_id)
        
        if parser_to_use:
            cli_args.extend(["--tool-call-parser", parser_to_use])
            print(f"Auto tool choice enabled with parser: {parser_to_use}")
        else:
            print("Auto tool choice enabled but no compatible parser found for model")
    elif config.tool_call_parser:
        cli_args.extend(["--tool-call-parser", config.tool_call_parser])
        print(f"Tool call parser set: {config.tool_call_parser}")
    
    return cli_args


async def run_vllm_server_async(args, config: ServerConfig):
    """Async wrapper for vLLM server with health monitoring."""
    try:
        print("Starting vLLM server...")
        
        # Start health monitoring in background
        asyncio.create_task(health_server.monitor_vllm_health(config.host, config.port))
        
        # Start the vLLM server
        await run_server(args)
        
    except Exception as e:
        print(f"vLLM server error: {e}")
        health_server.set_error()
        raise


def print_startup_info(config: ServerConfig):
    """Print server startup information."""
    print("Dynamic LoRA vLLM Server")
    print("========================")
    
    if config.is_model_cached():
        print(f"Using pre-downloaded model from cache: {config.get_model_cache_path()}")
    else:
        print(f"Model will be downloaded at runtime to: {config.cache_dir}")
    
    print(f"Starting health server on {config.host}:{config.health_port}")
    print(f"Starting vLLM server on {config.host}:{config.port} with model: {config.model_id}")
    print(f"LoRA configuration: max_loras={config.max_loras}, max_lora_rank={config.max_lora_rank}, max_cpu_loras={config.max_cpu_loras}")
    
    # Print multimodal configuration
    if config.max_images_per_prompt > 0 or config.max_videos_per_prompt > 0 or config.max_audios_per_prompt > 0:
        print(f"Multimodal configuration:")
        print(f"  - Image: fetch_timeout={config.image_fetch_timeout}s, max_per_prompt={config.max_images_per_prompt}")
        print(f"  - Video: fetch_timeout={config.video_fetch_timeout}s, max_per_prompt={config.max_videos_per_prompt}")
        print(f"  - Audio: fetch_timeout={config.audio_fetch_timeout}s, max_per_prompt={config.max_audios_per_prompt}")


def main():
    """Main entry point."""
    # Load configuration from environment
    config = ServerConfig.from_env()
    config.setup_environment()
    
    # Print startup info
    print_startup_info(config)
    
    # Start health server in a separate thread
    health_thread = threading.Thread(
        target=health_server.run,
        args=(config.host, config.health_port),
        daemon=True
    )
    health_thread.start()
    
    try:
        # Register custom LoRA resolver
        register_custom_lora_resolver(config)
        
        # Set up vLLM CLI environment
        cli_env_setup()
        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        
        # Build and parse arguments
        cli_args = build_cli_args(config)
        args = parser.parse_args(cli_args)
        validate_parsed_serve_args(args)
        
        # Run the server
        uvloop.run(run_vllm_server_async(args, config))
        
    except Exception as e:
        print(f"Server startup error: {e}")
        health_server.set_error()
        raise


if __name__ == "__main__":
    main()

import os
import asyncio
import threading
import uvloop
from enum import Enum
from fastapi import FastAPI
from fastapi.responses import Response
import uvicorn
from vllm.entrypoints.openai.api_server import run_server, cli_env_setup
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser
from vllm.lora.resolver import LoRAResolverRegistry
from hf_lora_resolver import HuggingFaceLoRAResolver


class ServerState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


# Global server state
server_state = ServerState.INITIALIZING


def get_env_var(name: str, default_value: str = None, var_type: type = str):
    """Get environment variable with type conversion and default value."""
    value = os.getenv(name, default_value)
    if value is None:
        return None
    if var_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    return var_type(value)


def create_health_app():
    """Create FastAPI app for health checks."""
    app = FastAPI()
    
    @app.get("/ping")
    async def ping():
        if server_state is ServerState.INITIALIZING:
            return Response(status_code=204)
        elif server_state is ServerState.READY:
            return Response(status_code=200)
        else:  # ERROR state
            return Response(status_code=500)
    
    return app


def run_health_server(host: str, port: int):
    """Run the health check server."""
    app = create_health_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


def register_custom_lora_resolver():
    """Register our custom HuggingFace LoRA resolver."""
    cache_dir = get_env_var("CACHE_DIR", ".cache/huggingface")
    hf_token = get_env_var("HF_TOKEN")
    copy_chat_template = get_env_var("COPY_CHAT_TEMPLATE", "true", bool)
    
    # Register HuggingFace LoRA resolver
    hf_resolver = HuggingFaceLoRAResolver(
        cache_dir=cache_dir, 
        hf_token=hf_token, 
        copy_chat_template=copy_chat_template
    )
    LoRAResolverRegistry.register_resolver("hf_resolver", hf_resolver)
    print(f"Registered HuggingFace LoRA resolver (copy_chat_template={copy_chat_template})")


async def monitor_vllm_health(host: str, port: int):
    """Monitor vLLM server health and update global state."""
    global server_state
    import aiohttp
    
    url = f"http://{host}:{port}/health"
    
    # Wait for server to be reachable
    max_attempts = 60  # 5 minutes
    for _ in range(max_attempts):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        server_state = ServerState.READY
                        print("vLLM server is ready!")
                        return
        except Exception:
            pass
        
        await asyncio.sleep(5)
    
    # If we get here, server didn't become ready in time
    print("vLLM server failed to become ready within timeout")
    server_state = ServerState.ERROR


async def run_vllm_server_async(args):
    """Async wrapper for vLLM server with state management."""
    global server_state
    try:
        print("Starting vLLM server...")
        
        # Start health monitoring in background
        asyncio.create_task(monitor_vllm_health(args.host, args.port))
        
        # Start the vLLM server
        await run_server(args)
        
    except Exception as e:
        print(f"vLLM server error: {e}")
        server_state = ServerState.ERROR
        raise


def infer_tool_call_parser(model_id: str) -> str:
    """
    Infer the appropriate tool call parser based on model ID.
    Returns the parser name or None if no suitable parser is found.
    """
    model_lower = model_id.lower()
    
    # Hermes models
    if "nousresearch/hermes-" in model_lower:
        return "hermes"
    
    # Mistral models  
    if "mistralai/mistral-" in model_lower:
        return "mistral"
    
    # Llama models
    if "meta-llama/llama-" in model_lower:
        if "llama-4" in model_lower:
            return "llama4_pythonic"
        elif "llama-3.1" in model_lower or "llama-3.2" in model_lower:
            return "llama3_json"
    
    # IBM Granite models
    if "ibm-granite/granite-" in model_lower:
        if "granite-20b-functioncalling" in model_lower:
            return "granite-20b-fc"
        return "granite"
    
    # InternLM models
    if "internlm/internlm2_5-" in model_lower or "internlm/internlm2.5-" in model_lower:
        return "internlm"
    
    # Jamba models
    if "ai21labs/ai21-jamba-" in model_lower:
        return "jamba"
    
    # xLAM models
    if ("salesforce/llama-xlam-" in model_lower or 
        "salesforce/xlam-" in model_lower or 
        "salesforce/qwen-xlam-" in model_lower):
        return "xlam"
    
    # Qwen models (use hermes parser)
    if ("qwen/qwen2.5-" in model_lower or 
        "qwen/qwq-" in model_lower):
        return "hermes"
    
    # MiniMax models
    if "minimaxai/minimax-m1-" in model_lower:
        return "minimax_m1"
    
    # DeepSeek-V3 models
    if ("deepseek-ai/deepseek-v3-" in model_lower or 
        "deepseek-ai/deepseek-r1-" in model_lower):
        return "deepseek_v3"
    
    # Kimi-K2 models
    if "moonshotai/kimi-k2-" in model_lower:
        return "kimi_k2"
    
    # Hunyuan models
    if "tencent/hunyuan-a13b-" in model_lower:
        return "hunyuan_a13b"
    
    # Return None if no parser is found
    return None


def main():
    """Main entry point - Use the exact same pattern as vLLM's official server."""
    global server_state
    print("Dynamic LoRA vLLM Server")
    print("========================")
    
    # Enable runtime LoRA updating
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    
    # Get configuration
    model_id = get_env_var("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
    host = get_env_var("HOST", "0.0.0.0")
    port = get_env_var("PORT", "8000", int)
    health_port = get_env_var("PORT_HEALTH", "8001", int)
    max_loras = get_env_var("MAX_LORAS", "10", int)
    max_lora_rank = get_env_var("MAX_LORA_RANK", "16", int)
    max_cpu_loras = get_env_var("MAX_CPU_LORAS", "5", int)
    api_key = get_env_var("API_KEY")
    cache_dir = get_env_var("CACHE_DIR", ".cache/huggingface")
    enable_auto_tool_choice = get_env_var("ENABLE_AUTO_TOOL_CHOICE", "true", bool)
    tool_call_parser = get_env_var("TOOL_CALL_PARSER")
    
    # Multimodal configuration
    image_fetch_timeout = get_env_var("IMAGE_FETCH_TIMEOUT", "5", int)
    video_fetch_timeout = get_env_var("VIDEO_FETCH_TIMEOUT", "30", int)
    audio_fetch_timeout = get_env_var("AUDIO_FETCH_TIMEOUT", "10", int)
    max_images_per_prompt = get_env_var("MAX_IMAGES_PER_PROMPT", "4", int)
    max_videos_per_prompt = get_env_var("MAX_VIDEOS_PER_PROMPT", "1", int)
    max_audios_per_prompt = get_env_var("MAX_AUDIOS_PER_PROMPT", "1", int)
    
    # Set multimodal environment variables for vLLM
    if image_fetch_timeout:
        os.environ["VLLM_IMAGE_FETCH_TIMEOUT"] = str(image_fetch_timeout)
    if video_fetch_timeout:
        os.environ["VLLM_VIDEO_FETCH_TIMEOUT"] = str(video_fetch_timeout)
    if audio_fetch_timeout:
        os.environ["VLLM_AUDIO_FETCH_TIMEOUT"] = str(audio_fetch_timeout)
    
    # Ensure HuggingFace uses the correct cache directory
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    # Check if model is pre-downloaded
    model_cache_path = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
    if os.path.exists(model_cache_path):
        print(f"Using pre-downloaded model from cache: {model_cache_path}")
    else:
        print(f"Model will be downloaded at runtime to: {cache_dir}")
    
    print(f"Starting health server on {host}:{health_port}")
    print(f"Starting vLLM server on {host}:{port} with model: {model_id}")
    print(f"LoRA configuration: max_loras={max_loras}, max_lora_rank={max_lora_rank}, max_cpu_loras={max_cpu_loras}")
    
    # Print multimodal configuration if any limits are set
    if max_images_per_prompt > 0 or max_videos_per_prompt > 0 or max_audios_per_prompt > 0:
        print(f"Multimodal configuration:")
        print(f"  - Image: fetch_timeout={image_fetch_timeout}s, max_per_prompt={max_images_per_prompt}")
        print(f"  - Video: fetch_timeout={video_fetch_timeout}s, max_per_prompt={max_videos_per_prompt}")
        print(f"  - Audio: fetch_timeout={audio_fetch_timeout}s, max_per_prompt={max_audios_per_prompt}")
    
    # Start health server in a separate thread
    health_thread = threading.Thread(
        target=run_health_server,
        args=(host, health_port),
        daemon=True
    )
    health_thread.start()
    
    try:
        # Register our custom LoRA resolver before starting the server
        register_custom_lora_resolver()
        
        # Use the exact same pattern as vLLM's official server
        cli_env_setup()
        parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        
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
            "--gpu-memory-utilization", "0.8",
            "--download-dir", cache_dir
        ]
        
        if api_key:
            cli_args.extend(["--api-key", api_key])
        
        # Add multimodal limits if configured
        mm_limits = {}
        if max_images_per_prompt > 0:
            mm_limits["image"] = max_images_per_prompt
        if max_videos_per_prompt > 0:
            mm_limits["video"] = max_videos_per_prompt
        if max_audios_per_prompt > 0:
            mm_limits["audio"] = max_audios_per_prompt
        
        if mm_limits:
            import json
            cli_args.extend(["--limit-mm-per-prompt", json.dumps(mm_limits)])
        
        # Add auto tool choice configuration
        if enable_auto_tool_choice:
            cli_args.append("--enable-auto-tool-choice")
            
            # Determine tool call parser
            parser_to_use = tool_call_parser
            if not parser_to_use:
                parser_to_use = infer_tool_call_parser(model_id)
            
            if parser_to_use:
                cli_args.extend(["--tool-call-parser", parser_to_use])
                print(f"Auto tool choice enabled with parser: {parser_to_use}")
            else:
                print("Auto tool choice enabled but no compatible parser found for model")
        elif tool_call_parser:
            # If auto tool choice is disabled but parser is explicitly set, still use it
            cli_args.extend(["--tool-call-parser", tool_call_parser])
            print(f"Tool call parser set: {tool_call_parser}")
        
        # Parse arguments using vLLM's official parser
        args = parser.parse_args(cli_args)
        validate_parsed_serve_args(args)
        
        # Run the server using our async wrapper for proper state management
        uvloop.run(run_vllm_server_async(args))
        
    except Exception as e:
        print(f"Server startup error: {e}")
        server_state = ServerState.ERROR
        raise


if __name__ == "__main__":
    main()
import json
import os
import asyncio
import threading
import uvloop
import uvicorn
from vllm.entrypoints.openai.api_server import run_server, cli_env_setup
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.engine.arg_utils import FlexibleArgumentParser
from vllm.lora.resolver import LoRAResolverRegistry
from hf_lora_resolver import HuggingFaceLoRAResolver
from model_config import get_override_model_config
from utils import ServerState, create_health_app, get_env_var, infer_tool_call_parser


server_state = ServerState.INITIALIZING


def run_health_server(host: str, port: int):
    app = create_health_app(get_state=lambda: server_state)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


def register_custom_lora_resolver():
    cache_dir = get_env_var("CACHE_DIR", ".cache/huggingface")
    hf_token = get_env_var("HF_TOKEN")
    copy_chat_template = get_env_var("COPY_CHAT_TEMPLATE", "true", bool)
    hf_resolver = HuggingFaceLoRAResolver(
        cache_dir=cache_dir,
        hf_token=hf_token,
        copy_chat_template=copy_chat_template,
    )
    LoRAResolverRegistry.register_resolver("hf_resolver", hf_resolver)
    print(f"Registered HuggingFace LoRA resolver (copy_chat_template={copy_chat_template})")


async def monitor_vllm_health(host: str, port: int):
    global server_state  # Must be at the top of the function
    import aiohttp
    url = f"http://{host}:{port}/health"
    max_attempts = 60
    for _ in range(max_attempts):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        server_state = ServerState.READY
                        print("vLLM server is ready!")
                        return
        except Exception:
            pass
        await asyncio.sleep(5)
    server_state = ServerState.ERROR
    print("vLLM server failed to become ready within timeout")


async def run_vllm_server_async(args):
    global server_state
    try:
        print("Starting vLLM server...")
        asyncio.create_task(monitor_vllm_health(args.host, args.port))
        await run_server(args)
    except Exception as e:
        print(f"vLLM server error: {e}")
        server_state = ServerState.ERROR
        raise


def main():
    global server_state
    print("Dynamic LoRA vLLM Server")
    print("========================")
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

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

    image_fetch_timeout = get_env_var("IMAGE_FETCH_TIMEOUT", "5", int)
    video_fetch_timeout = get_env_var("VIDEO_FETCH_TIMEOUT", "30", int)
    audio_fetch_timeout = get_env_var("AUDIO_FETCH_TIMEOUT", "10", int)
    max_images_per_prompt = get_env_var("MAX_IMAGES_PER_PROMPT", "4", int)
    max_videos_per_prompt = get_env_var("MAX_VIDEOS_PER_PROMPT", "1", int)
    max_audios_per_prompt = get_env_var("MAX_AUDIOS_PER_PROMPT", "1", int)

    if image_fetch_timeout:
        os.environ["VLLM_IMAGE_FETCH_TIMEOUT"] = str(image_fetch_timeout)
    if video_fetch_timeout:
        os.environ["VLLM_VIDEO_FETCH_TIMEOUT"] = str(video_fetch_timeout)
    if audio_fetch_timeout:
        os.environ["VLLM_AUDIO_FETCH_TIMEOUT"] = str(audio_fetch_timeout)
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    model_cache_path = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
    if os.path.exists(model_cache_path):
        print(f"Using pre-downloaded model from cache: {model_cache_path}")
    else:
        print(f"Model will be downloaded at runtime to: {cache_dir}")
    override_config = get_override_model_config(model_id)
    print(f"Starting health server on {host}:{health_port}")
    print(f"Starting vLLM server on {host}:{port} with model: {model_id}")
    if override_config:
        runner = override_config.get("runner")
        print(f"Override config: runner={runner}, enable_lora={override_config.get('enable_lora', True)}, enable_tool_choice={override_config.get('enable_tool_choice', True)}")
    else:
        print(f"LoRA configuration: max_loras={max_loras}, max_lora_rank={max_lora_rank}, max_cpu_loras={max_cpu_loras}")

    # Print multimodal configuration if any limits are set
    if max_images_per_prompt > 0 or max_videos_per_prompt > 0 or max_audios_per_prompt > 0:
        print("Multimodal configuration:")
        print(f"  Image: fetch_timeout={image_fetch_timeout}s, max_per_prompt={max_images_per_prompt}")
        print(f"  Video: fetch_timeout={video_fetch_timeout}s, max_per_prompt={max_videos_per_prompt}")
        print(f"  Audio: fetch_timeout={audio_fetch_timeout}s, max_per_prompt={max_audios_per_prompt}")

    health_thread = threading.Thread(target=run_health_server, args=(host, health_port), daemon=True)
    health_thread.start()

    try:
        register_custom_lora_resolver()
        cli_env_setup()
        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)

        # Build CLI args from override model config (config-driven; no task-specific branches)
        override_config = get_override_model_config(model_id)
        # Check if we should use eager mode (disables torch.compile for driver compatibility)
        use_eager = get_env_var("ENFORCE_EAGER", "false", bool)

        cli_args = [
            "--model", model_id,
            "--host", host,
            "--port", str(port),
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.8",
            "--download-dir", cache_dir,
        ]

        # Enable eager mode if requested or if using pooling runner (driver compatibility)
        if use_eager or (override_config and override_config.get("runner") == "pooling"):
            cli_args.append("--enforce-eager")
            print("Eager mode enabled (torch.compile disabled for driver compatibility)")
        if override_config and override_config.get("runner"):
            cli_args.extend(["--runner", override_config["runner"]])
        if override_config and override_config.get("hf_overrides"):
            cli_args.extend(["--hf-overrides", json.dumps(override_config["hf_overrides"])])
        use_lora = override_config is None or override_config.get("enable_lora", True)
        if use_lora:
            cli_args.extend([
                "--enable-lora",
                "--max-loras", str(max_loras),
                "--max-lora-rank", str(max_lora_rank),
                "--max-cpu-loras", str(max_cpu_loras),
            ])
        if override_config:
            print(f"Override config applied: runner={override_config.get('runner')}, enable_lora={use_lora}")

        if api_key:
            cli_args.extend(["--api-key", api_key])

        # Log rerank model input (training-level prompt) to server stdout when requested
        # vLLM's RequestLogger.log_inputs() logs the full_prompt at DEBUG; enable that path
        log_rerank_prompt = get_env_var("RERANK_LOG_PROMPT", "0", int) or get_env_var("VLLM_LOG_PROMPT", "0", int)
        if log_rerank_prompt:
            os.environ["VLLM_LOGGING_LEVEL"] = os.environ.get("VLLM_LOGGING_LEVEL", "DEBUG")
            cli_args.append("--enable-log-requests")
            cli_args.extend(["--max-log-len", "100000"])
            print("Rerank prompt logging enabled: full model input will appear in server log at DEBUG for each /v1/rerank request")

        # Add multimodal limits if configured
        mm_limits = {}
        if max_images_per_prompt > 0:
            mm_limits["image"] = max_images_per_prompt
        if max_videos_per_prompt > 0:
            mm_limits["video"] = max_videos_per_prompt
        if max_audios_per_prompt > 0:
            mm_limits["audio"] = max_audios_per_prompt
        if mm_limits:
            cli_args.extend(["--limit-mm-per-prompt", json.dumps(mm_limits)])
        # Add auto tool choice from override config (default True when no override)
        # vLLM requires --tool-call-parser when --enable-auto-tool-choice is set; only enable when we have a parser
        use_tool_choice = override_config is None or override_config.get("enable_tool_choice", True)
        if use_tool_choice:
            parser_to_use = tool_call_parser or infer_tool_call_parser(model_id)
            if enable_auto_tool_choice and parser_to_use:
                cli_args.append("--enable-auto-tool-choice")
                cli_args.extend(["--tool-call-parser", parser_to_use])
                print(f"Auto tool choice enabled with parser: {parser_to_use}")
            elif tool_call_parser:
                cli_args.extend(["--tool-call-parser", tool_call_parser])
                print(f"Tool call parser set: {tool_call_parser}")
            elif enable_auto_tool_choice and not parser_to_use:
                print("Auto tool choice skipped: no compatible parser for model (e.g. reranker)")

        # Parse arguments using vLLM's official parser
        args = parser.parse_args(cli_args)
        validate_parsed_serve_args(args)
        uvloop.run(run_vllm_server_async(args))
    except Exception as e:
        print(f"Server startup error: {e}")
        server_state = ServerState.ERROR
        raise


if __name__ == "__main__":
    main()

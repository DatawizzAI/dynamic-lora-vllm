"""Single utils module: env, health check, tool call parser. No vLLM dependency."""
from __future__ import annotations

import os
from enum import Enum

from fastapi import FastAPI
from fastapi.responses import Response


# --- env ---
def get_env_var(name: str, default_value: str = None, var_type: type = str):
    """Get env var with type conversion. bool: 'true'/'1'/'yes'/'on' -> True. Empty string uses default."""
    value = os.getenv(name, default_value)
    if value is None or value == "":
        if default_value is None:
            return None
        if var_type == bool:
            return default_value.lower() in ("true", "1", "yes", "on")
        try:
            return var_type(default_value)
        except (ValueError, TypeError):
            return default_value
    if var_type == bool:
        return value.lower() in ("true", "1", "yes", "on")
    try:
        return var_type(value)
    except (ValueError, TypeError):
        return default_value


# --- health check ---
class ServerState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


def create_health_app(get_state):
    """Create FastAPI app with /ping that returns 204/200/500 from get_state()."""
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        state = get_state()
        if state is ServerState.INITIALIZING:
            return Response(status_code=204)
        if state is ServerState.READY:
            return Response(status_code=200)
        return Response(status_code=500)

    return app


# --- tool call parser ---
def infer_tool_call_parser(model_id: str) -> str | None:
    """Return parser name for model_id, or None if unknown."""
    model_lower = model_id.lower()
    if "nousresearch/hermes-" in model_lower:
        return "hermes"
    if "mistralai/mistral-" in model_lower:
        return "mistral"
    if "meta-llama/llama-" in model_lower:
        if "llama-4" in model_lower:
            return "llama4_pythonic"
        if "llama-3.1" in model_lower or "llama-3.2" in model_lower:
            return "llama3_json"
    if "ibm-granite/granite-" in model_lower:
        if "granite-20b-functioncalling" in model_lower:
            return "granite-20b-fc"
        return "granite"
    if "internlm/internlm2_5-" in model_lower or "internlm/internlm2.5-" in model_lower:
        return "internlm"
    if "ai21labs/ai21-jamba-" in model_lower:
        return "jamba"
    if (
        "salesforce/llama-xlam-" in model_lower
        or "salesforce/xlam-" in model_lower
        or "salesforce/qwen-xlam-" in model_lower
    ):
        return "xlam"
    if "qwen/qwen2.5-" in model_lower or "qwen/qwq-" in model_lower:
        return "hermes"
    if "minimaxai/minimax-m1-" in model_lower:
        return "minimax_m1"
    if "deepseek-ai/deepseek-v3-" in model_lower or "deepseek-ai/deepseek-r1-" in model_lower:
        return "deepseek_v3"
    if "moonshotai/kimi-k2-" in model_lower:
        return "kimi_k2"
    if "tencent/hunyuan-a13b-" in model_lower:
        return "hunyuan_a13b"
    return None

"""
Main model serve config: aggregates all task-specific configs (reranker, conversational, etc.).
Server calls get_override_model_config(model_id) to fetch per-model vLLM serve args.
"""
from typing import Any, Optional
from .reranker_config import RERANKER_MODELS

# Aggregate all model configs from task-specific modules.
MODEL_SERVE_CONFIG: dict[str, dict[str, Any]] = {
    **RERANKER_MODELS,
    # Future: **EMBEDDING_MODELS, **CONVERSATIONAL_MODELS (if explicit overrides needed), etc.
}


def get_override_model_config(model_id: str) -> Optional[dict[str, Any]]:
    """
    Return override serve config for this model_id, or None for default (chat/LLM + LoRA + tool choice).
    Config keys the server respects: runner, hf_overrides, enable_lora, enable_tool_choice.
    """
    return MODEL_SERVE_CONFIG.get(model_id)

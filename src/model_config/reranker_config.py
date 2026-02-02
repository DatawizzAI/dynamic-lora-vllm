"""
Reranker (pooling) model configurations: /v1/rerank endpoint, no LoRA, no tool choice.
To add a reranker: append to RERANKER_MODELS with the appropriate hf_overrides.
"""
from typing import Any

# Default hf_overrides for official Qwen3-Reranker models (vLLM loads as sequence classification).
QWEN3_RERANKER_HF_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}

# Profile: pooling/reranker — no LoRA, no tool choice, expose /v1/rerank.
PROFILE_RERANKER: dict[str, Any] = {
    "runner": "pooling",
    "enable_lora": False,
    "enable_tool_choice": False,
}

# Reranker models: each entry inherits PROFILE_RERANKER + model-specific overrides.
RERANKER_MODELS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-Reranker-4B": {
        **PROFILE_RERANKER,
        "hf_overrides": QWEN3_RERANKER_HF_OVERRIDES,
    },
    "Qwen/Qwen3-Reranker-0.6B": {
        **PROFILE_RERANKER,
        "hf_overrides": QWEN3_RERANKER_HF_OVERRIDES,
    },
    "Qwen/Qwen3-Reranker-8B": {
        **PROFILE_RERANKER,
        "hf_overrides": QWEN3_RERANKER_HF_OVERRIDES,
    },
}

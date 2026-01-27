"""Tool call parser inference for different model families."""

from typing import Optional

# Mapping of model ID patterns to tool call parsers
# Patterns are matched case-insensitively against the model ID
TOOL_PARSER_MAPPINGS = {
    # Hermes models
    "nousresearch/hermes-": "hermes",
    
    # Mistral models
    "mistralai/mistral-": "mistral",
    
    # Llama 4 models
    "meta-llama/llama-4": "llama4_pythonic",
    
    # Llama 3.1 and 3.2 models
    "meta-llama/llama-3.1": "llama3_json",
    "meta-llama/llama-3.2": "llama3_json",
    
    # IBM Granite models (specific variant first)
    "ibm-granite/granite-20b-functioncalling": "granite-20b-fc",
    "ibm-granite/granite-": "granite",
    
    # InternLM models
    "internlm/internlm2_5-": "internlm",
    "internlm/internlm2.5-": "internlm",
    
    # Jamba models
    "ai21labs/ai21-jamba-": "jamba",
    
    # xLAM models
    "salesforce/llama-xlam-": "xlam",
    "salesforce/xlam-": "xlam",
    "salesforce/qwen-xlam-": "xlam",
    
    # Qwen models (use hermes parser)
    "qwen/qwen2.5-": "hermes",
    "qwen/qwq-": "hermes",
    
    # MiniMax models
    "minimaxai/minimax-m1-": "minimax_m1",
    
    # DeepSeek models
    "deepseek-ai/deepseek-v3-": "deepseek_v3",
    "deepseek-ai/deepseek-r1-": "deepseek_v3",
    
    # Kimi-K2 models
    "moonshotai/kimi-k2-": "kimi_k2",
    
    # Hunyuan models
    "tencent/hunyuan-a13b-": "hunyuan_a13b",
}


def infer_tool_call_parser(model_id: str) -> Optional[str]:
    """
    Infer the appropriate tool call parser based on model ID.
    
    Args:
        model_id: The model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        
    Returns:
        The parser name or None if no suitable parser is found.
    """
    model_lower = model_id.lower()
    
    for pattern, parser in TOOL_PARSER_MAPPINGS.items():
        if pattern in model_lower:
            return parser
    
    return None

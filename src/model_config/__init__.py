"""
Model serve configuration package.
Exports get_override_model_config() for server to look up per-model vLLM serve args.
"""
from .config import get_override_model_config

__all__ = ["get_override_model_config"]

"""Server configuration management."""

import os
from dataclasses import dataclass, field
from typing import Optional


def get_env_var(name: str, default_value: str = None, var_type: type = str):
    """Get environment variable with type conversion and default value."""
    value = os.getenv(name, default_value)
    if value is None:
        return None
    if var_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    return var_type(value)


@dataclass
class ServerConfig:
    """Configuration for the Dynamic LoRA vLLM server."""
    
    # Model configuration
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    health_port: int = 8001
    
    # LoRA configuration
    max_loras: int = 10
    max_lora_rank: int = 16
    max_cpu_loras: int = 5
    
    # Cache and storage
    cache_dir: str = ".cache/huggingface"
    
    # Authentication
    api_key: Optional[str] = None
    hf_token: Optional[str] = None
    
    # Tool calling
    enable_auto_tool_choice: bool = True
    tool_call_parser: Optional[str] = None
    
    # Chat template
    copy_chat_template: bool = True
    
    # Multimodal configuration
    image_fetch_timeout: int = 5
    video_fetch_timeout: int = 30
    audio_fetch_timeout: int = 10
    max_images_per_prompt: int = 4
    max_videos_per_prompt: int = 1
    max_audios_per_prompt: int = 1
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        return cls(
            model_id=get_env_var("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"),
            host=get_env_var("HOST", "0.0.0.0"),
            port=get_env_var("PORT", "8000", int),
            health_port=get_env_var("PORT_HEALTH", "8001", int),
            max_loras=get_env_var("MAX_LORAS", "10", int),
            max_lora_rank=get_env_var("MAX_LORA_RANK", "16", int),
            max_cpu_loras=get_env_var("MAX_CPU_LORAS", "5", int),
            cache_dir=get_env_var("CACHE_DIR", ".cache/huggingface"),
            api_key=get_env_var("API_KEY"),
            hf_token=get_env_var("HF_TOKEN"),
            enable_auto_tool_choice=get_env_var("ENABLE_AUTO_TOOL_CHOICE", "true", bool),
            tool_call_parser=get_env_var("TOOL_CALL_PARSER"),
            copy_chat_template=get_env_var("COPY_CHAT_TEMPLATE", "true", bool),
            image_fetch_timeout=get_env_var("IMAGE_FETCH_TIMEOUT", "5", int),
            video_fetch_timeout=get_env_var("VIDEO_FETCH_TIMEOUT", "30", int),
            audio_fetch_timeout=get_env_var("AUDIO_FETCH_TIMEOUT", "10", int),
            max_images_per_prompt=get_env_var("MAX_IMAGES_PER_PROMPT", "4", int),
            max_videos_per_prompt=get_env_var("MAX_VIDEOS_PER_PROMPT", "1", int),
            max_audios_per_prompt=get_env_var("MAX_AUDIOS_PER_PROMPT", "1", int),
        )
    
    def setup_environment(self):
        """Set up environment variables for vLLM."""
        # Enable runtime LoRA updating
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        
        # Set HuggingFace cache directory
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        
        # Set multimodal environment variables
        os.environ["VLLM_IMAGE_FETCH_TIMEOUT"] = str(self.image_fetch_timeout)
        os.environ["VLLM_VIDEO_FETCH_TIMEOUT"] = str(self.video_fetch_timeout)
        os.environ["VLLM_AUDIO_FETCH_TIMEOUT"] = str(self.audio_fetch_timeout)
    
    def get_model_cache_path(self) -> str:
        """Get the expected model cache path."""
        return os.path.join(self.cache_dir, f"models--{self.model_id.replace('/', '--')}")
    
    def is_model_cached(self) -> bool:
        """Check if the model is pre-downloaded."""
        return os.path.exists(self.get_model_cache_path())

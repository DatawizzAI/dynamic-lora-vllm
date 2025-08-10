import os
import asyncio
from pathlib import Path
from huggingface_hub import snapshot_download, login
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver


class HuggingFaceLoRAResolver(LoRAResolver):
    """LoRA resolver that downloads adapters from Hugging Face Hub."""
    
    def __init__(self, cache_dir: str = "/tmp/.cache/huggingface", hf_token: str = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        
        if self.hf_token:
            login(token=self.hf_token)

    async def resolve_lora(self, base_model_name: str, lora_name: str) -> LoRARequest:
        """
        Resolve and download LoRA adapter from Hugging Face Hub.
        
        Args:
            base_model_name: The base model name (not used for HF downloads)
            lora_name: The Hugging Face model ID of the LoRA adapter
            
        Returns:
            LoRARequest object with the downloaded adapter
        """
        local_path = self.cache_dir / lora_name.replace("/", "_")
        
        # Check if adapter is already cached
        if not local_path.exists() or not any(local_path.iterdir()):
            print(f"Downloading LoRA adapter {lora_name} from Hugging Face...")
            
            # Download adapter from Hugging Face Hub
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: snapshot_download(
                    repo_id=lora_name,
                    local_dir=str(local_path),
                    token=self.hf_token,
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "adapter_config.json", "adapter_model.*"],
                )
            )
            print(f"Downloaded LoRA adapter {lora_name} to {local_path}")
        else:
            print(f"Using cached LoRA adapter {lora_name} from {local_path}")

        # Create LoRA request
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_path=str(local_path),
            lora_int_id=abs(hash(lora_name)) % (2**31)  # Ensure positive 32-bit integer
        )
        
        return lora_request
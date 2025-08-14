import asyncio
import json
from pathlib import Path
from huggingface_hub import snapshot_download, login
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver


class HuggingFaceLoRAResolver(LoRAResolver):
    """LoRA resolver that downloads adapters from Hugging Face Hub."""
    
    def __init__(self, cache_dir: str = "/tmp/.cache/huggingface", hf_token: str = None, copy_chat_template: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token
        self.copy_chat_template = copy_chat_template
        
        if self.hf_token:
            login(token=self.hf_token)

    def _get_base_model_tokenizer_config_path(self, base_model_name: str) -> Path:
        """Get the path to the base model's tokenizer_config.json file."""
        # Convert model name to cache directory format (e.g., "Qwen/Qwen3-4B" -> "models--Qwen--Qwen3-4B")
        cache_model_name = f"models--{base_model_name.replace('/', '--')}"
        model_cache_dir = self.cache_dir / cache_model_name
        
        # Find the snapshot directory (there should be only one)
        snapshots_dir = model_cache_dir / "snapshots"
        if snapshots_dir.exists():
            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if snapshot_dirs:
                return snapshot_dirs[0] / "tokenizer_config.json"
        return None

    def _get_adapter_tokenizer_config_path(self, lora_name: str) -> Path:
        """Get the path to the adapter's tokenizer_config.json file."""
        local_path = self.cache_dir / lora_name.replace("/", "_")
        return local_path / "tokenizer_config.json"

    def _read_tokenizer_config(self, config_path: Path) -> dict:
        """Read and parse tokenizer_config.json file."""
        if not config_path or not config_path.exists():
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to read tokenizer config from {config_path}: {e}")
            return {}

    def _write_tokenizer_config(self, config_path: Path, config: dict) -> bool:
        """Write tokenizer_config.json file."""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"Warning: Failed to write tokenizer config to {config_path}: {e}")
            return False

    def _copy_chat_template_if_needed(self, base_model_name: str, lora_name: str):
        """Copy chat template from base model to adapter if needed."""
        if not self.copy_chat_template:
            return

        # Get paths
        adapter_config_path = self._get_adapter_tokenizer_config_path(lora_name)
        base_model_config_path = self._get_base_model_tokenizer_config_path(base_model_name)
        
        # Read adapter tokenizer config
        adapter_config = self._read_tokenizer_config(adapter_config_path)
        
        # Check if adapter already has a chat_template
        if adapter_config.get("chat_template"):
            print(f"Adapter {lora_name} already has a chat_template, skipping copy")
            return
        
        # Read base model tokenizer config
        base_model_config = self._read_tokenizer_config(base_model_config_path)
        base_chat_template = base_model_config.get("chat_template")
        
        if not base_chat_template:
            print(f"Base model {base_model_name} does not have a chat_template, skipping copy")
            return
        
        # Copy chat template to adapter config
        adapter_config["chat_template"] = base_chat_template
        
        # Write updated config
        if self._write_tokenizer_config(adapter_config_path, adapter_config):
            print(f"Copied chat_template from base model {base_model_name} to adapter {lora_name}")
        else:
            print(f"Failed to copy chat_template to adapter {lora_name}")

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

        # Copy chat template from base model if needed
        self._copy_chat_template_if_needed(base_model_name, lora_name)

        # Create LoRA request
        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_path=str(local_path),
            lora_int_id=abs(hash(lora_name)) % (2**31)  # Ensure positive 32-bit integer
        )
        
        return lora_request
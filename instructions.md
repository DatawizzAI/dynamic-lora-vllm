I need you help implementing a new service - a vLLM based LLM serving service that supports dynamically loading LoRA adapters from Hugging Face. I have the basic specification and some reference matetial, bellow.

# Dynamic LoRA vLLM

This service allows you to serve an LLM, and dynamically load LoRA adapterts on demand from HuggingFace (you can pass a token to access private models). It exposes an OpenAI compatible API based on vLLM's API compatible API server.

This service is packaged as a Docker image which can be run with the following (and more) env variable:
- PORT - the port the server listens on (default: 8000)
- HOST - the host the server listens on (default: 0.0.0.0)
- MODEL_ID - the ID of the base model to load (default: "meta-llama/Llama-3.2-1B-Instruct")
- HF_TOKEN - the Hugging Face token to access private models (optional)
- CACHE_DIR - the directory to store cached based model and LoRA weights (default: "/tmp/.cache/huggingface")
- MAX_LORAS - the maximum number of LoRA adapters that can be loaded at once (default: 10)
- MAX_LORA_RANK - the maximum rank of LoRA adapters (default: 16)
- MAX_CPU_LORAS - the maximum number of LoRA adapters that can be loaded on CPU (default: 5)

# Implementation

You'll set this up as a Docker image based service, running a python application that uses vLLM to serve the model. The application should use the vLLM OpenAI Compatible API server to handle requests. Make sure to use vLLM's built-in OpenAI compatability and do not implement the OpenAI API yourself.

Additionally, use vLLM's LoRA Loading API to dynamically load LoRA adapters from Hugging Face. The application should handle the following:

The model field of the request will either be the base model ID or a LoRA adapter ID. (alwats a Hugging Face model ID)

# Reference Material:

## vLLM Docs - LoRA Loading Plugin (https://docs.vllm.ai/en/v0.9.1/features/lora.html#using-plugins)
Alternatively, you can use the LoRAResolver plugin to dynamically load LoRA adapters. LoRAResolver plugins enable you to load LoRA adapters from both local and remote sources such as local file system and S3. On every request, when there's a new model name that hasn't been loaded yet, the LoRAResolver will try to resolve and load the corresponding LoRA adapter.

You can set up multiple LoRAResolver plugins if you want to load LoRA adapters from different sources. For example, you might have one resolver for local files and another for S3 storage. vLLM will load the first LoRA adapter that it finds.

You can either install existing plugins or implement your own. By default, vLLM comes with a resolver plugin to load LoRA adapters from a local directory. To enable this resolver, set VLLM_ALLOW_RUNTIME_LORA_UPDATING to True, set VLLM_PLUGINS to include lora_filesystem_resolver, and then set VLLM_LORA_RESOLVER_CACHE_DIR to a local directory. When vLLM receives a request using a LoRA adapter foobar, it will first look in the local directory for a directory foobar, and attempt to load the contents of that directory as a LoRA adapter. If successful, the request will complete as normal and that adapter will then be available for normal use on the server.

Alternatively, follow these example steps to implement your own plugin:

Implement the LoRAResolver interface.
Example of a simple S3 LoRAResolver implementation:
```python
import os
import s3fs
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver

class S3LoRAResolver(LoRAResolver):
    def __init__(self):
        self.s3 = s3fs.S3FileSystem()
        self.s3_path_format = os.getenv("S3_PATH_TEMPLATE")
        self.local_path_format = os.getenv("LOCAL_PATH_TEMPLATE")

    async def resolve_lora(self, base_model_name, lora_name):
        s3_path = self.s3_path_format.format(base_model_name=base_model_name, lora_name=lora_name)
        local_path = self.local_path_format.format(base_model_name=base_model_name, lora_name=lora_name)

        # Download the LoRA from S3 to the local path
        await self.s3._get(
            s3_path, local_path, recursive=True, maxdepth=1
        )

        lora_request = LoRARequest(
            lora_name=lora_name,
            lora_path=local_path,
            lora_int_id=abs(hash(lora_name))
        )
        return lora_request
Register LoRAResolver plugin.

from vllm.lora.resolver import LoRAResolverRegistry

s3_resolver = S3LoRAResolver()
LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)
```

For more details, refer to the vLLM's Plugins System.

## Discussion with Claude on Implementing this (use as a reference only, feel free to generate differently)

Yes, you can absolutely run the OpenAI-compatible server from Python code, which gives you full control over the configuration including custom plugins. Here's how:

```python
import uvicorn
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.entrypoints.openai.api_server import app, init_app_state
from vllm.plugins import load_general_plugins

# Define your custom LoRA loading plugin
class MyLoRALoadingPlugin:
    def __init__(self):
        # Your plugin initialization
        pass
    
    def load_adapter(self, adapter_name, adapter_path):
        # Custom loading logic
        pass
    
    # Implement other required methods based on vLLM's plugin interface

async def main():
    # Configure engine arguments with LoRA support
    engine_args = AsyncEngineArgs(
        model="your-base-model",
        enable_lora=True,
        max_lora_rank=32,  # Adjust based on your adapters
        max_loras=10,  # Maximum number of LoRA adapters to cache
        # Other arguments as needed
    )
    
    # Create the async engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Register your custom plugin
    # Note: The exact method depends on vLLM version
    # You might need to register it through the plugin system
    my_plugin = MyLoRALoadingPlugin()
    # Integration point varies by vLLM version
    
    # Initialize the FastAPI app with the engine
    await init_app_state(
        engine=engine,
        model_config=engine.model_config,
        state=app.state,
        # Additional configuration
    )
    
    # Run the server
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Alternative: Extending the Existing Server

Another approach is to create a wrapper around vLLM's OpenAI server implementation:

```python
from vllm.entrypoints.openai.api_server import create_app
from vllm import AsyncLLMEngine, AsyncEngineArgs
from typing import Optional

class CustomVLLMServer:
    def __init__(
        self,
        model: str,
        lora_plugin: Optional[object] = None,
        **kwargs
    ):
        self.engine_args = AsyncEngineArgs(
            model=model,
            enable_lora=True,
            **kwargs
        )
        self.lora_plugin = lora_plugin
        
    async def start(self, host="0.0.0.0", port=8000):
        # Create engine with custom configuration
        engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        
        # Apply your custom LoRA loading logic
        if self.lora_plugin:
            # Register or configure your plugin here
            # This depends on your plugin's interface
            pass
        
        # Create the FastAPI app
        app = create_app(engine)
        
        # Run the server
        import uvicorn
        config = uvicorn.Config(app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()

# Usage
if __name__ == "__main__":
    import asyncio
    
    server = CustomVLLMServer(
        model="meta-llama/Llama-2-7b-hf",
        max_lora_rank=32,
        max_loras=10
    )
    
    asyncio.run(server.start())
```

### Dynamic LoRA Loading via API

Once your server is running with the custom plugin, you can dynamically load LoRA adapters by including the adapter name in your requests:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "your-base-model",
        "prompt": "Hello, world!",
        "lora_request": {
            "lora_name": "adapter-1",
            "lora_path": "/path/to/adapter1"  # If using dynamic loading
        }
    }
)
```

The key advantage of approach #2 is that you have complete control over the server initialization and can integrate your custom plugin seamlessly while still maintaining full OpenAI API compatibility.
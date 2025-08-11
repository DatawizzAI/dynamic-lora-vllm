#!/usr/bin/env python3
"""
Test client for the Dynamic LoRA vLLM service.
"""

import requests
import json
import time
import argparse


def test_completions(base_url: str, model: str, prompt: str):
    """Test the completions endpoint."""
    url = f"{base_url}/v1/completions"
    
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    
    print(f"Testing completions with model: {model}")
    print(f"Prompt: {prompt}")
    
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print("Response:")
        print(json.dumps(result, indent=2))
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return False


def test_chat_completions(base_url: str, model: str, message: str):
    """Test the chat completions endpoint."""
    url = f"{base_url}/v1/chat/completions"
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": message}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    
    print(f"Testing chat completions with model: {model}")
    print(f"Message: {message}")
    
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print("Response:")
        print(json.dumps(result, indent=2))
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return False


def test_health(base_url: str):
    """Test the health endpoint."""
    url = f"{base_url}/health"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        print("Health check passed")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test the Dynamic LoRA vLLM service")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the service")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="Model to test with")
    parser.add_argument("--lora-model", help="LoRA adapter model to test with (optional)")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Prompt to test with")
    
    args = parser.parse_args()
    
    print("Dynamic LoRA vLLM Service Test")
    print("=" * 40)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    if not test_health(args.base_url):
        print("Service is not healthy. Exiting.")
        return
    
    # Wait a bit for the service to be ready
    print("\nWaiting 5 seconds for service to be ready...")
    time.sleep(5)
    
    # Test base model with completions
    print("\n2. Testing base model with completions...")
    success = test_completions(args.base_url, args.model, args.prompt)
    
    if success:
        print("\n3. Testing base model with chat completions...")
        test_chat_completions(args.base_url, args.model, args.prompt)
    
    # Test LoRA model if provided
    if args.lora_model:
        print(f"\n4. Testing LoRA model ({args.lora_model}) with completions...")
        test_completions(args.base_url, args.lora_model, args.prompt)
        
        print(f"\n5. Testing LoRA model ({args.lora_model}) with chat completions...")
        test_chat_completions(args.base_url, args.lora_model, args.prompt)
    
    print("\nTesting completed!")


if __name__ == "__main__":
    main()
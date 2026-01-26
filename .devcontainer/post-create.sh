#!/bin/bash
set -e

echo "🚀 Setting up Dynamic LoRA vLLM development environment..."

# Create .cache directory if it doesn't exist
echo "📁 Creating cache directory..."
mkdir -p /workspace/.cache/huggingface
chmod 755 /workspace/.cache/huggingface

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x run.sh 2>/dev/null || true
chmod +x test_client.py 2>/dev/null || true

# Create a development .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️ Creating development .env file..."
    cp .env.example .env 2>/dev/null || true
fi

# Check environment
echo "🔍 Checking environment..."
python -c "
import torch
import vllm
print(f'PyTorch version: {torch.__version__}')
print(f'vLLM version: {vllm.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Display helpful information
echo ""
echo "✅ Development environment ready!"
echo ""
echo "📋 Quick start commands:"
echo "  ./run.sh                    - Start the server"
echo "  python test_client.py       - Test the API"
echo "  jupyter lab --ip=0.0.0.0    - Start Jupyter Lab"
echo ""
echo "🔧 Environment:"
echo "  Base: NGC PyTorch 25.10 (torch 2.9.0)"
echo "  Python: $(python --version)"
echo ""
echo "🚀 Ready for development!"

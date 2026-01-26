#!/bin/bash
set -e

echo "🚀 Setting up Dynamic LoRA vLLM development environment..."

# Ensure venv is on PATH for new shells
echo 'export PATH="/workspace/.venv/bin:$PATH"' >> ~/.bashrc
PYTHON="/workspace/.venv/bin/python"

# Install any additional development dependencies into venv
echo "📦 Installing additional development dependencies..."
uv pip install --python "$PYTHON" python-dotenv rich click

# Install Claude Code CLI
echo "🤖 Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

# Note: Claude Code VS Code extension will be available after installation
echo "💡 Claude Code CLI installed. To use in VS Code:"
echo "   1. Run 'claude auth' to authenticate"
echo "   2. Use Cmd+Shift+P > 'Claude Code: Chat' to start"
echo "   3. Or run 'claude' in terminal for CLI access"

# Create .cache directory if it doesn't exist
echo "📁 Creating cache directory..."
mkdir -p /tmp/.cache/huggingface
chmod 755 /tmp/.cache/huggingface

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x run.sh 2>/dev/null || true
chmod +x test_client.py 2>/dev/null || true

# Set up git hooks if pre-commit config exists
if [ -f .pre-commit-config.yaml ]; then
    echo "🪝 Setting up pre-commit hooks..."
    pre-commit install || true
fi

# Create a development .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️ Creating development .env file..."
    cp .env.example .env 2>/dev/null || true
fi

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError:
    print('PyTorch not yet installed - will be available after requirements installation')
" 2>/dev/null || echo "CUDA check will be available after installing requirements"

# Display helpful information
echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "📋 Quick start commands:"
echo "  ./run.sh                    - Start the server locally"
echo "  python test_client.py       - Test the API"
echo "  docker-compose up           - Run in container"
echo "  jupyter lab --ip=0.0.0.0    - Start Jupyter Lab"
echo ""
echo "🔧 Configuration:"
echo "  Environment variables: .env"
echo "  Requirements: requirements.txt"
echo "  Documentation: README.md, CLAUDE.md"
echo ""
echo "🚀 Ready for development!"

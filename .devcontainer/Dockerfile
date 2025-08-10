FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/miniconda3/bin:$PATH"

# Install system dependencies and development tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    build-essential \
    python3-dev \
    vim \
    nano \
    htop \
    tmux \
    tree \
    jq \
    zsh \
    sudo \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI for Docker-in-Docker scenarios
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user for development
RUN groupadd --gid 1000 vscode && \
    useradd --uid 1000 --gid vscode --shell /bin/zsh --create-home vscode && \
    echo vscode ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/vscode && \
    chmod 0440 /etc/sudoers.d/vscode

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda3 && \
    rm miniconda.sh && \
    chown -R vscode:vscode /opt/miniconda3

# Switch to non-root user
USER vscode
WORKDIR /home/vscode

# Initialize conda for the user
RUN /opt/miniconda3/bin/conda init bash && \
    /opt/miniconda3/bin/conda init zsh

# Create conda environment with Python 3.11
RUN /opt/miniconda3/bin/conda create -n vllm python=3.11 -y

# Set up shell environment
SHELL ["/opt/miniconda3/bin/conda", "run", "-n", "vllm", "/bin/bash", "-c"]

# Install development and debugging tools
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    notebook \
    jupyterlab \
    pytest \
    pytest-asyncio \
    black \
    flake8 \
    pylint \
    isort \
    mypy \
    pre-commit \
    httpx \
    requests

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up git configuration
RUN git config --global init.defaultBranch main && \
    git config --global pull.rebase false

# Default command
CMD ["/bin/zsh"]
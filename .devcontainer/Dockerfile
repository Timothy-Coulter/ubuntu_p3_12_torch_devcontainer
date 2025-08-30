# syntax=docker/dockerfile:1.7

# Volume-optimized multi-stage build for PyTorch development environment
# Designed for use with named Docker volumes and repository cloning
# Uses PyTorch 2.8.0 with CUDA 12.9 and cuDNN 9 runtime with system Python and UV

# ==============================================================================
# PyTorch Base Stage - Official PyTorch image with CUDA runtime
# ==============================================================================
# Set default platform (fix for devcontainer build issue)
ARG TARGETPLATFORM=linux/amd64
ARG BUILDPLATFORM
ARG TARGETARCH

FROM --platform=$TARGETPLATFORM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS pytorch-base

# Build arguments for cache control and versioning
ARG BUILDKIT_INLINE_CACHE=1
ARG CUDA_VERSION=12.9
ARG PYTORCH_VERSION=2.8.0
ARG APP_USER=ubuntu
ARG APP_UID=1000
ARG APP_GID=1000
# Configurable CUDA architectures
ARG TORCH_CUDA_ARCH_LIST=7.0,7.5,8.0,8.6,8.9,9.0

# Verify PyTorch and CUDA installation with system Python and validate versions
RUN python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" && \
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import sys; print(f'Python {sys.version}')" && \
    python -c "import torch; assert torch.version.cuda == '${CUDA_VERSION}', f'CUDA version mismatch: expected ${CUDA_VERSION}, got {torch.version.cuda}'" && \
    echo "✅ PyTorch base image verified with system Python $(python --version) and CUDA ${CUDA_VERSION}"

# ==============================================================================
# System Dependencies Stage - Essential tools and libraries
# ==============================================================================
FROM pytorch-base AS system-deps

# System package installation with optimized caching
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    # Essential development tools
    ca-certificates curl git git-lfs wget unzip \
    # Build essentials for Python packages
    build-essential pkg-config cmake ninja-build \
    # System libraries commonly needed by ML packages
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk-3-dev libcanberra-gtk3-dev \
    # Audio processing libraries
    libsndfile1-dev libsox-dev \
    # SSH and process management
    openssh-client tini sudo \
    # Performance monitoring tools
    htop iotop ncdu tree vim tmux screen \
    # Math libraries
    bc \
    # Additional utilities for debugging
    strace lsof procps \
    # Cleanup in same layer
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && git lfs install --system \
    && apt-get autoremove -y \
    && apt-get autoclean

# ==============================================================================
# User Setup Stage - Create non-root user with proper permissions
# ==============================================================================
FROM system-deps AS user-setup

# Enhanced user creation with better group management
RUN if ! getent group ${APP_USER} >/dev/null; then \
    groupadd -g ${APP_GID} ${APP_USER}; \
    fi && \
    if ! id -u ${APP_USER} >/dev/null 2>&1; then \
    useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER}; \
    fi && \
    usermod -aG sudo ${APP_USER} && \
    # Add user to common groups for better volume compatibility
    usermod -aG users,staff ${APP_USER} 2>/dev/null || true && \
    # Enable passwordless sudo for development convenience
    echo "${APP_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${APP_USER} && \
    chmod 0440 /etc/sudoers.d/${APP_USER} && \
    # Create all necessary directories with proper ownership
    mkdir -p \
    /home/${APP_USER}/.cache/huggingface \
    /home/${APP_USER}/.cache/torch \
    /home/${APP_USER}/.cache/uv \
    /home/${APP_USER}/.cache/pip \
    /home/${APP_USER}/.kaggle \
    /home/${APP_USER}/.local/bin \
    /workspaces \
    /workspaces/torch-starter && \
    # Set ownership with enhanced error handling
    chown -R ${APP_USER}:${APP_USER} /home/${APP_USER} /workspaces 2>/dev/null || true && \
    # Set proper permissions with fallback
    chmod 755 /home/${APP_USER} /workspaces && \
    chmod -R u+rwx /home/${APP_USER}/.cache /home/${APP_USER}/.local 2>/dev/null || true && \
    # Create symlinks for common cache locations
    ln -sf /home/${APP_USER}/.cache/huggingface /home/${APP_USER}/.cache/transformers 2>/dev/null || true

# ==============================================================================
# UV Package Manager Stage - Install UV with system Python
# ==============================================================================
FROM user-setup AS uv-install

# Switch to non-root user
USER ${APP_USER}
ENV HOME=/home/${APP_USER}
WORKDIR /workspaces/torch-starter

# UV installation and configuration for system Python
ENV UV_INSTALL_DIR=/home/${APP_USER}/.local/bin \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_SYSTEM_PYTHON=1 \
    UV_CONCURRENT_DOWNLOADS=20 \
    UV_HTTP_TIMEOUT=600 \
    UV_CACHE_DIR=/home/${APP_USER}/.cache/uv \
    PATH=/home/${APP_USER}/.local/bin:${PATH}

# Install UV with cache mounting for faster rebuilds and version pinning
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    UV_VERSION=0.1.45 curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv --version && \
    # Verify UV can find system Python
    uv python list && \
    echo "✅ UV installed for system Python usage"

# ==============================================================================
# Development Environment Setup - Configure environment for volume-based development
# ==============================================================================
FROM uv-install AS dev-environment

# Enhanced Python performance environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONOPTIMIZE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Enhanced memory management
    MALLOC_ARENA_MAX=4 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TRIM_THRESHOLD_=131072

# Architecture-specific optimizations
RUN if [ "${TARGETARCH}" = "arm64" ]; then \
    echo "# ARM64 (Apple Silicon) optimizations" >> ~/.bashrc && \
    echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> ~/.bashrc && \
    echo "export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0" >> ~/.bashrc; \
    elif [ "${TARGETARCH}" = "amd64" ]; then \
    echo "# AMD64 optimizations" >> ~/.bashrc && \
    echo "export OMP_NUM_THREADS=4" >> ~/.bashrc && \
    echo "export MKL_NUM_THREADS=4" >> ~/.bashrc && \
    echo "export OPENBLAS_NUM_THREADS=4" >> ~/.bashrc; \
    fi

# Create enhanced convenience scripts and aliases
RUN echo '#!/bin/bash\nwatch -n 1 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv"' > /home/${APP_USER}/.local/bin/gpu-monitor && \
    chmod +x /home/${APP_USER}/.local/bin/gpu-monitor && \
    # Enhanced GPU info script
    echo '#!/bin/bash\necho "🖥️  GPU Information:"\nnvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPUs found"\necho "\n🔥 PyTorch GPU Status:"\npython -c "import torch; print(f\"CUDA Available: {torch.cuda.is_available()}\"); print(f\"Device Count: {torch.cuda.device_count()}\"); [print(f\"Device {i}: {torch.cuda.get_device_name(i)}\") for i in range(torch.cuda.device_count())]"' > /home/${APP_USER}/.local/bin/gpu-info && \
    chmod +x /home/${APP_USER}/.local/bin/gpu-info && \
    # Create UV management script
    echo '#!/bin/bash\necho "🔧 UV System Package Management"\necho "Current packages:"\nuv pip list --system\necho "\n📊 UV Cache Status:"\necho "Cache location: $UV_CACHE_DIR"\ndu -sh "$UV_CACHE_DIR" 2>/dev/null || echo "Cache directory not found"' > /home/${APP_USER}/.local/bin/uv-status && \
    chmod +x /home/${APP_USER}/.local/bin/uv-status && \
    # Enhanced project setup script
    echo '#!/bin/bash\nif [ -f "pyproject.toml" ]; then\n  echo "📦 Installing project with UV..."\n  echo "🔍 Checking dependencies..."\n  uv pip install --system -e .\n  echo "✅ Project installed"\n  echo "🧪 Running basic import test..."\n  python -c "import torch; print(f\"✅ PyTorch {torch.__version__} available\")" || echo "⚠️  PyTorch import failed"\nelse\n  echo "❌ No pyproject.toml found"\n  echo "💡 Make sure you are in the project root directory"\nfi' > /home/${APP_USER}/.local/bin/install-project && \
    chmod +x /home/${APP_USER}/.local/bin/install-project && \
    # Enhanced system diagnostics script
    echo '#!/bin/bash\necho "🔍 System Diagnostics"\necho "=================="\necho "🐍 Python: $(python --version)"\necho "🔥 PyTorch: $(python -c \"import torch; print(torch.__version__)\" 2>/dev/null || echo \"Not available\")"\necho "🎯 CUDA: $(python -c \"import torch; print(torch.version.cuda)\" 2>/dev/null || echo \"Not available\")"\necho "💾 Memory: $(free -h | grep Mem: | awk \"{print \\$3\\\"/\\\"\\$2}\")"\necho "💿 Disk: $(df -h /workspaces | tail -1 | awk \"{print \\$3\\\"/\\\"\\$2 \\\" (\\\" \\$5 \\\" used)\\\"}\")"' > /home/${APP_USER}/.local/bin/system-info && \
    chmod +x /home/${APP_USER}/.local/bin/system-info && \
    # Setup enhanced convenience aliases in bashrc
    echo 'export PATH=/home/ubuntu/.local/bin:$PATH' >> ~/.bashrc && \
    echo '# UV aliases' >> ~/.bashrc && \
    echo 'alias uv-install="uv pip install --system"' >> ~/.bashrc && \
    echo 'alias uv-list="uv pip list --system"' >> ~/.bashrc && \
    echo 'alias uv-sync="uv pip sync --system"' >> ~/.bashrc && \
    echo 'alias uv-uninstall="uv pip uninstall --system"' >> ~/.bashrc && \
    echo '# PyTorch/CUDA aliases' >> ~/.bashrc && \
    echo 'alias torch-info="python -c \"import torch; print(f\\\"PyTorch: {torch.__version__}\\\"); print(f\\\"CUDA: {torch.version.cuda}\\\"); print(f\\\"cuDNN: {torch.backends.cudnn.version()}\\\"); print(f\\\"Devices: {torch.cuda.device_count()}\\\")\\""' >> ~/.bashrc && \
    echo 'alias cuda-info="nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv"' >> ~/.bashrc && \
    echo '# Navigation and utilities' >> ~/.bashrc && \
    echo 'alias ll="ls -alF"' >> ~/.bashrc && \
    echo 'alias la="ls -A"' >> ~/.bashrc && \
    echo 'alias l="ls -CF"' >> ~/.bashrc && \
    echo 'cd /workspaces/torch-starter 2>/dev/null || cd /workspaces' >> ~/.bashrc && \
    echo '# Welcome message' >> ~/.bashrc && \
    echo 'echo "🚀 PyTorch DevContainer Ready!"' >> ~/.bashrc && \
    echo 'echo "💡 Use gpu-info for GPU status, system-info for diagnostics"' >> ~/.bashrc && \
    echo "✅ Enhanced development environment setup completed"

# ==============================================================================
# Final Stage - Runtime configuration without project files
# ==============================================================================
FROM dev-environment AS final

# Create enhanced template files for better development experience
RUN echo '# torch-starter\n\nPyTorch development environment with CUDA support.\n\n## Quick Start\n\n1. Clone your repository into this volume\n2. Run `install-project` to install dependencies\n3. Start developing!\n\n## Available Commands\n\n### Package Management\n- `uv-install <package>` - Install packages with UV\n- `uv-list` - List installed packages\n- `uv-status` - Show UV cache status\n- `install-project` - Install current project\n\n### GPU Monitoring\n- `gpu-monitor` - Monitor GPU usage in real-time\n- `gpu-info` - Show GPU information\n- `torch-info` - Show PyTorch/CUDA info\n\n### System Diagnostics\n- `system-info` - Show system diagnostics\n- `./dev.sh verify-setup` - Run comprehensive verification\n- `python verify_setup/test_gpu.py` - Test GPU functionality\n\n## Environment Variables\n\n- `CUDA_VERSION`: ' ${CUDA_VERSION} '\n- `PYTORCH_VERSION`: ' ${PYTORCH_VERSION} '\n- `UV_CACHE_DIR`: Cache location for UV\n- `HF_HOME`: Hugging Face cache location\n\n## Troubleshooting\n\n### CUDA Issues\n1. Check GPU availability: `nvidia-smi`\n2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`\n3. Run diagnostics: `./dev.sh doctor`\n\n### Permission Issues\n1. Check volume ownership: `ls -la /workspaces`\n2. Fix permissions: `sudo chown -R ubuntu:ubuntu /workspaces`\n\n### Performance Issues\n1. Monitor resources: `htop` and `gpu-monitor`\n2. Clear caches: `./dev.sh clean`\n3. Restart container: Rebuild in VS Code\n' > /workspaces/torch-starter/README.md && \
    # Create enhanced gitignore template
    echo '__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\nwheels/\n*.egg-info/\n.installed.cfg\n*.egg\nPYMANIFEST\n\n# Environment files\n.env\n.env.local\n.venv\nenv/\nvenv/\nENV/\nenv.bak/\nvenv.bak/\n\n# IDE and cache\n.vscode/settings.json\n.mypy_cache/\n.pytest_cache/\n.ruff_cache/\n.coverage\nhtmlcov/\n\n# Data and models\ndata/\nlogs/\nmodels/\ncheckpoints/\n*.log\n\n# System files\n.DS_Store\nThumbs.db\n\n# Jupyter\n.ipynb_checkpoints/\n*/.ipynb_checkpoints/*\n\n# Large files\n*.bin\n*.safetensors\n*.ckpt\n*.pth\n*.pkl\n*.h5\n' > /workspaces/torch-starter/.gitignore.template && \
    # Create environment template
    echo '# Environment Configuration Template\n# Copy to .env.local and fill in your API keys\n# DO NOT COMMIT .env.local TO GIT\n\n# Hugging Face\n# Get from: https://huggingface.co/settings/tokens\n# HUGGINGFACE_TOKEN=hf_...\n\n# OpenAI\n# Get from: https://platform.openai.com/api-keys\n# OPENAI_API_KEY=sk-...\n\n# Weights & Biases\n# Get from: https://wandb.ai/settings\n# WANDB_API_KEY=...\n\n# Anthropic (Claude)\n# Get from: https://console.anthropic.com/\n# ANTHROPIC_API_KEY=sk-ant-...\n\n# Kaggle\n# Get from: https://www.kaggle.com/settings/account\n# KAGGLE_USERNAME=...\n# KAGGLE_KEY=...\n\n# Custom paths (optional)\n# HF_HOME=/home/ubuntu/.cache/huggingface\n# TORCH_HOME=/home/ubuntu/.cache/torch\n# DATA_DIR=/workspaces/torch-starter/data\n' > /workspaces/torch-starter/.env.example && \
    echo "✅ Volume-based development environment ready with enhanced templates"

# Runtime verification command with enhanced diagnostics
CMD echo "🔍 Verifying volume-based environment..." && \
    echo "📊 System Information:" && \
    echo "  Architecture: $(uname -m)" && \
    echo "  Kernel: $(uname -r)" && \
    echo "  Container Runtime: $(cat /.dockerenv >/dev/null 2>&1 && echo 'Docker' || echo 'Unknown')" && \
    python -c "import sys; print(f'✅ Python {sys.version.split()[0]}')" && \
    python -c "import torch; print(f'✅ PyTorch {torch.__version__} - CUDA: {torch.version.cuda}')" && \
    python -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import torch; print(f'✅ CUDA Devices: {torch.cuda.device_count()}')" && \
    uv --version && echo "✅ UV package manager ready" && \
    echo "✅ Environment verification completed" && \
    echo "📁 Workspace ready at /workspaces/torch-starter" && \
    echo "💡 Clone your repository or create new projects here" && \
    echo "🔧 Run 'system-info' for detailed diagnostics" && \
    /bin/bash -l

# ==============================================================================
# Runtime Configuration and Health Checks
# ==============================================================================

# Expose common development ports
EXPOSE 8888 8000 6006 7860 3000 5000

# Comprehensive health check that verifies CUDA functionality with timeout
HEALTHCHECK --interval=300s --timeout=60s --start-period=120s --retries=3 \
    CMD timeout 30s /bin/bash -c "python -c \"import torch; print(f'Health check: PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); \
    if torch.cuda.is_available(): \
        try: \
            x = torch.tensor([1.0], device='cuda'); print(f'GPU test: {x.device}'); del x; torch.cuda.empty_cache(); \
        except Exception as e: \
            print(f'GPU test failed: {e}'); exit(1); \
    print('Health check passed'); exit(0)\"" || exit 1

# Set up container entrypoint with proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]

# ==============================================================================
# Metadata and Labels
# ==============================================================================
LABEL org.opencontainers.image.title="torch_starter-volume-enhanced" \
    org.opencontainers.image.description="Enhanced volume-optimized PyTorch 2.8.0 development environment with CUDA 12.9, UV package manager, and system Python" \
    org.opencontainers.image.version="2.8-cuda${CUDA_VERSION}-volume-enhanced" \
    org.opencontainers.image.vendor="PyTorch DevContainer Enhanced" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.source="https://github.com/yourusername/torch-starter" \
    cuda.version="${CUDA_VERSION}" \
    pytorch.version="${PYTORCH_VERSION}" \
    python.version="system" \
    deployment.type="volume" \
    architecture="${TARGETARCH:-amd64}" \
    enhanced.features="multi-arch,diagnostics,security"

# Enhanced development environment variables for better IDE integration
ENV PYTHONPATH=/workspaces/torch-starter:${PYTHONPATH} \
    JUPYTER_ENABLE_LAB=yes \
    JUPYTER_TOKEN="" \
    JUPYTER_ALLOW_ROOT=yes \
    # Enhanced cache locations
    HF_HOME=/home/${APP_USER}/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/${APP_USER}/.cache/huggingface \
    TORCH_HOME=/home/${APP_USER}/.cache/torch \
    KAGGLE_CONFIG_DIR=/home/${APP_USER}/.kaggle \
    # Enhanced performance optimizations
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4 \
    # Enhanced UV system integration
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/home/${APP_USER}/.cache/uv \
    # Enhanced workspace configuration
    WORKSPACE_ROOT=/workspaces/torch-starter \
    # Enhanced CUDA configuration
    CUDA_VERSION=${CUDA_VERSION} \
    PYTORCH_VERSION=${PYTORCH_VERSION} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # Development mode indicators
    DEVELOPMENT_MODE=1 \
    CONTAINER_ENHANCED=true
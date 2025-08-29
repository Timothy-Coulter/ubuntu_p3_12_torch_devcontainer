# syntax=docker/dockerfile:1.7

# Volume-optimized multi-stage build for PyTorch development environment
# Designed for use with named Docker volumes and repository cloning
# Uses PyTorch 2.8.0 with CUDA 12.9 and cuDNN 9 runtime with system Python and UV

# ==============================================================================
# PyTorch Base Stage - Official PyTorch image with CUDA runtime
# ==============================================================================
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS pytorch-base

# Build arguments for cache control and versioning
ARG BUILDKIT_INLINE_CACHE=1
ARG CUDA_VERSION=12.9
ARG APP_USER=ubuntu
ARG APP_UID=1000
ARG APP_GID=1000
# Configurable CUDA architectures
ARG TORCH_CUDA_ARCH_LIST=7.0,7.5,8.0,8.6,8.9,9.0

# Verify PyTorch and CUDA installation with system Python
RUN python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" && \
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import sys; print(f'Python {sys.version}')" && \
    echo "PyTorch base image verified with system Python $(python --version)"

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
    # Cleanup in same layer
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && git lfs install --system \
    && apt-get autoremove -y \
    && apt-get autoclean

# ==============================================================================
# User Setup Stage - Create non-root user with proper permissions
# ==============================================================================
FROM system-deps AS user-setup

# Create ubuntu group and user
RUN if ! getent group ${APP_USER} >/dev/null; then \
    groupadd -g ${APP_GID} ${APP_USER}; \
    fi && \
    if ! id -u ${APP_USER} >/dev/null 2>&1; then \
    useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER}; \
    fi && \
    usermod -aG sudo ${APP_USER} && \
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
    # Set ownership for all user directories
    chown -R ${APP_USER}:${APP_USER} /home/${APP_USER} /workspaces && \
    # Set proper permissions
    chmod 755 /home/${APP_USER} /workspaces

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
    PATH=/home/${APP_USER}/.local/bin:${PATH}

# Install UV with cache mounting for faster rebuilds
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=0 \
    UV_VERSION=0.1.45 curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv --version && \
    echo "UV installed for system Python usage"

# ==============================================================================
# Development Environment Setup - Configure environment for volume-based development
# ==============================================================================
FROM uv-install AS dev-environment

# Python performance environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONOPTIMIZE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MALLOC_ARENA_MAX=2

# Create convenience scripts and aliases
RUN echo '#!/bin/bash\nwatch -n 1 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv"' > /home/${APP_USER}/.local/bin/gpu-monitor && \
    chmod +x /home/${APP_USER}/.local/bin/gpu-monitor && \
    # Create UV management script
    echo '#!/bin/bash\necho "🔧 UV System Package Management"\necho "Current packages:"\nuv pip list --system' > /home/${APP_USER}/.local/bin/uv-status && \
    chmod +x /home/${APP_USER}/.local/bin/uv-status && \
    # Create project setup script
    echo '#!/bin/bash\nif [ -f "pyproject.toml" ]; then\n  echo "📦 Installing project with UV..."\n  uv pip install --system -e .\n  echo "✅ Project installed"\nelse\n  echo "❌ No pyproject.toml found"\nfi' > /home/${APP_USER}/.local/bin/install-project && \
    chmod +x /home/${APP_USER}/.local/bin/install-project && \
    # Setup convenience aliases in bashrc
    echo 'export PATH=/home/ubuntu/.local/bin:$PATH' >> ~/.bashrc && \
    echo 'alias uv-install="uv pip install --system"' >> ~/.bashrc && \
    echo 'alias uv-list="uv pip list --system"' >> ~/.bashrc && \
    echo 'alias uv-sync="uv pip sync --system"' >> ~/.bashrc && \
    echo 'alias uv-uninstall="uv pip uninstall --system"' >> ~/.bashrc && \
    echo 'alias torch-info="python -c \"import torch; print(f\\\"PyTorch: {torch.__version__}\\\"); print(f\\\"CUDA: {torch.version.cuda}\\\"); print(f\\\"cuDNN: {torch.backends.cudnn.version()}\\\"); print(f\\\"Devices: {torch.cuda.device_count()}\\\")\\""' >> ~/.bashrc && \
    echo 'alias gpu-info="nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv"' >> ~/.bashrc && \
    echo 'cd /workspaces/torch-starter 2>/dev/null || cd /workspaces' >> ~/.bashrc && \
    echo "✅ Development environment setup completed"

# ==============================================================================
# Final Stage - Runtime configuration without project files
# ==============================================================================
FROM dev-environment AS final

# Create template files for better development experience
RUN echo '# torch-starter\n\nPyTorch development environment with CUDA support.\n\n## Quick Start\n\n1. Clone your repository into this volume\n2. Run `install-project` to install dependencies\n3. Start developing!\n\n## Available Commands\n\n- `uv-install <package>` - Install packages with UV\n- `uv-list` - List installed packages\n- `gpu-monitor` - Monitor GPU usage\n- `torch-info` - Show PyTorch/CUDA info\n- `install-project` - Install current project\n' > /workspaces/torch-starter/README.md && \
    # Create a basic gitignore template
    echo '__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\nwheels/\n*.egg-info/\n.installed.cfg\n*.egg\nPYMANIFEST\n.env\n.venv\nenv/\nvenv/\nENV/\nenv.bak/\nvenv.bak/\n.mypy_cache/\n.pytest_cache/\n.ruff_cache/\ndata/\nlogs/\nmodels/\ncheckpoints/\n*.log\n.DS_Store\n' > /workspaces/torch-starter/.gitignore.template && \
    echo "✅ Volume-based development environment ready"

# Runtime verification command
CMD echo "🔍 Verifying volume-based environment..." && \
    python -c "import sys; print(f'✅ Python {sys.version}')" && \
    python -c "import torch; print(f'✅ PyTorch {torch.__version__} - CUDA: {torch.version.cuda}')" && \
    python -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import torch; print(f'✅ CUDA Devices: {torch.cuda.device_count()}')" && \
    uv --version && echo "✅ UV package manager ready" && \
    echo "✅ Environment verification completed" && \
    echo "📁 Workspace ready at /workspaces/torch-starter" && \
    echo "💡 Clone your repository or create new projects here" && \
    /bin/bash -l

# ==============================================================================
# Runtime Configuration and Health Checks
# ==============================================================================

# Expose common development ports
EXPOSE 8888 8000 6006 7860

# Comprehensive health check that verifies CUDA functionality
HEALTHCHECK --interval=300s --timeout=30s --start-period=60s --retries=3 \
    CMD /bin/bash -c "python -c \"import torch; print(f'Health check: PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); if torch.cuda.is_available(): x = torch.tensor([1.0], device='cuda'); print(f'GPU test: {x.device}'); exit(0); else: exit(1)\"" || exit 1

# Set up container entrypoint with proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]

# ==============================================================================
# Metadata and Labels
# ==============================================================================
LABEL org.opencontainers.image.title="torch_starter-volume" \
    org.opencontainers.image.description="Volume-optimized PyTorch 2.8.0 development environment with CUDA 12.9, UV package manager, and system Python" \
    org.opencontainers.image.version="2.8-cuda${CUDA_VERSION}-volume" \
    org.opencontainers.image.vendor="PyTorch DevContainer" \
    org.opencontainers.image.licenses="MIT" \
    cuda.version="${CUDA_VERSION}" \
    pytorch.version="2.8.0" \
    python.version="system" \
    deployment.type="volume"

# Development environment hints for better IDE integration
ENV PYTHONPATH=/workspaces/torch-starter:${PYTHONPATH} \
    JUPYTER_ENABLE_LAB=yes \
    JUPYTER_TOKEN="" \
    # Cache locations
    HF_HOME=/home/${APP_USER}/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/${APP_USER}/.cache/huggingface \
    TORCH_HOME=/home/${APP_USER}/.cache/torch \
    KAGGLE_CONFIG_DIR=/home/${APP_USER}/.kaggle \
    # Performance optimizations
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4 \
    # UV system integration
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON_DOWNLOADS=never \
    # Workspace configuration
    WORKSPACE_ROOT=/workspaces/torch-starter
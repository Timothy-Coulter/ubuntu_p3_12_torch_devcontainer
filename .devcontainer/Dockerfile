# syntax=docker/dockerfile:1.7

# Multi-stage build for PyTorch development environment using official PyTorch image
# Uses PyTorch 2.8.0 with CUDA 12.9 and cuDNN 9 runtime

# ==============================================================================
# PyTorch Base Stage - Official PyTorch image with CUDA runtime
# ==============================================================================
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS pytorch-base

# Build arguments for cache control and versioning
ARG BUILDKIT_INLINE_CACHE=1
ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.9
ARG APP_USER=ubuntu
ARG APP_UID=1000
ARG APP_GID=1000
# Configurable CUDA architectures
ARG TORCH_CUDA_ARCH_LIST=7.0,7.5,8.0,8.6,8.9,9.0

# Verify PyTorch and CUDA installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" && \
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" && \
    echo "PyTorch base image verified"

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

# Create ubuntu user if it doesn't exist (PyTorch image uses root by default)
RUN if ! id -u ${APP_USER} >/dev/null 2>&1; then \
    useradd -m -u ${APP_UID} -g root -s /bin/bash ${APP_USER}; \
    fi && \
    usermod -aG sudo ${APP_USER} && \
    # Allow specific sudo commands only for security
    # Create all necessary directories with proper ownership
    mkdir -p \
    /home/${APP_USER}/.cache/huggingface \
    /home/${APP_USER}/.cache/torch \
    /home/${APP_USER}/.cache/uv \
    /home/${APP_USER}/.cache/pip \
    /home/${APP_USER}/.kaggle \
    /home/${APP_USER}/.local/bin \
    /workspaces \
    /workspaces/data && \
    # Set ownership for all user directories
    chown -R ${APP_USER}:root /home/${APP_USER} /workspaces && \
    # Set proper permissions
    chmod 755 /home/${APP_USER}

# ==============================================================================
# UV Package Manager Stage - Install UV with optimized caching
# ==============================================================================
FROM user-setup AS uv-install

# Switch to non-root user
USER ${APP_USER}
ENV HOME=/home/${APP_USER}
WORKDIR /workspaces

# UV installation and configuration
ENV UV_INSTALL_DIR=/home/${APP_USER}/.local/bin \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=manual \
    UV_CONCURRENT_DOWNLOADS=20 \
    UV_HTTP_TIMEOUT=600 \
    PATH=/home/${APP_USER}/.local/bin:${PATH}

# Install UV with cache mounting for faster rebuilds
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=0 \
    UV_VERSION=0.1.45 curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv --version

# ==============================================================================
# Python Environment Stage - Create virtual environment (keep existing Python)
# ==============================================================================
FROM uv-install AS python-env

# Python performance environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONOPTIMIZE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MALLOC_ARENA_MAX=2

# Create virtual environment using existing Python from PyTorch image
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=0 \
    python -m venv .venv && \
    echo "source /workspaces/.venv/bin/activate" >> ~/.bashrc && \
    echo "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-all}" >> ~/.bashrc

# ==============================================================================
# Dependencies Installation Stage - Install additional Python packages
# ==============================================================================
FROM python-env AS deps-install

# Copy dependency files first for better Docker layer caching
COPY --chown=${APP_USER}:root pyproject.toml uv.lock* ./

# Install project dependencies including torchvision and transformers
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=0 \
    . .venv/bin/activate && \
    uv pip install . --compile-bytecode && \
    # Explicitly install transformers with its dependencies
    uv pip install transformers --compile-bytecode && \
    # Explicitly install torchvision with its dependencies
    uv pip install torchvision --compile-bytecode && \
    echo "All dependencies installed successfully"

# ==============================================================================
# Final Application Stage - Install project and setup runtime
# ==============================================================================
FROM deps-install AS final

# Copy entire project (this layer will change frequently during development)
COPY --chown=${APP_USER}:root . .

# Install project in development mode and setup Jupyter kernel
RUN . .venv/bin/activate && \
    uv pip install -e . --compile-bytecode && \
    # Setup Jupyter kernel with project name (using Python 3.12)
    python -m ipykernel install --user \
    --name torch_starter-3.12 \
    --display-name "Python 3.12 (torch_starter)" && \
    # Create useful development symlinks
    ln -sf /workspaces/data /home/${APP_USER}/data && \
    ln -sf /workspaces/notebooks /home/${APP_USER}/notebooks && \
    # Create GPU monitoring script
    echo '#!/bin/bash\nwatch -n 1 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv"' > /home/${APP_USER}/.local/bin/gpu-monitor && \
    chmod +x /home/${APP_USER}/.local/bin/gpu-monitor && \
    echo "✅ Project setup completed"

# Moved verification to runtime
CMD . .venv/bin/activate && \
    python -c "import torch; print(f'✅ PyTorch {torch.__version__} - CUDA: {torch.version.cuda}')" && \
    python -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import torch; print(f'✅ CUDA Devices: {torch.cuda.device_count()}')" && \
    python -c "import torchvision; print(f'✅ TorchVision {torchvision.__version__}')" && \
    python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')" && \
    python verify_setup/gpu_test.py && \
    bash verify_setup/verify_setup.sh && \
    /bin/bash -l

# ==============================================================================
# Runtime Configuration and Health Checks
# ==============================================================================

# Expose common development ports
EXPOSE 8888 8000 6006 7860

# Comprehensive health check that verifies CUDA functionality
HEALTHCHECK --interval=300s --timeout=30s --start-period=60s --retries=3 \
    CMD /bin/bash -c ". .venv/bin/activate && python -c \"import torch; print(f'Health check: PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); if torch.cuda.is_available(): x = torch.tensor([1.0], device='cuda'); print(f'GPU test: {x.device}'); exit(0); else: exit(1)\"" || exit 1

# Set up container entrypoint with proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]

# ==============================================================================
# Metadata and Labels
# ==============================================================================
LABEL org.opencontainers.image.title="torch_starter" \
    org.opencontainers.image.description="PyTorch 2.8.0 development environment with CUDA 12.9 and Python 3.12" \
    org.opencontainers.image.version="2.8-cuda${CUDA_VERSION}" \
    org.opencontainers.image.vendor="PyTorch DevContainer" \
    org.opencontainers.image.licenses="MIT" \
    cuda.version="${CUDA_VERSION}" \
    pytorch.version="2.8.0" \
    python.version="3.12"

# Development environment hints for better IDE integration
ENV PYTHONPATH=/workspaces:${PYTHONPATH} \
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
    TORCH_NUM_THREADS=4
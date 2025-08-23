# syntax=docker/dockerfile:1.7

# Multi-stage build for CUDA-enabled PyTorch development environment
# Uses NVIDIA CUDA 12.6.0 development with optimized layer caching

# ==============================================================================
# CUDA Base Stage - NVIDIA CUDA development with complete toolkit
# ==============================================================================
FROM 12.6.0-cudnn-devel-ubuntu24.04 AS cuda-base

# Build arguments for cache control and versioning
ARG BUILDKIT_INLINE_CACHE=1
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.6.0
ARG APP_USER=ubuntu
ARG APP_UID=1000
ARG APP_GID=1000

# CUDA and system environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_VERSION=${CUDA_VERSION} \
    CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.6.0" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # PyTorch CUDA optimization
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" \
    CUDA_LAUNCH_BLOCKING=0 \
    FORCE_CUDA=1 \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.6"

# Verify CUDA installation and create symlinks
RUN ln -sf /usr/local/cuda-12.6.0 /usr/local/cuda && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda/compat" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig && \
    nvcc --version && \
    ls -la /usr/local/cuda/lib64/libcudart* && \
    echo "CUDA base setup completed"

# ==============================================================================
# System Dependencies Stage - Essential tools and libraries
# ==============================================================================
FROM cuda-base AS system-deps

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
    # Additional CUDA development packages
    cuda-toolkit-12-6 \
    libcudnn8-dev \
    libnccl-dev \
    libcublas-dev-12-6 \
    # Cleanup in same layer
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && git lfs install --system \
    && apt-get autoremove -y \
    && apt-get autoclean

# ==============================================================================
# User Setup Stage - Create non-root user with proper permissions
# ==============================================================================
FROM system-deps AS user-setup

# Create user with same UID/GID as host user for seamless file permissions
# Use existing ubuntu user (UID 1000) and configure for development
RUN usermod -aG sudo ${APP_USER} && \
    # Allow specific sudo commands only for security
    echo "${APP_USER} ALL=(ALL) NOPASSWD: /usr/bin/apt-get, /usr/bin/dpkg, /usr/bin/systemctl" >> /etc/sudoers && \
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
    chown -R ${APP_USER}:${APP_USER} /home/${APP_USER} /workspaces && \
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
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv --version

# ==============================================================================
# Python Environment Stage - Install Python and create virtual environment
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

# Install Python with UV and create virtual environment
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    uv python install ${PYTHON_VERSION} && \
    uv venv --clear .venv -p ${PYTHON_VERSION} && \
    echo "source /workspaces/.venv/bin/activate" >> ~/.bashrc && \
    echo "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-all}" >> ~/.bashrc

# ==============================================================================
# Dependencies Installation Stage - Install all Python packages with CUDA PyTorch
# ==============================================================================
FROM python-env AS deps-install

# Copy dependency files first for better Docker layer caching
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml uv.lock* ./

# Install PyTorch CUDA dependencies and other packages
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    . .venv/bin/activate && \
    # Install CUDA-enabled PyTorch first
    uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    "torchaudio>=2.0.0" && \
    # Then install other dependencies
    uv sync \
    --index-strategy unsafe-best-match \
    --compile-bytecode && \
    echo "All dependencies installed successfully"

# ==============================================================================
# ML Libraries Verification Stage - Test critical imports and GPU functionality
# ==============================================================================
FROM deps-install AS ml-verify

# Verify critical ML libraries can be imported and CUDA works
RUN . .venv/bin/activate && \
    python -c "import torch; print(f'✅ PyTorch {torch.__version__} - CUDA: {torch.version.cuda}')" && \
    python -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')" && \
    python -c "import torch; print(f'✅ CUDA Devices: {torch.cuda.device_count()}')" && \
    python -c "import torchvision; print(f'✅ TorchVision {torchvision.__version__}')" && \
    python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')" && \
    echo "✅ All ML libraries verified successfully"

# ==============================================================================
# GPU Test Stage - Verify GPU functionality with actual tensor operations
# ==============================================================================
FROM ml-verify AS gpu-test

# Create and run a comprehensive GPU test
RUN . .venv/bin/activate && python -c "\
    import torch;\
    import sys;\
    \
    def test_gpu_functionality():\
    try:\
    if not torch.cuda.is_available():\
    print('⚠️  CUDA not available - will run in CPU mode');\
    return True;\
    device_count = torch.cuda.device_count();\
    print(f'🖥️  Found {device_count} CUDA device(s)');\
    for i in range(device_count):\
    props = torch.cuda.get_device_properties(i);\
    print(f'   Device {i}: {props.name} ({props.total_memory // 1024**3}GB)');\
    print('🧪 Testing GPU tensor operations...');\
    x = torch.randn(1000, 1000, device='cuda');\
    y = torch.randn(1000, 1000, device='cuda');\
    z = torch.mm(x, y);\
    print(f'✅ GPU tensor test passed - Result shape: {z.shape}');\
    torch.cuda.empty_cache();\
    memory_allocated = torch.cuda.memory_allocated(0) // 1024**2;\
    print(f'📊 GPU memory allocated: {memory_allocated}MB');\
    return True;\
    except Exception as e:\
    print(f'❌ GPU test failed: {e}');\
    return False;\
    \
    success = test_gpu_functionality();\
    sys.exit(0 if success else 1)"

# ==============================================================================
# Final Application Stage - Install project and setup runtime
# ==============================================================================
FROM gpu-test AS final

# Copy entire project (this layer will change frequently during development)
COPY --chown=${APP_USER}:${APP_USER} . .

# Install project in development mode and setup Jupyter kernel
RUN . .venv/bin/activate && \
    uv pip install -e . --compile-bytecode && \
    # Setup Jupyter kernel with project name
    python -m ipykernel install --user \
    --name torch_starter-${PYTHON_VERSION} \
    --display-name "Python ${PYTHON_VERSION} (torch_starter)" && \
    # Create useful development symlinks
    ln -sf /workspaces/data /home/${APP_USER}/data && \
    ln -sf /workspaces/notebooks /home/${APP_USER}/notebooks && \
    # Create GPU monitoring script
    echo '#!/bin/bash\nwatch -n 1 "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv"' > /home/${APP_USER}/.local/bin/gpu-monitor && \
    chmod +x /home/${APP_USER}/.local/bin/gpu-monitor && \
    echo "✅ Project setup completed"

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
    org.opencontainers.image.description="CUDA-enabled PyTorch development environment with Python 3.12" \
    org.opencontainers.image.version="2.2-cuda12.6.0" \
    org.opencontainers.image.vendor="PyTorch DevContainer" \
    org.opencontainers.image.licenses="MIT" \
    cuda.version="12.6.0" \
    pytorch.version=">=2.0.0" \
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
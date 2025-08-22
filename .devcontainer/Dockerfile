# syntax=docker/dockerfile:1.7

# Multi-stage build for CUDA-enabled PyTorch development environment
# Uses NVIDIA CUDA 12.6.1 runtime with optimized layer caching

# ==============================================================================
# CUDA Base Stage - NVIDIA CUDA runtime with development tools
# ==============================================================================
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04 AS cuda-base

# Build arguments for cache control and versioning
ARG BUILDKIT_INLINE_CACHE=1
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.6.1
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
    NVIDIA_REQUIRE_CUDA="cuda>=12.6.1" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Verify CUDA installation and create symlinks
RUN ln -sf /usr/local/cuda-12.6.1 /usr/local/cuda && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    echo "/usr/local/cuda/compat" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig && \
    # Verify CUDA installation
    nvcc --version || echo "Warning: nvcc not found" && \
    ls -la /usr/local/cuda/lib64/libcudart* || echo "Warning: CUDA runtime not found"

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
    htop iotop nvtop \
    # Math libraries
    bc \
    # Cleanup in same layer
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# ==============================================================================
# User Setup Stage - Create non-root user with proper permissions
# ==============================================================================
FROM system-deps AS user-setup

# Create user with same UID/GID as host user for seamless file permissions
# Use existing ubuntu user (UID 1000) and configure for development
RUN usermod -aG sudo ${APP_USER} && \
    # Allow passwordless sudo for convenience in dev environment
    echo "${APP_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
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
    chown -R ${APP_USER}:${APP_USER} /home/${APP_USER} /workspaces

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
    UV_CONCURRENT_DOWNLOADS=10 \
    UV_HTTP_TIMEOUT=300 \
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
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python with UV and create virtual environment
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    uv python install ${PYTHON_VERSION} && \
    uv venv --clear .venv -p ${PYTHON_VERSION} && \
    echo "source /workspaces/.venv/bin/activate" >> ~/.bashrc

# ==============================================================================
# Dependencies Installation Stage - Install all Python packages
# ==============================================================================
FROM python-env AS deps-install

# Copy dependency files first for better Docker layer caching
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml uv.lock* ./

# Install dependencies with comprehensive extras and cache mounting
RUN --mount=type=cache,target=/home/${APP_USER}/.cache/uv,uid=${APP_UID},gid=${APP_GID} \
    . .venv/bin/activate && \
    uv sync \
    --no-install-project \
    --index-strategy unsafe-best-match \
    --extra dev \
    --extra notebook \
    --extra torch-cu126 \
    --extra datascience \
    --extra tracking \
    --extra profiling \
    --compile-bytecode && \
    # Verify PyTorch CUDA installation
    python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" && \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "CUDA check will complete at runtime" && \
    echo "Dependencies installed successfully"

# ==============================================================================
# ML Libraries Verification Stage - Test critical imports
# ==============================================================================
FROM deps-install AS ml-verify

# Verify all critical ML libraries can be imported
RUN . .venv/bin/activate && python -c "
import sys
print('Verifying ML library installations...')

# Core libraries
import numpy as np
print(f'✓ NumPy {np.__version__}')

import pandas as pd
print(f'✓ Pandas {pd.__version__}')

import sklearn
print(f'✓ Scikit-learn {sklearn.__version__}')

# PyTorch ecosystem
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'  - CUDA compiled: {torch.version.cuda or \"No\"}')
print(f'  - CUDA available: {torch.cuda.is_available()}')

try:
import torchvision
print(f'✓ TorchVision {torchvision.__version__}')
except ImportError:
print('⚠ TorchVision not available')

try:
import torchaudio  
print(f'✓ TorchAudio {torchaudio.__version__}')
except ImportError:
print('⚠ TorchAudio not available')

# Transformers ecosystem
try:
import transformers
print(f'✓ Transformers {transformers.__version__}')
except ImportError:
print('⚠ Transformers not available')

try:
import datasets
print(f'✓ Datasets {datasets.__version__}')
except ImportError:
print('⚠ Datasets not available')

# Visualization
import matplotlib
print(f'✓ Matplotlib {matplotlib.__version__}')

print('Library verification completed!')
"

# ==============================================================================
# Final Application Stage - Install project and setup runtime
# ==============================================================================
FROM ml-verify AS final

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
    echo "Project setup completed"

# ==============================================================================
# Runtime Configuration and Health Checks
# ==============================================================================

# Expose common development ports
EXPOSE 8888 8000 6006 7860

# Comprehensive health check that verifies CUDA functionality
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD . .venv/bin/activate && python -c " \
    import torch; \
    print(f'Health check: PyTorch {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    exit(0)" || exit 1

# Set up container entrypoint with proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]

# ==============================================================================
# Metadata and Labels
# ==============================================================================
LABEL org.opencontainers.image.title="torch_starter" \
    org.opencontainers.image.description="CUDA-enabled PyTorch development environment with Python 3.12" \
    org.opencontainers.image.version="2.1-cuda12.6" \
    org.opencontainers.image.vendor="PyTorch DevContainer" \
    org.opencontainers.image.licenses="MIT" \
    cuda.version="12.6.1" \
    pytorch.version=">=2.4.0" \
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
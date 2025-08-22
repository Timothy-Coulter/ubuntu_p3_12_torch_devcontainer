# syntax=docker/dockerfile:1.7

# Multi-stage build for better caching and smaller final image
# Using NVIDIA CUDA 12.6 base image with cuDNN runtime pre-installed
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04 AS base-deps

# Build arguments for cache control
ARG BUILDKIT_INLINE_CACHE=1
ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_EXTRA=torch-cu126

# Environment setup - consolidated for better layer caching
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_INSTALL_DIR=/usr/local/bin \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System dependencies - optimized layer with cleanup
# CUDA runtime libraries already included in base image
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    # Essential tools
    ca-certificates curl git git-lfs wget \
    # Build tools (minimal set)
    build-essential pkg-config \
    # SSH and process management
    openssh-client tini \
    # Performance tools
    htop iotop bc \
    # Additional utilities
    zsh sudo \
    # Cleanup in same layer
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# Set CUDA environment variables for CUDA 12.6
ENV CUDA_HOME=/usr/local/cuda-12.6 \
    PATH=/usr/local/cuda-12.6/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /root/.cargo/bin/uv --version

# Add uv to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# User setup
ARG APP_USER=vscode
ENV HOME=/home/${APP_USER}

# Create all directories in single layer
RUN mkdir -p \
    ${HOME}/.cache/huggingface \
    ${HOME}/.cache/torch \
    ${HOME}/.cache/uv \
    ${HOME}/.kaggle \
    /workspaces/data \
    && useradd -m -s /bin/zsh -u 1000 ${APP_USER} \
    && chown -R ${APP_USER}:${APP_USER} ${HOME} /workspaces \
    && usermod -aG sudo ${APP_USER} \
    && echo "${APP_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to user early
USER ${APP_USER}
WORKDIR /workspaces

# Ensure uv is available for the user
ENV PATH="/home/${APP_USER}/.cargo/bin:${PATH}"

# Python environment setup stage
FROM base-deps AS python-env

# UV configuration for performance
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=automatic

# Install Python with cache
RUN --mount=type=cache,target=/home/vscode/.cache/uv,uid=1000 \
    uv python install ${PYTHON_VERSION}

# Dependency installation stage
FROM python-env AS deps-install

# Copy dependency files first (better caching)
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml uv.lock ./

# Create venv and install dependencies with cache mounting
RUN --mount=type=cache,target=/home/vscode/.cache/uv,uid=1000 \
    uv venv --clear .venv -p ${PYTHON_VERSION} \
    && . .venv/bin/activate \
    && uv sync \
    --no-install-project \
    --index-strategy unsafe-best-match \
    --extra dev \
    --extra notebook \
    --extra ${TORCH_CUDA_EXTRA} \
    --compile-bytecode \
    && echo "External dependencies installed"

# Final stage - minimal layers for runtime
FROM deps-install AS final

# Copy source code (separate layer for development)
COPY --chown=${APP_USER}:${APP_USER} . .

# Install project and setup Jupyter kernel in single layer
RUN . .venv/bin/activate \
    && uv pip install -e . --compile-bytecode \
    && python -m ipykernel install --user \
    --name torch_starter-${PYTHON_VERSION} \
    --display-name "Python ${PYTHON_VERSION} (torch_starter)" \
    && echo "Project setup completed"

# Enhanced validation with CUDA test
RUN . .venv/bin/activate \
    && python -c "import sys; print(f'Python {sys.version}')" \
    && python -c "import torch; print(f'PyTorch {torch.__version__}')" \
    && python -c "import transformers; print(f'Transformers {transformers.__version__}')" \
    && python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" \
    && python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')" \
    || echo "Warning: Some packages may not be available or CUDA setup incomplete"

# Runtime configuration
EXPOSE 8888 8000 6006

# Enhanced health check with CUDA
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /bin/bash -c "source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available(), \"CUDA not available\"'" || exit 1

# Optimized entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]

# Labels for better organization
LABEL org.opencontainers.image.title="torch_starter"
LABEL org.opencontainers.image.description="Optimized PyTorch development environment with CUDA 12.6 support"
LABEL org.opencontainers.image.version="2.2-cuda126-optimized"
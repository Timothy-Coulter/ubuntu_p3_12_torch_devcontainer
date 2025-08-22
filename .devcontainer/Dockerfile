# syntax=docker/dockerfile:1.7

# Multi-stage build for better caching and smaller final image
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-24.04 AS base-deps

# Build arguments for cache control
ARG BUILDKIT_INLINE_CACHE=1
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=13.0

# Environment setup - consolidated for better layer caching
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_VERSION=${CUDA_VERSION} \
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
    htop iotop \
    bc \
    # Cleanup in same layer
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# Install uv
RUN --mount=type=cache,target=/home/vscode/.cache/uv,uid=1000 \
    curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv --version

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
    && chown -R ${APP_USER}:${APP_USER} ${HOME} /workspaces

# Switch to user early
USER ${APP_USER}
WORKDIR /workspaces

# Python environment setup stage
FROM base-deps AS python-env

# UV configuration for performance
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=manual

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
    --extra torch-cu130 \
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

# Quick validation (non-blocking)
RUN . .venv/bin/activate \
    && python -c "import sys; print(f'Python {sys.version}')" \
    && python -c "import torch; print(f'PyTorch {torch.__version__}')" \
    && python -c "import transformers; print(f'Transformers {transformers.__version__}')" \
    || echo "Warning: Some packages may not be available"

# Runtime configuration
EXPOSE 8888 8000 6006

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import torch" || exit 1

# Optimized entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]

# Labels for better organization
LABEL org.opencontainers.image.title="torch_starter"
LABEL org.opencontainers.image.description="Optimized PyTorch development environment"
LABEL org.opencontainers.image.version="2.0-optimized"
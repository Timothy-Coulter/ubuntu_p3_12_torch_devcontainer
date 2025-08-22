# syntax=docker/dockerfile:1.7

# DEBUG VERSION - Simplified build with better error handling and logging
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 AS base-debug

# Build arguments
ARG BUILDKIT_INLINE_CACHE=1
ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_EXTRA=torch-cu130

# Environment setup with debug logging
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Add debug helper script
RUN echo '#!/bin/bash\necho "[DEBUG] $1"\necho "Current user: $(whoami)"\necho "Current dir: $(pwd)"\necho "Environment: $*"\n' > /usr/local/bin/debug-log && \
    chmod +x /usr/local/bin/debug-log

RUN debug-log "Starting base system setup"

# System dependencies with better error handling
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    debug-log "Installing system packages" && \
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
    && debug-log "System packages installed successfully" \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system \
    && debug-log "Git LFS configured"

# Set CUDA environment variables for CUDA 13.0
ENV CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN debug-log "CUDA environment configured" && \
    ls -la /usr/local/ && \
    echo "CUDA_HOME: $CUDA_HOME" && \
    echo "PATH: $PATH"

# User setup BEFORE uv installation to avoid permission issues
ARG APP_USER=vscode
ENV HOME=/home/${APP_USER}

RUN debug-log "Creating user and directories" && \
    (groupadd --gid 1000 ${APP_USER} 2>/dev/null || echo "Group 1000 already exists, checking if it's suitable...") && \
    (useradd --uid 1000 --gid 1000 -m ${APP_USER} 2>/dev/null || echo "User creation failed, trying alternative approach...") && \
    if ! id ${APP_USER} &>/dev/null; then \
    echo "Creating user with alternative method..." && \
    useradd --uid 1001 --create-home ${APP_USER} && \
    usermod -aG $(getent group 1000 | cut -d: -f1) ${APP_USER}; \
    fi && \
    mkdir -p \
    /home/${APP_USER}/.cache/huggingface \
    /home/${APP_USER}/.cache/torch \
    /home/${APP_USER}/.cache/uv \
    /home/${APP_USER}/.kaggle \
    /workspaces/data \
    && chown -R ${APP_USER}:${APP_USER} /home/${APP_USER} /workspaces \
    && debug-log "User and directories created successfully"

# Switch to user BEFORE installing uv
USER ${APP_USER}
WORKDIR /workspaces

RUN debug-log "Installing uv as user ${APP_USER}"

# Install uv as the user to avoid permission issues
RUN debug-log "Installing uv for user" && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && \
    export PATH="$HOME/.local/bin:$PATH" && \
    which uv && \
    uv --version && \
    debug-log "uv installed successfully"

# UV configuration for performance
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=manual \
    PATH="/home/${APP_USER}/.local/bin:${PATH}"

# Install Python with debug output
RUN debug-log "Installing Python ${PYTHON_VERSION}" && \
    uv python install ${PYTHON_VERSION} && \
    uv python list && \
    debug-log "Python installation completed"

# Copy dependency files first (better caching)
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml ./
COPY --chown=${APP_USER}:${APP_USER} uv.lock ./

RUN debug-log "Creating virtual environment" && \
    uv venv --clear .venv -p ${PYTHON_VERSION} && \
    debug-log "Virtual environment created"

# Install dependencies with detailed logging
RUN debug-log "Starting dependency installation" && \
    . .venv/bin/activate && \
    debug-log "Virtual environment activated" && \
    echo "Python path: $(which python)" && \
    echo "Python version: $(python --version)" && \
    uv sync \
    --no-install-project \
    --index-strategy unsafe-best-match \
    --extra dev \
    --extra notebook \
    --extra ${TORCH_CUDA_EXTRA} \
    --index pytorch-cu130 \
    --compile-bytecode \
    --verbose && \
    debug-log "Dependencies installed successfully"

# Copy source code
COPY --chown=${APP_USER}:${APP_USER} . .

# Install project with debug output
RUN debug-log "Installing project in development mode" && \
    . .venv/bin/activate && \
    uv pip install -e . --compile-bytecode && \
    python -m ipykernel install --user \
    --name torch_starter-${PYTHON_VERSION} \
    --display-name "Python ${PYTHON_VERSION} (torch_starter)" && \
    debug-log "Project installation completed"

# Enhanced validation with detailed CUDA test
RUN debug-log "Running validation tests" && \
    . .venv/bin/activate && \
    python -c "import sys; print(f'Python {sys.version}')" && \
    python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No CUDA devices')" && \
    python -c "import transformers; print(f'Transformers {transformers.__version__}')" && \
    debug-log "Validation completed"

# Runtime configuration
EXPOSE 8888 8000 6006

# Simple health check without CUDA requirement (since build environment may not have GPU)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Optimized entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]

# Labels for better organization
LABEL org.opencontainers.image.title="torch_starter"
LABEL org.opencontainers.image.description="PyTorch development environment"
LABEL org.opencontainers.image.version="v1"
# syntax=docker/dockerfile:1

# Ubuntu 24.04 base with optimized package installation at build time
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/usr/local/bin \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Core OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs \
    build-essential pkg-config \
    openssh-client tini \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# Install uv (system-wide)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version

# App user: reuse existing 'vscode' user from base image
ARG APP_USER=vscode
ENV HOME=/home/${APP_USER}

# Cache and data directories (will be bound to host volumes at runtime)
ENV HF_HOME=${HOME}/.cache/huggingface \
    TRANSFORMERS_CACHE=${HOME}/.cache/huggingface \
    TORCH_HOME=${HOME}/.cache/torch \
    KAGGLE_CONFIG_DIR=${HOME}/.kaggle \
    DATA_DIR=/workspaces/data

RUN mkdir -p ${HF_HOME} ${TORCH_HOME} ${KAGGLE_CONFIG_DIR} ${DATA_DIR} && \
    chown -R ${APP_USER}:${APP_USER} ${HOME} /workspaces

# Declare volumes for host cache mounting
VOLUME ["${HF_HOME}", "${TORCH_HOME}", "${KAGGLE_CONFIG_DIR}", "/workspaces/data"]

# Set working directory and switch to app user
WORKDIR /workspaces
USER ${APP_USER}

# Environment variables for uv optimization
ENV UV_LINK_MODE=copy \
    UV_VENV_CLEAR=1

# OPTIMIZATION: Copy dependency files first (cached layer unless dependencies change)
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml uv.lock ./

# OPTIMIZATION: Install Python and create virtual environment with all dependencies at build time
RUN set -euo pipefail && \
    uv python install 3.12 && \
    uv venv --clear .venv -p 3.12 && \
    . .venv/bin/activate && \
    uv sync --index-strategy unsafe-best-match --extra dev --extra notebook --extra torch-cu124 && \
    echo "✅ All dependencies installed successfully"

# OPTIMIZATION: Setup Jupyter kernel at build time
RUN set -euo pipefail && \
    . .venv/bin/activate && \
    python -m ipykernel install --user --name torch_starter-3.12 --display-name "Python 3.12 (torch_starter)" && \
    echo "✅ Jupyter kernel installed"

# OPTIMIZATION: Copy source code last (this layer rebuilds when source changes, but deps are cached)
COPY --chown=${APP_USER}:${APP_USER} . .

# Validate environment is working (but don't fail build if optional components missing)
RUN set -euo pipefail && \
    . .venv/bin/activate && \
    python -c "import sys; print(f'Python {sys.version}')" && \
    python -c "import torch; print(f'PyTorch {torch.__version__}')" && \
    python -c "import transformers; print(f'Transformers {transformers.__version__}')" && \
    echo "✅ Environment validation completed"

# Expose common ports
EXPOSE 8888 8000

# Proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
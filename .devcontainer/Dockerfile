# syntax=docker/dockerfile:1

# Ubuntu 24.04 base with smallest common tooling; we'll install uv and Python via uv.
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/usr/local/bin \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Core OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs \
    build-essential pkg-config \
    openssh-client tini \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# Install uv (system-wide)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv --version

# App user
ARG APP_USER=app
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} ${APP_USER} && \
    useradd -m -s /bin/bash -u ${APP_UID} -g ${APP_GID} ${APP_USER}

# Cache and data directories (these will be bound to host volumes at runtime)
ENV HOME=/home/${APP_USER}
ENV HF_HOME=${HOME}/.cache/huggingface \
    TRANSFORMERS_CACHE=${HOME}/.cache/huggingface \
    TORCH_HOME=${HOME}/.cache/torch \
    KAGGLE_CONFIG_DIR=${HOME}/.kaggle \
    DATA_DIR=/workspace/data

RUN mkdir -p ${HF_HOME} ${TORCH_HOME} ${KAGGLE_CONFIG_DIR} ${DATA_DIR} && \
    chown -R ${APP_USER}:${APP_USER} ${HOME} /workspace

# Declare volumes so users can mount host caches easily:
# - On Linux: mount ${HOME}/.cache/huggingface, ${HOME}/.cache/torch, ${HOME}/.kaggle, and ~/data
# - On Windows (Docker Desktop/WSL): mount %USERPROFILE% equivalents or WSL paths
VOLUME ["${HF_HOME}", "${TORCH_HOME}", "${KAGGLE_CONFIG_DIR}", "/workspace/data"]

WORKDIR /workspace
USER ${APP_USER}

# Create a project venv and install dependencies using uv.
# Copy only the lock & metadata first for better caching; will copy the rest later.
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml /workspace/pyproject.toml
# Optional: include lock file if present for deterministic builds
# COPY --chown=${APP_USER}:${APP_USER} uv.lock /workspace/uv.lock

# Ensure Python 3.12 and venv are present, then sync project deps (except torch CUDA, installed below).
RUN uv python install 3.12 && \
    uv venv .venv -p 3.12 && \
    . .venv/bin/activate && \
    uv sync --frozen || uv sync

# Install PyTorch CUDA 12.4 wheels. Requires host NVIDIA driver and nvidia-container-toolkit at runtime.
# If you want CPU-only images, remove the index-url and install default wheels.
RUN . .venv/bin/activate && \
    uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Copy the remainder of the repository
COPY --chown=${APP_USER}:${APP_USER} . /workspace

# Register a Jupyter kernel for convenience
RUN . .venv/bin/activate && \
    python -m ipykernel install --user --name torch_starter-3.12 --display-name "Python 3.12 (torch_starter)"

# Expose a common Jupyter port if needed
EXPOSE 8888

# Proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
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

# App user: reuse existing 'vscode' user from base image to avoid UID/GID conflicts
ARG APP_USER=vscode
# Using the existing vscode user; no need to create user/group

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

# Development dependencies and the virtual environment are created during container postCreate (see devcontainer.json).
# We avoid copying project files or installing Python packages at build time to keep images small and builds fast.

# Expose a common Jupyter port if needed
EXPOSE 8888

# Proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
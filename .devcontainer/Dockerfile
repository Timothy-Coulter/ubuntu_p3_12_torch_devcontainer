# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04 AS base-deps

ARG BUILDKIT_INLINE_CACHE=1
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda-12.6 \
    PATH=/usr/local/cuda-12.6/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# System deps
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs build-essential pkg-config \
    openssh-client tini zsh sudo htop iotop bc \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# Install uv system-wide
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv $HOME/.local/bin/uv /usr/local/bin/ \
    && mv $HOME/.local/bin/uvx /usr/local/bin/ \
    && uv --version

# User
ARG APP_USER=ubuntu
ENV HOME=/home/${APP_USER}
RUN mkdir -p ${HOME}/.cache && chown -R ${APP_USER}:${APP_USER} ${HOME}
USER ${APP_USER}
WORKDIR /workspaces

# -------------------
# Python stage
FROM base-deps AS python-env
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN uv python install ${PYTHON_VERSION}

# -------------------
# Dependencies stage
FROM python-env AS deps-install

COPY --chown=${APP_USER}:${APP_USER} pyproject.toml uv.lock ./

# Create venv
RUN uv venv --clear .venv -p ${PYTHON_VERSION} \
    && . .venv/bin/activate \
    # Install all deps EXCEPT torch, torchvision, torchaudio
    && uv sync --no-install-project --index-strategy unsafe-best-match \
       --extra dev --extra notebook \
    && echo "Base dependencies installed"

# ⚡ Install torch from official CUDA 12.6 wheels using UV
RUN . .venv/bin/activate \
    && uv pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
       torch torchvision torchaudio \
    && echo "Torch stack installed via PyTorch CU126 wheels using UV"

# -------------------
# Final runtime
FROM deps-install AS final

USER ${APP_USER}
COPY --chown=${APP_USER}:${APP_USER} . .

RUN . .venv/bin/activate \
    && uv pip install -e . --compile-bytecode \
    && python -m ipykernel install --user \
       --name torch_starter-${PYTHON_VERSION} \
       --display-name "Python ${PYTHON_VERSION} (torch_starter)" \
    && echo "Project setup completed"

# Validate torch & CUDA
RUN . .venv/bin/activate \
    && python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Avail:', torch.cuda.is_available())"

EXPOSE 8888 8000 6006
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]
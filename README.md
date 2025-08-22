# Ubuntu PyTorch Dev Container with CUDA 13.0

An optimized deep learning development environment with PyTorch, CUDA 13.0, and VS Code Dev Containers. This setup provides a streamlined, reliable environment with pre-configured CUDA 13.0 support.

## Features

- **CUDA 13.0 Ready**: Pre-installed CUDA 13.0 runtime with cuDNN for maximum compatibility
- **Latest PyTorch**: Automatically configured with PyTorch optimized for CUDA 13.0
- **Optimized Performance**: Multi-stage Docker build with caching for fast rebuilds
- **GPU Ready**: Full NVIDIA GPU support with pre-installed CUDA runtime libraries
- **Modern Tooling**: uv package manager, ruff linter, mypy type checking
- **Pre-configured**: Jupyter Lab, Hugging Face, Transformers, and more

## Quick Start

1. Open in VS Code with Dev Containers extension
2. Container automatically builds with CUDA 13.0 and latest compatible PyTorch
3. Start developing with full GPU acceleration

## CUDA 13.0 Configuration

This dev container is optimized for CUDA 13.0:

### Pre-installed Components

- **Base Image**: `nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04`
- **CUDA Runtime**: Version 13.0 with cuDNN support
- **PyTorch**: Latest version with CUDA 13.0 support (`torch-cu130`)

### Benefits of Fixed CUDA Version

- **Reliability**: No version compatibility issues
- **Performance**: Optimized for specific CUDA version
- **Simplicity**: No complex version management
- **Faster Builds**: Pre-installed CUDA libraries

## PyTorch Installation

The setup uses PyTorch optimized for CUDA 13.0:

Install dependencies with:
```bash
./dev.sh sync  # Installs torch-cu130 extra
```

## Performance Optimizations

- Multi-stage Docker build for minimal image size
- Layer caching for fast rebuilds
- uv package manager for fast dependency resolution
- Pre-compiled bytecode for faster startup
- Optimized CUDA library installation

## Development Workflow

```bash
# Sync dependencies
./dev.sh sync

# Run all quality checks
./dev.sh all-checks

# Start Jupyter Lab
./dev.sh jupyter

# Run tests
./dev.sh test

# Profile code
./dev.sh profile script.py
```

## GPU Verification

The setup includes comprehensive CUDA 13.0 verification:

```bash
# Verify CUDA 13.0 setup
./dev.sh verify-setup

# Run GPU benchmarks
./dev.sh benchmark
```

## Security Features

- Secure API key management with `.env.local`
- Pre-commit hooks for code quality
- Security scanning with safety
- Restricted file permissions for sensitive data

## Requirements

- Docker Desktop with NVIDIA Container Toolkit
- VS Code with Dev Containers extension
- NVIDIA GPU with compatible drivers

## Troubleshooting

### CUDA 13.0 Issues

If you encounter CUDA issues:

1. Verify your host NVIDIA driver supports CUDA 13.0 (requires 530+ driver)
2. Check that the NVIDIA Container Toolkit is properly installed
3. Ensure Docker has GPU access enabled

### PyTorch Installation Issues

If PyTorch fails to install:

1. Check network connectivity to PyTorch CUDA 13.0 repository
2. Try clearing uv cache: `uv cache clean`
3. Rebuild container: Ctrl+Shift+P → "Dev Container: Rebuild Container"

## Contributing

This template is designed to be easily customizable for your specific deep learning projects.
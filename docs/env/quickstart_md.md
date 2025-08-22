# Quick Start Guide

Get up and running with the PyTorch development environment in under 10 minutes.

## Prerequisites

- **Docker Desktop** (with GPU support if available)
- **VS Code** with the Dev Containers extension
- **Git** for version control
- **4GB+ RAM** (8GB+ recommended)
- **NVIDIA GPU** (optional but recommended for deep learning)

## 🚀 Getting Started

### 1. Clone and Open

```bash
git clone <your-repo-url>
cd torch_starter
code .
```

### 2. Open in Dev Container

1. VS Code will prompt: **"Reopen in Container"** → Click **Reopen in Container**
2. Or press `Ctrl+Shift+P` → Type: **"Dev Container: Reopen in Container"**
3. Wait 5-10 minutes for first-time build (subsequent starts: ~30 seconds)

### 3. Verify Installation

```bash
# In the VS Code terminal:
./dev.sh verify-setup
```

Expected output:
```
🎉 Environment Verification Summary
==================================================
✅ Environment verification completed successfully!
```

### 4. Quick Test

```bash
# Test PyTorch with CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run sample code
python -c "
import torch
x = torch.randn(3, 3)
print('CPU tensor:', x)
if torch.cuda.is_available():
    x_gpu = x.cuda()
    print('GPU tensor:', x_gpu.device)
"
```

### 5. Start Jupyter (Optional)

```bash
./dev.sh jupyter
```

Navigate to the displayed URL (with secure token).

## 🎯 What You Get

- **Python 3.12** with optimized virtual environment
- **PyTorch** with CUDA 12.4 support
- **Transformers & Datasets** for NLP/ML
- **Jupyter Lab** for interactive development
- **Modern dev tools**: ruff, mypy, pytest
- **Pre-configured caching** for Hugging Face, PyTorch models

## 📁 Directory Structure

```
torch_starter/
├── src/                    # Your main source code
├── notebooks/              # Jupyter notebooks
├── data/                   # Datasets (auto-mounted)
├── tests/                  # Unit tests
├── configs/                # Configuration files
├── scripts/                # Utility scripts
└── docs/                   # Documentation
```

## 🔧 Essential Commands

```bash
# Environment management
./dev.sh sync              # Sync dependencies
./dev.sh clean             # Clean caches

# Code quality
./dev.sh format            # Format code
./dev.sh lint              # Check code quality
./dev.sh test              # Run tests
./dev.sh all-checks        # Run all quality checks

# Package management
./dev.sh add-temp numpy    # Add package temporarily
./dev.sh add-perm wandb    # Add package permanently

# Development
./dev.sh jupyter           # Start Jupyter Lab
./dev.sh benchmark         # Performance test
```

## ⚡ Performance Tips

- **Use GPU**: Verify CUDA with `torch.cuda.is_available()`
- **Enable Docker GPU**: Ensure `--gpus all` in Docker settings
- **Monitor resources**: Use `./dev.sh docker-stats`
- **Cache optimization**: Large models auto-cache to persistent volumes

## 🔒 API Keys Setup

```bash
# Interactive setup
./dev.sh setup-keys

# Or manually create .env.local
cp .env.example .env.local
# Edit with your actual keys
```

## 🚨 Troubleshooting

### Container won't start
```bash
# Check Docker is running
docker --version

# Clean Docker cache
./dev.sh docker-cleanup

# Rebuild container
Ctrl+Shift+P → "Dev Container: Rebuild Container"
```

### Import errors
```bash
# Resync dependencies
./dev.sh sync

# Clear Python caches
./dev.sh clean
```

### CUDA not available
- Ensure Docker Desktop has GPU support enabled
- Check: Settings → Resources → WSL Integration (Windows)
- Verify with: `docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu20.04 nvidia-smi`

## 📚 Next Steps

- Read [Template Structure](template_structure.md) for project organization
- Check [Environment Guide](environment.md) for detailed configuration
- See [Development Workflow](dev.md) for best practices
- Browse [Jupyter Setup](jupyter_setup_guide.md) for notebook configuration

## 🆘 Need Help?

- Check [Troubleshooting Guide](troubleshooting.md)
- Review [Performance Tips](performance_tips.md)
- Open an issue on GitHub
- Check Docker Desktop logs

---

**Estimated setup time**: 5-10 minutes (first time), 30 seconds (subsequent starts)

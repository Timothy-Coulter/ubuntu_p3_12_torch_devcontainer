# Environment Guide

Comprehensive guide to the development environment configuration, including dependencies, caching, and optimization strategies.

## 🐍 Python Environment

### Python Version
- **Version**: Python 3.12.x (latest stable)
- **Rationale**: 
  - Performance improvements over 3.11
  - Modern syntax features
  - Excellent PyTorch compatibility
  - Strong typing support

### Package Management
- **Tool**: [uv](https://astral.sh/uv) - Ultra-fast Python package installer
- **Benefits**:
  - 10-100x faster than pip
  - Deterministic dependency resolution
  - Built-in virtual environment management
  - Compatible with pip and PyPI

### Virtual Environment
```bash
# Location: .venv/
# Activation: Automatic in dev container
# Manual activation: source .venv/bin/activate
```

## 📦 Dependencies

### Core Scientific Stack
```toml
[project.dependencies]
numpy = ">=1.26,<2.0"          # Numerical computing
pandas = ">=2.0,<3.0"          # Data manipulation  
scipy = ">=1.11,<2.0"          # Scientific algorithms
scikit-learn = ">=1.3,<2.0"    # Machine learning basics
matplotlib = ">=3.8,<4.0"      # Basic plotting
```

### Deep Learning Stack
```toml
# PyTorch ecosystem
torch = ">=2.0,<3.0"           # Core PyTorch
torchvision = ">=0.15,<1.0"    # Computer vision
torchaudio = ">=2.0,<3.0"      # Audio processing

# Hugging Face ecosystem  
transformers = ">=4.42,<5.0"   # Transformer models
datasets = ">=2.19,<3.0"       # ML datasets
accelerate = ">=0.31,<1.0"     # Distributed training
safetensors = ">=0.4,<1.0"     # Safe tensor serialization
```

### Development Tools
```toml
[project.optional-dependencies.dev]
ruff = ">=0.5.3,<1.0"          # Fast linting/formatting
mypy = ">=1.10,<2.0"           # Type checking
pytest = ">=8.2,<9.0"          # Testing framework
pytest-cov = ">=5.0,<6.0"      # Coverage reporting
```

### Optional Groups
- **`datascience`**: Extended data science tools (seaborn, plotly)
- **`notebook`**: Jupyter Lab and extensions
- **`tracking`**: Experiment tracking (wandb, tensorboard)
- **`profiling`**: Performance analysis tools
- **`security`**: Security scanning tools

## 🏗️ Container Architecture

### Base Image
- **Image**: `mcr.microsoft.com/vscode/devcontainers/base:ubuntu-24.04`
- **Rationale**: 
  - Official Microsoft dev container base
  - Ubuntu 24.04 LTS for stability
  - Pre-configured for VS Code integration

### Multi-Stage Build
```dockerfile
# Stage 1: Base dependencies and system packages
FROM ubuntu:24.04 AS base-deps

# Stage 2: Python environment setup  
FROM base-deps AS python-env

# Stage 3: Dependency installation with caching
FROM python-env AS deps-install

# Stage 4: Final runtime image
FROM deps-install AS final
```

### Optimization Features
- **BuildKit caching**: Aggressive layer caching
- **Mount caches**: uv and apt package caches
- **Multi-arch support**: AMD64 and ARM64 
- **Minimal final image**: Only runtime dependencies

## 💾 Caching Strategy

### Persistent Volumes
```yaml
volumes:
  torch-starter-hf-cache:     # Hugging Face models/datasets
  torch-starter-torch-cache:  # PyTorch models
  torch-starter-kaggle-cache: # Kaggle credentials
  torch-starter-data:         # Project datasets
  torch-starter-uv-cache:     # Python packages
```

### Cache Directories
| Cache Type | Location | Purpose |
|------------|----------|---------|
| Hugging Face | `/home/vscode/.cache/huggingface` | Models, datasets, tokenizers |
| PyTorch | `/home/vscode/.cache/torch` | Pre-trained models |
| uv | `/home/vscode/.cache/uv` | Package downloads |
| Kaggle | `/home/vscode/.kaggle` | API credentials |
| Project Data | `/workspaces/torch_starter/data` | Datasets, outputs |

### Cache Benefits
- **Faster rebuilds**: Dependencies cached between sessions
- **Offline capability**: Previously downloaded models available offline
- **Team sharing**: Shared cache volumes reduce download times
- **Cost savings**: Reduced bandwidth usage

## 🔧 Environment Variables

### Core Python Settings
```bash
PYTHONDONTWRITEBYTECODE=1     # Skip .pyc files
PYTHONUNBUFFERED=1            # Immediate stdout/stderr
PYTHONFAULTHANDLER=1          # Better error traces
PYTHONHASHSEED=random         # Randomized hash seeds
```

### Performance Tuning
```bash
# NumPy/BLAS threading
OPENBLAS_NUM_THREADS=4
MKL_NUM_THREADS=4
NUMEXPR_NUM_THREADS=4
OMP_NUM_THREADS=4

# PyTorch settings
TORCH_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=all
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### ML Framework Caches
```bash
HF_HOME=/home/vscode/.cache/huggingface
TRANSFORMERS_CACHE=/home/vscode/.cache/huggingface
TORCH_HOME=/home/vscode/.cache/torch
KAGGLE_CONFIG_DIR=/home/vscode/.kaggle
```

### uv Configuration
```bash
UV_LINK_MODE=copy             # Copy instead of symlink
UV_COMPILE_BYTECODE=1         # Pre-compile Python files
UV_CONCURRENT_DOWNLOADS=10    # Parallel downloads
UV_HTTP_TIMEOUT=300           # Extended timeout
```

## 🎯 GPU Support

### CUDA Configuration
- **Version**: CUDA 12.4
- **Driver compatibility**: 525+ (automatic in Docker)
- **Memory management**: Automatic mixed precision support
- **Multi-GPU**: Full support with proper device mapping

### GPU Runtime Options
```yaml
runtime: nvidia  # Docker nvidia-container-runtime
gpus: all       # All available GPUs
shm_size: 4gb   # Large shared memory for data loading
ulimits:
  memlock: -1   # Unlimited locked memory
```

### Verification
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# List GPU devices  
python -c "import torch; print(torch.cuda.device_count())"

# GPU memory info
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

## 📊 Performance Monitoring

### Resource Limits
```yaml
memory: 12gb        # Container memory limit
cpus: 8.0          # CPU cores
shm_size: 4gb      # Shared memory
ulimits:
  nofile: 65536    # File descriptors
  stack: 67108864  # Stack size
```

### Monitoring Tools
```bash
# Container stats
./dev.sh docker-stats

# System resources
htop

# GPU utilization
nvidia-smi

# Python profiling
./dev.sh profile script.py
./dev.sh memory-profile script.py
```

## 🔒 Security Configuration

### Container Security
```yaml
security_opt:
  - seccomp:unconfined    # Allow debugging
  - apparmor:unconfined   # Disable AppArmor restrictions

cap_add:
  - SYS_PTRACE           # Enable debugging
  - SYS_ADMIN            # Administrative capabilities
```

### File Permissions
- **API keys**: 600 (owner read/write only)
- **Cache directories**: 755 (owner full, group/other read)
- **Virtual environment**: 755 (shared read access)

### Network Security
- **Jupyter**: Secure tokens by default
- **API endpoints**: HTTPS only
- **Credentials**: Environment variables, not hardcoded

## 🔄 Environment Lifecycle

### Initial Setup (First Run)
1. **Container build**: ~10-15 minutes
2. **Python environment**: ~2-3 minutes  
3. **Dependency installation**: ~3-5 minutes
4. **Cache warmup**: As needed

### Daily Development
1. **Container start**: ~15-30 seconds
2. **Environment activation**: Automatic
3. **Cache hits**: ~90%+ for common models

### Maintenance
```bash
# Update dependencies
./dev.sh lock-update

# Clean caches  
./dev.sh clean

# Rebuild container
# VS Code: Ctrl+Shift+P → "Dev Container: Rebuild Container"

# Optimize performance
./dev.sh optimize
```

## 🚀 Optimization Tips

### Build Performance
- **Layer ordering**: Put frequently changing files last
- **Cache mounts**: Use for package downloads
- **Multi-stage**: Separate build and runtime dependencies
- **Parallel builds**: Enable BuildKit parallel processing

### Runtime Performance  
- **Memory mapping**: Use memory-mapped files for large datasets
- **Data loading**: Optimize `num_workers` for your CPU count
- **GPU memory**: Use gradient checkpointing for large models
- **Compilation**: JIT compilation with `torch.compile()`

### Development Workflow
- **Hot reloading**: Use `%autoreload` in Jupyter
- **Incremental testing**: Run only changed tests
- **Background tasks**: Use tmux/screen for long-running processes
- **Resource monitoring**: Keep `htop`/`nvidia-smi` open

## 📝 Environment Validation

### Automated Checks
```bash
./dev.sh verify-setup      # Full environment validation
./dev.sh benchmark         # Performance testing
./dev.sh all-checks       # Code quality validation
```

### Manual Verification
```bash
# Python and packages
python --version
pip list | grep torch
python -c "import torch; print(torch.__version__)"

# GPU support
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Cache directories
ls -la ~/.cache/
df -h /workspaces/torch_starter/data/
```

This environment provides a **production-ready**, **high-performance** foundation for PyTorch development with **enterprise-grade** optimization and security features.

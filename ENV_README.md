# Environment README

This document provides comprehensive documentation for the **torch_starter** deep learning development environment. This template facilitates rapid setup for deep learning projects with containerized development, GPU support, and modern Python tooling.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop with WSL2 (Windows) or Docker Engine (Linux/macOS)
- VS Code with Dev Containers extension
- NVIDIA GPU + drivers (optional, for CUDA acceleration)
- Git

### Getting Started
1. **Clone and Open**: Clone this repository and open in VS Code
2. **Dev Container**: VS Code will prompt to "Reopen in Container" - click it
3. **Initial Build**: First build takes ~15-20 minutes (packages pre-installed)
4. **Lightning Fast Startup**: Subsequent container starts in ~30 seconds! 🚀
5. **Verify Environment**: Run `bash ./dev.sh verify-setup`
6. **Setup API Keys** (optional): Run `bash ./dev.sh setup-keys`

### First Steps
```bash
# Verify everything works
./dev.sh verify-setup

# Run sample test
./dev.sh test

# Start developing in src/ folder
# Create notebooks in notebooks/ folder
# Add scripts in scripts/ folder
```

---

## ⚡ Performance Optimization

This environment uses an **optimized build strategy** for dramatically faster development:

| **Metric** | **Before** | **Optimized** | **Improvement** |
|------------|------------|---------------|-----------------|
| **Daily Startup** | 5-10 minutes | ~30 seconds | **10x faster** |
| **First Build** | ~5 minutes | ~15-20 minutes | Slower (one-time) |
| **Package Addition** | ~2 minutes | ~30 sec (temp) | Same/faster |
| **Team Onboarding** | 5-10 min per startup | ~30 sec per startup | **Consistent** |

### How It Works
- **Build-time Installation**: All packages pre-installed during container build
- **Runtime Activation**: Virtual environment instantly available
- **Hybrid Package Management**: Temporary additions for experiments, permanent for team
- **Cached Layers**: Docker optimizations for faster rebuilds

---

## 📋 Environment Specifications

### Container Environment
- **Base**: Ubuntu 24.04 LTS
- **Python**: 3.12.x (managed by uv)
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (faster pip/conda alternative)
- **CUDA**: 12.4 (if GPU available)
- **GPU Support**: NVIDIA Docker runtime with `--gpus all`
- **🚀 Optimization**: Pre-built packages for instant startup

### Core Dependencies
- **Deep Learning**: PyTorch, torchvision, torchaudio (CUDA 12.4)
- **ML Libraries**: transformers, datasets, accelerate, scikit-learn
- **Data Science**: numpy, pandas, scipy, matplotlib, seaborn
- **Development**: jupyter, jupyterlab, ipykernel
- **Code Quality**: ruff, mypy, pytest, pre-commit

### Development Tools
- **Linting**: Ruff (replaces flake8, isort, black)
- **Type Checking**: MyPy (strict mode)
- **Testing**: pytest with coverage
- **Formatting**: Ruff formatter
- **Git Hooks**: pre-commit configuration
- **AI Assistant**: Roo (Claude integration)

---

## 📁 Project Structure

```
├── src/                    # Main source code directory
├── notebooks/              # Jupyter notebooks for demos/experiments  
├── scripts/                # Utility scripts (not main codebase)
├── tests/                  # Test files (pytest)
├── configs/                # Configuration files
│   └── data_paths.example.yaml  # Data path templates
├── torch_starter/          # Sample package structure
│   └── core/               # Core modules
├── verify_setup/           # Environment verification scripts
├── .devcontainer/          # Dev container configuration
├── .vscode/                # VS Code settings and launch configs
├── .roo/                   # Roo AI assistant configuration
├── .mcp/                   # MCP (Model Context Protocol) servers
├── .dev.sh                 # Developer utility script
└── pyproject.toml          # Project configuration and dependencies
```

### Folder Guidelines
- **`src/`**: Your main application code and modules
- **`notebooks/`**: Jupyter notebooks for exploration, demos, documentation
- **`scripts/`**: One-off scripts, data processing, utilities
- **`configs/`**: Configuration files, hyperparameters, data paths
- **`tests/`**: Unit tests, integration tests (use pytest)

---

## 🛠️ Development Workflow

### Environment Management (Optimized)
The environment is now **pre-built** for lightning-fast startup! ⚡

```bash
# Environment is automatically activated on container start
# All packages pre-installed during build

# For manual activation (if needed)
source .venv/bin/activate

# Sync base dependencies (rarely needed)
./dev.sh sync

# Update lock file and rebuild container
./dev.sh lock-update && rebuild container
```

### Package Management (New Optimized Workflow)
Two-tier package management for flexibility:

#### Quick Experimentation (Temporary Packages)
```bash
# Add package for quick testing/debugging (available immediately)
./dev.sh add-temp ipdb
./dev.sh add-temp requests-oauthlib
./dev.sh add-temp plotly

# Packages available instantly but lost on container rebuild
```

#### Permanent Additions (Team Environment)
```bash
# Add to project permanently (requires rebuild)
./dev.sh add-perm scikit-image
./dev.sh add-perm wandb

# Alternative: manually edit pyproject.toml, then:
./dev.sh rebuild-image  # Instructions for rebuilding
```

#### Package Management Commands
```bash
./dev.sh add-temp <package>   # Temporary (~30 seconds)
./dev.sh add-perm <package>   # Permanent (needs rebuild)
./dev.sh sync-temp           # Sync temporary changes only
./dev.sh rebuild-image       # Rebuild instructions
```

### Code Quality
```bash
# Format code
./dev.sh format

# Run linter
./dev.sh lint

# Fix auto-fixable lint issues
./dev.sh lint-fix

# Fix import ordering
./dev.sh fix-imports

# Run type checker
./dev.sh typecheck

# Run tests
./dev.sh test

# Run all quality checks
./dev.sh all-checks
```

### Data Science Workflow
```bash
# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Run notebooks with correct kernel
# Use "Python 3.12 (torch_starter)" kernel in Jupyter

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🗂️ Cache Management & Data Handling

### Cache Directories
The environment automatically mounts host caches to avoid re-downloading:

| Cache Type | Container Path | Host Path (Windows) |
|------------|----------------|---------------------|
| Hugging Face | `/home/vscode/.cache/huggingface` | `%USERPROFILE%\.cache\huggingface` |
| PyTorch | `/home/vscode/.cache/torch` | `%USERPROFILE%\.cache\torch` |
| Kaggle | `/home/vscode/.kaggle` | `%USERPROFILE%\.kaggle` |
| Data | `/workspaces/*/data` | `%USERPROFILE%\data` |

### Data Path Configuration
Copy [`configs/data_paths.example.yaml`](configs/data_paths.example.yaml) to `configs/data_paths.yaml` and customize:

```yaml
# configs/data_paths.yaml
defaults:
  data_root: ${DATA_DIR:-${PWD}/data}
  
datasets:
  my_dataset:
    path: ${DATA_DIR}/my_dataset
    
models:
  checkpoints: ${DATA_DIR}/checkpoints
```

### Environment Variables
```bash
# Automatically set in container
DATA_DIR=/workspaces/${PROJECT_NAME}/data
HF_HOME=/home/vscode/.cache/huggingface
TORCH_HOME=/home/vscode/.cache/torch
KAGGLE_CONFIG_DIR=/home/vscode/.kaggle
```

---

## 🔐 API Keys & Secrets Management

### Automated Setup
```bash
# Interactive API key setup
./.dev.sh setup-keys
```

This script prompts for and securely stores:
- **Hugging Face**: Token for model downloads
- **Kaggle**: Username and API key for datasets
- **Weights & Biases**: API key for experiment tracking
- **OpenAI**: API key for LLM integrations

### Manual Setup
Create [`.env.local`](.env.local) (gitignored) in project root:

```bash
# .env.local
HUGGINGFACE_TOKEN=hf_your_token_here
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
WANDB_API_KEY=your_wandb_key
OPENAI_API_KEY=sk-your_openai_key
```

### Usage in Code
```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

# Access API keys
hf_token = os.getenv('HUGGINGFACE_TOKEN')
wandb_key = os.getenv('WANDB_API_KEY')
```

---

## 🔧 VS Code Integration

### Extensions (Auto-installed)
- **Python**: Python language support and debugging
- **Pylance**: Advanced Python language server
- **Jupyter**: Notebook support in VS Code
- **Ruff**: Fast Python linting and formatting
- **Docker**: Container management
- **GitLens**: Enhanced Git integration
- **Roo**: AI-powered coding assistant

### Debugging Configuration
Pre-configured debug settings in [`.vscode/launch.json`](.vscode/launch.json):

- **Python: Current File**: Debug active Python file
- **Pytest: All tests**: Debug test suite

### Settings Highlights
- **Auto-formatting**: On save with Ruff
- **Import organization**: Automatic with Ruff
- **Type checking**: Real-time with Pylance + MyPy
- **Jupyter kernel**: `torch_starter-3.12` auto-selected
- **File exclusions**: Hide cache/build directories

---

## 🔍 Verification & Troubleshooting

### Environment Verification
```bash
# Comprehensive environment check
./dev.sh verify-setup
```

This verifies:
- ✅ Python 3.12 installation
- ✅ Package imports and versions
- ✅ PyTorch + CUDA availability
- ✅ Cache directory permissions
- ✅ GPU runtime (nvidia-smi)

### Common Issues & Solutions

#### CUDA Not Available
```bash
# Check GPU passthrough
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4-runtime-ubuntu22.04 nvidia-smi

# Rebuild container after driver updates
# Dev Container: Rebuild Container (Ctrl+Shift+P)
```

#### Import Errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Re-sync dependencies
./dev.sh sync

# Clear caches
./dev.sh clean
```

#### VS Code Issues
```bash
# Reset Python interpreter
# Ctrl+Shift+P -> "Python: Select Interpreter"
# Choose: ./venv/bin/python

# Reload window
# Ctrl+Shift+P -> "Developer: Reload Window"
```

#### Container Build Failures
```bash
# Clean Docker cache
docker system prune -f

# Rebuild without cache (15-20 min for optimized build)
# Dev Container: Rebuild Container (No Cache)
```

#### Slow Container Startup (Optimized Build)
```bash
# Container should start in ~30 seconds with pre-built packages
# If taking longer, packages may not be pre-installed:

# Check if packages are built into image
docker images | grep torch_starter

# If container is slow, rebuild to get optimized version
# Dev Container: Rebuild Container
```

#### Adding Packages
```bash
# Quick temporary addition (immediate)
./dev.sh add-temp package-name

# Permanent addition (requires rebuild)
./dev.sh add-perm package-name
# Then: Dev Container: Rebuild Container

# Package missing after container restart?
# Temporary packages are lost - add permanently:
# Edit pyproject.toml and rebuild container
```

---

## 🚀 Common Workflows

### Starting a New Project (Optimized)
1. **Structure**: Create modules in `src/`, experiments in `notebooks/`
2. **Dependencies**:
   - Quick experiments: `./dev.sh add-temp package-name`
   - Permanent deps: Edit [`pyproject.toml`](pyproject.toml) → rebuild container
3. **Data**: Configure paths in `configs/data_paths.yaml`
4. **Tests**: Write tests in `tests/` directory
5. **Quality**: Run `./.dev.sh all-checks` regularly

### Package Development Workflow
```bash
# 1. Experiment with packages
./dev.sh add-temp wandb
./dev.sh add-temp plotly
# Test and develop...

# 2. Commit useful packages permanently
./dev.sh add-perm wandb  # Adds to pyproject.toml
./dev.sh add-perm plotly

# 3. Rebuild container for team
# Dev Container: Rebuild Container (15-20 min)

# 4. Team gets pre-built environment
# Next container start: ~30 seconds!
```

### Training a Model
```bash
# 1. Prepare data (in notebooks/ or scripts/)
jupyter lab

# 2. Implement model (in src/)
# Edit src/models.py, src/training.py, etc.

# 3. Configure experiment
# Edit configs/experiment.yaml

# 4. Run training
python src/train.py --config configs/experiment.yaml

# 5. Monitor with tensorboard
tensorboard --logdir logs/
```

### Experiment Tracking
```python
# With Weights & Biases
import wandb

wandb.init(
    project="my-project",
    config={"lr": 0.001, "epochs": 10}
)

# Log metrics
wandb.log({"loss": loss, "accuracy": acc})
```

### Packaging and Distribution
```bash
# Build package
uv build

# Install locally
uv pip install -e .

# Run tests before release
./dev.sh all-checks
```

---

## 📚 Additional Resources

### Package Management with uv
- [uv Documentation](https://github.com/astral-sh/uv)
- Fast pip replacement with better dependency resolution
- Commands: `uv add`, `uv remove`, `uv sync`, `uv lock`

### Roo AI Assistant
- Integrated Claude-powered coding assistant
- Access via Ctrl+Shift+P -> "Roo" commands
- Configuration in [`.roo/roo.json`](.roo/roo.json)

### PyTorch Ecosystem
- Models download to `HF_HOME` cache (persistent)
- CUDA 12.4 wheels from PyTorch index
- Automatic mixed precision with `accelerate`

### Development Best Practices
- **Type Hints**: Use throughout codebase (enforced by MyPy)
- **Testing**: Write tests for core logic in `tests/`
- **Documentation**: Use docstrings and inline comments
- **Git Hooks**: Pre-commit runs formatting and linting
- **Reproducibility**: Pin versions in `uv.lock`

---

## 🆘 Support & Contributing

### Getting Help
1. **Environment Issues**: Run `./.dev.sh verify-setup`
2. **VS Code Problems**: Check Output panel for errors
3. **Package Issues**: Check `uv.lock` and re-sync
4. **GPU Issues**: Verify Docker GPU runtime setup

### Contributing
1. **Format Code**: `./.dev.sh format`
2. **Run Tests**: `./.dev.sh test`
3. **Type Check**: `./.dev.sh typecheck`
4. **All Checks**: `./.dev.sh all-checks`

### Updating Template
```bash
# Update dependencies
./dev.sh lock-update

# Update container
# Dev Container: Rebuild Container

# Commit changes
git add . && git commit -m "Update environment"
```

---

*This environment template provides a robust foundation for deep learning development with modern tooling, containerization, and AI assistance. Happy coding! 🚀*
---

## 🔄 Migration to Optimized Environment

This environment has been **optimized for 10x faster daily startup** by moving package installation from runtime to build time.

### What Changed
- **Dockerfile**: Now pre-installs all packages during build (~15-20 min one-time)
- **devcontainer.json**: Minimal startup command (~30 seconds)
- **dev.sh**: New package management commands for hybrid workflow
- **Workflow**: Two-tier package management (temporary vs permanent)

### Upgrading Your Environment
```bash
# 1. Backup current work
git add . && git commit -m "Backup before optimization"

# 2. Rebuild container with optimized configuration
# Dev Container: Rebuild Container (Ctrl+Shift+P)
# First build: ~15-20 minutes (downloads packages)

# 3. Verify optimization
./dev.sh verify-setup
# Should show instant environment activation!

# 4. Future startups: ~30 seconds instead of 5-10 minutes! 🚀
```

### New Package Workflow
- **Experimentation**: `./dev.sh add-temp package` (instant)
- **Production**: Edit `pyproject.toml` + rebuild (permanent)
- **Team Sync**: Rebuild after dependency changes

### Rollback (if needed)
```bash
# Revert to previous devcontainer configuration
git checkout HEAD~1 -- .devcontainer/
# Dev Container: Rebuild Container
```

---

*Environment optimized for developer productivity - enjoy the speed boost! 🚀*
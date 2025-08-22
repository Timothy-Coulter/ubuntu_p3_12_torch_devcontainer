# Template Structure Guide

Understanding the organization and purpose of each directory and file in the PyTorch development template.

## 📁 Root Directory Structure

```
torch_starter/
├── .devcontainer/          # Dev Container configuration
├── .github/                # GitHub workflows (if added)
├── .roo/                   # Roo development tool config
├── .mcp/                   # MCP (Model Context Protocol) config
├── .vscode/                # VS Code settings
├── configs/                # Application configuration files
├── data/                   # Datasets and data files
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility and automation scripts
├── src/                    # Alternative source location
├── templates/              # File templates and examples
├── tests/                  # Unit and integration tests
├── torch_starter/          # Main Python package
├── verify_setup/           # Setup verification scripts
├── dev.sh                  # Main development script
├── pyproject.toml          # Python project configuration
└── README.md               # Project overview
```

## 🐳 Dev Container Configuration

### `.devcontainer/`
- **`Dockerfile`**: Multi-stage optimized container build
  - Base: Ubuntu 24.04 with VS Code integration
  - Python 3.12 with uv package manager
  - PyTorch with CUDA 12.4 support
  - Optimized for caching and performance

- **`devcontainer.json`**: Container runtime configuration
  - GPU passthrough (`--gpus all`)
  - Volume mounts for persistent caches
  - Environment variables
  - VS Code extensions and settings

### Key Features:
- **Multi-stage build**: Optimizes Docker layer caching
- **Volume mounts**: Persistent caches for Hugging Face, PyTorch, etc.
- **Performance tuning**: Memory limits, shared memory configuration
- **Security**: Non-root user, proper permissions

## 🔧 Configuration Files

### Core Configuration
- **`pyproject.toml`**: Python project metadata and tool configuration
  - Dependencies with version constraints
  - Development tools (ruff, mypy, pytest)
  - Optional dependency groups
  - Tool configurations

- **`.gitignore`**: Git exclusions
  - Python artifacts (`__pycache__`, `.pyc`)
  - Virtual environments
  - Data and model files
  - IDE configurations
  - Sensitive files (`.env.local`)

- **`.gitattributes`**: Git file handling
  - Line ending normalization
  - Binary file detection
  - Shell script permissions

### Development Tools
- **`.pre-commit-config.yaml`**: Pre-commit hooks
  - Code formatting (ruff, black)
  - Quality checks
  - Security scanning

- **`.vscode/settings.json`**: VS Code configuration
  - Python interpreter path
  - Linting and formatting on save
  - Test framework configuration
  - File exclusions for performance

## 📦 Python Package Structure

### `torch_starter/` (Main Package)
```
torch_starter/
├── __init__.py             # Package initialization
├── core/                   # Core functionality
│   ├── __init__.py
│   └── sample.py           # Sample module
├── models/                 # ML model definitions
├── data/                   # Data loading and processing
├── training/               # Training loops and utilities
├── inference/              # Inference and evaluation
└── utils/                  # Common utilities
```

### Design Principles:
- **Modular structure**: Separate concerns (models, data, training)
- **Clear imports**: Explicit `__all__` declarations
- **Type hints**: Full type annotation support
- **Testable**: Each module has corresponding tests

## 📊 Data and Configuration

### `data/`
- **Purpose**: Datasets, model weights, experiment outputs
- **Mount**: Persistent Docker volume to host filesystem
- **Structure**:
  ```
  data/
  ├── raw/                  # Original datasets
  ├── processed/            # Cleaned/preprocessed data
  ├── models/               # Model checkpoints
  ├── outputs/              # Experiment results
  └── cache/                # Temporary cache files
  ```

### `configs/`
- **Purpose**: Application configuration files
- **Examples**:
  - `data_paths.yaml`: Dataset and model path registry
  - `model_configs/`: Model hyperparameters
  - `training_configs/`: Training configurations

## 📝 Development and Testing

### `tests/`
- **Structure**: Mirrors main package structure
- **Framework**: pytest with coverage reporting
- **Types**:
  - Unit tests: Individual functions/classes
  - Integration tests: End-to-end workflows
  - Performance tests: Benchmarking

### `scripts/`
- **Purpose**: Standalone utility scripts
- **Examples**:
  - Data preprocessing
  - Model training scripts
  - Evaluation pipelines
  - Deployment utilities

## 📚 Documentation Structure

### `docs/`
```
docs/
├── env/                    # Environment setup guides
├── src/                    # Source code documentation
├── examples/               # Usage examples
├── tutorials/              # Step-by-step tutorials
└── api/                    # API reference
```

## 🔄 Development Workflow

### `dev.sh` Script
Central development automation with commands:
- **Environment**: `sync`, `clean`, `verify-setup`
- **Code Quality**: `format`, `lint`, `test`, `all-checks`
- **Packages**: `add-temp`, `add-perm`, `remove`
- **Development**: `jupyter`, `benchmark`, `profile`

### `.roo/` Configuration
Integration with Roo development tool:
- **Scripts**: Predefined command shortcuts
- **Project metadata**: Name, preferred mode
- **MCP integration**: Model Context Protocol support

## 🛠️ Specialized Directories

### `templates/`
- **Purpose**: File templates and examples
- **Contents**:
  - `.env.example`: Environment variable template
  - Configuration file templates
  - Code templates for common patterns

### `verify_setup/`
- **Purpose**: Environment verification and setup
- **Scripts**:
  - `verify_setup.sh`: Comprehensive environment validation
  - `setup_api_keys.sh`: Interactive API key configuration

### `notebooks/`
- **Purpose**: Jupyter notebooks for exploration and prototyping
- **Organization**: By project phase or topic
- **Integration**: Automatic kernel registration with project environment

## 🔒 Security Considerations

### Sensitive Files
- **`.env.local`**: Never committed, contains API keys
- **`~/.kaggle/kaggle.json`**: Secure permissions (600)
- **Cache directories**: User-only access in container

### Best Practices
- **Separate environments**: Development vs. production configs
- **Key rotation**: Regular API key updates
- **Permission management**: Minimal required permissions
- **Audit trails**: Log access to sensitive resources

## 🎯 Customization Points

### Common Modifications
1. **Package name**: Update `torch_starter` references
2. **Dependencies**: Modify `pyproject.toml` dependency groups
3. **Container base**: Change Ubuntu version or base image
4. **Python version**: Update from 3.12 to desired version
5. **GPU support**: Modify CUDA version for compatibility

### Extension Areas
1. **CI/CD**: Add `.github/workflows/` for automation
2. **Monitoring**: Add logging and metrics collection
3. **Deployment**: Add Docker production images
4. **Documentation**: Expand with project-specific guides

---

This structure balances **flexibility** with **convention**, providing a solid foundation for PyTorch development while allowing customization for specific project needs.

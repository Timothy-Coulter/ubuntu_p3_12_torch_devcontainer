🔥 torch-starter: CUDA-Enabled PyTorch DevContainer

A production-ready, CUDA-enabled PyTorch development environment using VS Code Dev Containers, UV package manager, and Python 3.12. Optimized for deep learning, transformers, and GPU-accelerated machine learning workflows.
✨ Features

    🚀 CUDA 12.1 Support: Pre-configured with NVIDIA CUDA 12.1 and cuDNN 8
    ⚡ Fast Package Management: UV for lightning-fast dependency resolution
    🐍 Python 3.12: Latest Python with performance improvements
    🤗 ML-Ready: Pre-installed PyTorch, Transformers, and essential ML libraries
    📊 Development Tools: Jupyter, VS Code extensions, debugging, and profiling
    🔧 Quality Assurance: Ruff, MyPy, Pytest with comprehensive configuration
    📦 Persistent Storage: Docker volumes for caches, data, and models
    🖥️ GPU Monitoring: Built-in tools for GPU utilization and performance tracking

🏗️ Architecture

📦 torch-starter/
├── .devcontainer/
│   ├── Dockerfile              # Multi-stage CUDA build
│   └── devcontainer.json       # VS Code configuration
├── torch_starter/              # Your project code
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks
├── data/                       # Datasets (mounted volume)
├── verify_setup/               # Environment verification scripts
├── pyproject.toml             # Project dependencies and configuration
└── README.md                  # This file

🚀 Quick Start
Prerequisites

    Docker: Latest version with BuildKit support
    VS Code: With Dev Containers extension
    NVIDIA GPU: With compatible drivers (optional but recommended)
    NVIDIA Container Toolkit: For GPU support in Docker

GPU Setup (Recommended)

    Install NVIDIA Container Toolkit:
    bash

    # Ubuntu/Debian
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker

    Verify GPU Access:
    bash

    docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi

Development Setup

    Clone and Open:
    bash

    git clone https://github.com/yourusername/torch-starter.git
    cd torch-starter
    code .

    Open in Dev Container:
        Press Ctrl+Shift+P (or Cmd+Shift+P on Mac)
        Type "Dev Containers: Reopen in Container"
        Wait for the container to build (first time takes ~10-15 minutes)
    Verify Setup:
    bash

    # Basic environment verification
    python verify_setup/verify_environment.py

    # Comprehensive GPU testing
    python verify_setup/test_gpu.py --verbose --benchmark

    # Quick PyTorch CUDA check
    python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"

🔧 Configuration
CUDA Version

The container uses CUDA 12.1 for optimal PyTorch compatibility. To change:

    Edit .devcontainer/Dockerfile:
    dockerfile

    FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS cuda-base

    Update .devcontainer/devcontainer.json:
    json

    "CUDA_VERSION": "12.1"

    Rebuild the container

Python Dependencies

Dependencies are managed in pyproject.toml with multiple optional groups:
bash

# Install development tools
uv sync --group dev

# Install data science packages
uv sync --group datascience

# Install everything
uv sync --group all

Available Dependency Groups

    dev: Linting, testing, formatting tools
    datascience: Advanced analytics and visualization
    notebook: Jupyter and interactive development
    tracking: MLOps and experiment tracking
    data: Data acquisition and external APIs
    profiling: Performance analysis and debugging
    nlp: Natural language processing extensions
    cv: Computer vision frameworks
    rl: Reinforcement learning
    llm: Large language models and text generation

🧪 Testing and Quality
Running Tests
bash

# Run all tests
pytest

# Run with coverage
pytest --cov=torch_starter --cov-report=html

# Run GPU-specific tests
pytest -m gpu

# Run without slow tests
pytest -m "not slow"

Code Quality
bash

# Format code
ruff format .

# Lint and fix
ruff check --fix .

# Type checking
mypy torch_starter/

# Security scanning
bandit -r torch_starter/

GPU Testing
bash

# Comprehensive GPU test suite
python verify_setup/test_gpu.py --verbose --benchmark

# Monitor GPU during development
gpu-monitor  # Alias for nvidia-smi watch

# Get GPU info
gpu-info     # Alias for detailed GPU information

📊 Development Workflow
Jupyter Integration
bash

# Start JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# The kernel "torch_starter-3.12" is automatically available

GPU Monitoring

Built-in aliases for GPU monitoring:
bash

gpu-monitor    # Watch GPU utilization in real-time
gpu-info       # Detailed GPU information
torch-info     # PyTorch CUDA configuration

Experiment Tracking
bash

# Install tracking dependencies
uv sync --group tracking

# Start TensorBoard
tensorboard --logdir=logs --host=0.0.0.0 --port=6006

# Start MLflow
mlflow ui --host=0.0.0.0 --port=5000

🐳 Docker Volumes

The container uses persistent Docker volumes for:

    torch-starter-hf-cache: Hugging Face models and datasets
    torch-starter-torch-cache: PyTorch models and weights
    torch-starter-data: Your datasets and data files
    torch-starter-models: Saved models and checkpoints
    torch-starter-uv-cache: UV package cache for faster rebuilds

🔍 Troubleshooting
CUDA Not Available

    Check GPU Detection:
    bash

    nvidia-smi

    Verify Docker GPU Access:
    bash

    docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi

    Check PyTorch Installation:
    bash

    python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

    Reinstall PyTorch CUDA:
    bash

    uv pip uninstall torch torchvision torchaudio
    uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

Container Build Issues

    Clear Docker Cache:
    bash

    docker system prune -a

    Rebuild Container:
    bash

    # In VS Code: Ctrl+Shift+P -> "Dev Containers: Rebuild Container"

    Check Available Memory:
        Ensure at least 16GB RAM available
        Increase Docker memory limit if needed

Permission Issues
bash

# Fix ownership of workspace files
sudo chown -R $USER:$USER /workspaces/torch-starter

# Reset Docker volumes (destructive)
docker volume rm torch-starter-data torch-starter-models

🤝 Contributing

    Fork the Repository
    Create Feature Branch: git checkout -b feature/your-feature
    Run Tests: pytest && python verify_setup/verify_environment.py
    Check Code Quality: ruff check . && mypy torch_starter/
    Commit Changes: git commit -m "Add your feature"
    Push Branch: git push origin feature/your-feature
    Create Pull Request

📋 Requirements
Minimum System Requirements

    CPU: 8+ cores recommended
    RAM: 16GB minimum, 32GB recommended
    Storage: 50GB free space
    GPU: NVIDIA GPU with CUDA Compute Capability 7.0+ (optional)

Compatible GPUs

    RTX 30/40 Series: RTX 3060, 3070, 3080, 3090, 4070, 4080, 4090
    RTX 20 Series: RTX 2060, 2070, 2080, 2080 Ti
    GTX 16 Series: GTX 1660, 1660 Ti (limited CUDA features)
    Professional: Quadro RTX series, Tesla V100, A100, H100

📚 Resources
Documentation

    PyTorch Documentation
    Transformers Documentation
    UV Documentation
    VS Code Dev Containers

Tutorials and Examples

    Basic PyTorch: See notebooks/01_pytorch_basics.ipynb
    Transformers: See notebooks/02_transformers_intro.ipynb
    GPU Optimization: See notebooks/03_gpu_optimization.ipynb
    Custom Models: See notebooks/04_custom_models.ipynb

Community

    PyTorch Community
    Hugging Face Community
    Discord Server (if applicable)

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    NVIDIA for CUDA and GPU computing support
    PyTorch Team for the amazing framework
    Hugging Face for democratizing NLP
    Astral for UV package manager
    VS Code Team for Dev Containers

<div align="center">

Happy Deep Learning! 🚀🔥

</div>

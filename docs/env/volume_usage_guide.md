# Volume-Based PyTorch Development Setup

## Quick Start

### Option 1: Automatic Repository Cloning
1. **Update the clone URL** in `devcontainer.json`:
   ```json
   "clone-and-setup": "bash -c 'if [ ! -f \"/workspaces/torch-starter/pyproject.toml\" ]; then echo \"🔄 Repository not found, cloning...\"; cd /workspaces && rm -rf torch-starter/* torch-starter/.[!.]* 2>/dev/null || true; git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /tmp/torch-starter && cp -r /tmp/torch-starter/* /tmp/torch-starter/.[!.]* /workspaces/torch-starter/ 2>/dev/null || true && rm -rf /tmp/torch-starter; fi'"
   ```

2. **Open in VS Code**:
   - Use "Dev Containers: Clone Repository in Container Volume"
   - Select your repository
   - Choose the devcontainer configuration

### Option 2: Manual Setup
1. **Create the volume**:
   ```bash
   docker volume create torch-starter-workspace
   ```

2. **Open VS Code Dev Container** with volume mount

3. **Clone your repository** inside the container:
   ```bash
   cd /workspaces/torch-starter
   git clone https://github.com/your-username/your-repo.git .
   ```

4. **Install project**:
   ```bash
   install-project
   # or manually:
   uv pip install --system -e .
   ```

## Key Benefits

### ✅ **Persistent Development Environment**
- All code, dependencies, and caches persist across container restarts
- No need to rebuild or reinstall dependencies
- Git history and branches maintained

### ✅ **Fast Startup**
- Container starts immediately (no build required after first setup)
- PyTorch and CUDA pre-installed and verified
- UV package manager ready to use

### ✅ **Optimized for GPU Development**
- CUDA 12.9 support with PyTorch 2.8.0
- GPU monitoring tools included
- Optimized memory settings for ML workloads

### ✅ **VS Code Integration**
- Jupyter kernels auto-configured
- Python interpreter correctly set
- Extensions and settings optimized for ML development

## Available Commands

| Command | Description |
|---------|-------------|
| `uv-install <package>` | Install Python packages with UV |
| `uv-list` | List all installed packages |
| `uv-status` | Show UV package management status |
| `install-project` | Install current project with dependencies |
| `gpu-monitor` | Real-time GPU utilization monitoring |
| `gpu-info` | Show GPU specifications |
| `torch-info` | Display PyTorch and CUDA information |

## File Structure

```
/workspaces/torch-starter/          # Your project workspace (persistent)
├── .devcontainer/                  # DevContainer configuration
│   ├── devcontainer.json          # Volume-optimized config
│   └── Dockerfile.volume          # Volume-optimized Dockerfile
├── torch_starter/                  # Your Python package
├── tests/                          # Test files
├── notebooks/                      # Jupyter notebooks
├── data/                          # Data files (mounted volume)
├── pyproject.toml                 # Project configuration
└── README.md                      # Project documentation

/home/ubuntu/.cache/               # Cached data (persistent volumes)
├── huggingface/                   # HuggingFace models and datasets
├── torch/                         # PyTorch model cache
├── uv/                           # UV package cache
└── models/                       # Additional model cache
```

## Environment Variables

The container sets up these key environment variables:

```bash
# Python and UV Configuration
UV_SYSTEM_PYTHON=1                # Use system Python
UV_PYTHON_DOWNLOADS=never         # Don't download Python
PYTHONPATH=/workspaces/torch-starter

# CUDA Configuration
CUDA_VISIBLE_DEVICES=all
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6

# Cache Locations
HF_HOME=/home/ubuntu/.cache/huggingface
TORCH_HOME=/home/ubuntu/.cache/torch
UV_CACHE_DIR=/home/ubuntu/.cache/uv
```

## Troubleshooting

### Repository Not Auto-Cloned
```bash
# Manual clone if auto-clone fails
cd /workspaces/torch-starter
git clone https://github.com/your-username/your-repo.git .
install-project
```

### Dependencies Not Installing
```bash
# Check UV status
uv-status

# Reinstall project
cd /workspaces/torch-starter
uv pip install --system -e . --force-reinstall
```

### GPU Not Available
```bash
# Check GPU status
nvidia-smi
torch-info

# Verify CUDA in PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Jupyter Kernel Missing
```bash
# Reinstall kernel
python -m ipykernel install --user --name torch_starter-3.11 --display-name "Python 3.11 (torch_starter)"
```

## Performance Tips

1. **Use UV for package management** - Much faster than pip
2. **Monitor GPU memory** with `gpu-monitor` during training
3. **Leverage persistent volumes** - Models and datasets cached between sessions
4. **Use Jupyter notebooks** in the `notebooks/` directory for experimentation
5. **Enable GPU memory optimization** - Already configured in the environment

## Migration from Local Development

1. **Copy your existing project** to the volume workspace
2. **Update import paths** if needed (PYTHONPATH is set automatically)
3. **Install dependencies** with `install-project` or `uv-install`
4. **Update any absolute paths** to use the container workspace

This setup provides a robust, persistent, and high-performance PyTorch development environment optimized for GPU workloads and volume-based development workflows.
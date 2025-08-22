#!/usr/bin/env bash
# Optimized developer helper script with performance improvements and security fixes
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

activate_venv() {
  if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.venv/bin/activate"
  elif [[ -f "${ROOT_DIR}/.venv/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.venv/Scripts/activate"
  else
    log_error "Virtual environment not found. Run './dev.sh sync' first."
    exit 1
  fi
}

ensure_uv_and_python() {
  if ! command -v uv &>/dev/null; then
    log_error "uv is required. Install from https://astral.sh/uv"
    exit 1
  fi
  
  log_info "Setting up Python environment..."
  
  # Check if Python 3.12 is available
  if ! uv python list | grep -q "3.12"; then
    log_info "Installing Python 3.12..."
    uv python install 3.12 || {
      log_error "Failed to install Python 3.12"
      exit 1
    }
  fi
  
  if [[ ! -d "${ROOT_DIR}/.venv" ]]; then
    log_info "Creating virtual environment..."
    uv venv "${ROOT_DIR}/.venv" -p 3.12 || {
      log_error "Failed to create virtual environment"
      exit 1
    }
  fi
  
  activate_venv
  log_info "Syncing dependencies..."
  uv sync --compile-bytecode || {
    log_error "Failed to sync dependencies"
    exit 1
  }
  log_success "Environment setup complete"
 
# CUDA version management

check_cuda_compatibility() {
    local pytorch_cuda_version
    pytorch_cuda_version=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
    
    if [[ "$pytorch_cuda_version" == "none" ]]; then
        fail "PyTorch was not compiled with CUDA support"
        info "Reinstall with: uv sync --extra torch-cu124"
        return 1
    fi
    
    # Check actual CUDA runtime version
    local runtime_version
    if command -v nvidia-smi >/dev/null 2>&1; then
        runtime_version=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -1)
        info "NVIDIA Runtime CUDA version: $runtime_version"
    fi
    
    info "PyTorch CUDA version: $pytorch_cuda_version"
    
    # More flexible version checking
    local pytorch_major=$(echo "$pytorch_cuda_version" | cut -d. -f1)
    if [[ "$pytorch_major" == "12" ]]; then
        pass "PyTorch CUDA 12.x compatibility confirmed"
    else
        warn "Unexpected PyTorch CUDA version: $pytorch_cuda_version"
    fi
}

get_cuda_version() {
  if [[ -n "${CUDA_VERSION:-}" ]]; then
    echo "${CUDA_VERSION}"
  elif [[ -f ".devcontainer/devcontainer.json" ]]; then
    # Extract CUDA version from devcontainer.json
    python3 -c "
import json
try:
    with open('.devcontainer/devcontainer.json') as f:
        config = json.load(f)
        cuda_version = config.get('containerEnv', {}).get('CUDA_VERSION', '12.4')
        print(cuda_version)
except:
    print('12.4')
" 2>/dev/null || echo "12.4"
  else
    echo "12.4"
  fi
}

set_cuda_version() {
  local version="${1:-12.4}"
  local devcontainer_file=".devcontainer/devcontainer.json"
  
  if [[ ! -f "$devcontainer_file" ]]; then
    log_error "Dev container configuration not found"
    return 1
  fi
  
  # Update containerEnv.CUDA_VERSION
  python3 -c "
import json
import sys

try:
    with open('$devcontainer_file', 'r') as f:
        config = json.load(f)
    
    if 'containerEnv' not in config:
        config['containerEnv'] = {}
    
    config['containerEnv']['CUDA_VERSION'] = '$version'
    
    # Update NVIDIA feature cudaVersion
    if 'features' in config:
        for feature_name, feature_config in config['features'].items():
            if 'nvidia-cuda' in feature_name:
                feature_config['cudaVersion'] = '\${containerEnv:CUDA_VERSION}'
    
    # Update build args
    if 'build' not in config:
        config['build'] = {}
    if 'args' not in config['build']:
        config['build']['args'] = {}
    config['build']['args']['CUDA_VERSION'] = '\${containerEnv:CUDA_VERSION}'
    
    with open('$devcontainer_file', 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f'Updated CUDA version to {version}')
except Exception as e:
    print(f'Error updating CUDA version: {e}', file=sys.stderr)
    sys.exit(1)
" || {
    log_error "Failed to update CUDA version in devcontainer.json"
    return 1
  }
  
  log_success "CUDA version set to $version"
  log_info "Rebuild container to apply changes: Ctrl+Shift+P → 'Dev Container: Rebuild Container'"
}
}

# Performance-optimized commands with better error handling
cmd_format() {
  log_info "Formatting code with ruff..."
  activate_venv
  
  if ruff format . --diff --check >/dev/null 2>&1; then
    log_success "Code already formatted"
  else
    ruff format . || {
      log_error "Code formatting failed"
      return 1
    }
    log_success "Code formatted"
  fi
}

cmd_lint() {
  log_info "Running linter..."
  activate_venv
  ruff check . --output-format=github || {
    log_error "Linting found issues"
    return 1
  }
  log_success "Linting passed"
}

cmd_lint_fix() {
  log_info "Auto-fixing lint issues..."
  activate_venv
  ruff check --fix . --unsafe-fixes || log_warning "Some issues could not be auto-fixed"
  ruff format . || log_warning "Formatting encountered issues"
  log_success "Lint fixes applied"
}

cmd_fix_imports() {
  log_info "Fixing import order..."
  activate_venv
  ruff check --select I --fix . || log_warning "Some imports could not be auto-fixed"
  log_success "Imports organized"
}

cmd_typecheck() {
  log_info "Type checking with mypy..."
  activate_venv
  mypy . --show-error-codes --pretty || {
    log_error "Type checking failed"
    return 1
  }
  log_success "Type checking passed"
}

cmd_test() {
  log_info "Running tests..."
  activate_venv
  pytest -xvs --tb=short --disable-warnings || {
    log_error "Tests failed"
    return 1
  }
  log_success "All tests passed"
}

cmd_test_cov() {
  log_info "Running tests with coverage..."
  activate_venv
  pytest --cov=torch_starter --cov-report=html --cov-report=term-missing || {
    log_error "Tests with coverage failed"
    return 1
  }
  log_success "Tests with coverage completed"
}

cmd_all_checks() {
  log_info "Running all quality checks..."
  local start_time=$(date +%s)
  local failed_checks=()
  
  # Ensure environment is set up first
  ensure_uv_and_python
  
  cmd_format || failed_checks+=("format")
  cmd_lint || failed_checks+=("lint")
  cmd_fix_imports || failed_checks+=("imports")
  cmd_typecheck || failed_checks+=("typecheck")
  cmd_test || failed_checks+=("test")
  
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  
  if [[ ${#failed_checks[@]} -eq 0 ]]; then
    log_success "All checks completed successfully in ${duration}s"
  else
    log_error "Failed checks: ${failed_checks[*]} (completed in ${duration}s)"
    return 1
  fi
}

cmd_sync() {
  log_info "Syncing environment..."
  ensure_uv_and_python
}

cmd_sync_minimal() {
  log_info "Minimal dependency sync..."
  activate_venv
  uv sync --no-dev --compile-bytecode || {
    log_error "Minimal sync failed"
    return 1
  }
  log_success "Minimal sync complete"
}

cmd_lock_update() {
  log_info "Updating lock file..."
  activate_venv || true
  uv lock --upgrade || {
    log_error "Lock file update failed"
    return 1
  }
  uv sync --compile-bytecode || {
    log_error "Sync after lock update failed"
    return 1
  }
  log_success "Dependencies updated and synced"
}

# In dev.sh
cmd_validate_lock() {
    if uv lock --check 2>/dev/null; then
        log_success "Lock file is up to date"
    else
        log_warning "Lock file is outdated"
        log_info "Run './dev.sh lock-update' to update"
        return 1
    fi
}

cmd_clean() {
  log_info "Cleaning cache directories..."
  find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
  find . -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true
  find . -type d -name ".mypy_cache" -prune -exec rm -rf {} + 2>/dev/null || true
  find . -type d -name ".ruff_cache" -prune -exec rm -rf {} + 2>/dev/null || true
  rm -rf .coverage htmlcov build dist *.egg-info 2>/dev/null || true
  find . -type f -name "*.pyc" -delete 2>/dev/null || true
  log_success "Caches cleaned"
}

cmd_verify_setup() {
  log_info "Verifying environment setup..."
  if [[ -f "${ROOT_DIR}/verify_setup/verify_setup.sh" ]]; then
    bash "${ROOT_DIR}/verify_setup/verify_setup.sh" || {
      log_warning "Detailed verification encountered issues"
    }
  else
    log_warning "Verification script not found, running basic checks..."
    activate_venv
    python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
except ImportError:
    print('PyTorch not installed')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('Transformers not installed')
" || {
      log_error "Basic verification failed"
      return 1
    }
    log_success "Basic verification complete"
  fi
}

cmd_setup_keys() {
  if [[ -f "${ROOT_DIR}/verify_setup/setup_api_keys.sh" ]]; then
    bash "${ROOT_DIR}/verify_setup/setup_api_keys.sh" || {
      log_warning "API key setup encountered issues"
    }
  else
    log_warning "API key setup script not found"
    log_info "Create .env.local file manually with your API keys"
    log_info "See .env.example for template"
  fi
}

# Optimized package management with better error handling
cmd_add_temp() {
  local package="${1:-}"
  if [[ -z "$package" ]]; then
    log_error "Usage: ./dev.sh add-temp <package>"
    exit 1
  fi
  
  log_info "Adding temporary package: $package"
  activate_venv
  uv add "$package" --compile-bytecode || {
    log_error "Failed to add package: $package"
    return 1
  }
  log_success "Package '$package' added (temporary until container rebuild)"
}

cmd_add_perm() {
  local package="${1:-}"
  if [[ -z "$package" ]]; then
    log_error "Usage: ./dev.sh add-perm <package>"
    exit 1
  fi
  
  log_info "Adding permanent package: $package"
  
  # Backup pyproject.toml
  cp pyproject.toml pyproject.toml.bak || {
    log_error "Failed to backup pyproject.toml"
    return 1
  }
  
  # Add to dependencies using uv
  activate_venv 2>/dev/null || true
  uv add "$package" || {
    log_error "Failed to add package: $package"
    log_info "Restoring backup..."
    mv pyproject.toml.bak pyproject.toml
    return 1
  }
  
  rm pyproject.toml.bak
  log_success "Package '$package' added to pyproject.toml"
  log_warning "Container rebuild required for team sync"
}

cmd_remove() {
  local package="${1:-}"
  if [[ -z "$package" ]]; then
    log_error "Usage: ./dev.sh remove <package>"
    exit 1
  fi
  
  log_info "Removing package: $package"
  activate_venv
  uv remove "$package" || {
    log_error "Failed to remove package: $package"
    return 1
  }
  log_success "Package '$package' removed"
}

cmd_rebuild_image() {
  log_info "Container rebuild instructions:"
  echo ""
  echo "🔄 To rebuild the container:"
  echo "1. In VS Code: Press Ctrl+Shift+P (Cmd+Shift+P on Mac)"
  echo "2. Type: 'Dev Container: Rebuild Container'"
  echo "3. Select the command and wait for rebuild"
  echo ""
  echo "📊 Build times:"
  echo "• First build: ~10-15 minutes (optimized)"
  echo "• Subsequent builds: ~5-8 minutes (cached layers)"
  echo "• Daily startup: ~15-30 seconds"
  echo ""
  echo "🏗️  Alternatively from terminal:"
  echo "   docker build -f .devcontainer/Dockerfile ."
}

cmd_sync_temp() {
  log_info "Syncing temporary packages..."
  activate_venv
  uv sync --no-install-project --compile-bytecode || {
    log_error "Temporary package sync failed"
    return 1
  }
  log_success "Temporary packages synced"
}

# Performance monitoring
cmd_profile() {
  local script="${1:-}"
  if [[ -z "$script" ]]; then
    log_error "Usage: ./dev.sh profile <script.py>"
    exit 1
  fi
  
  if [[ ! -f "$script" ]]; then
    log_error "Script not found: $script"
    return 1
  fi
  
  log_info "Profiling $script with viztracer..."
  activate_venv
  if ! python -c "import viztracer" 2>/dev/null; then
    log_warning "Installing viztracer temporarily..."
    uv add viztracer --compile-bytecode || {
      log_error "Failed to install viztracer"
      return 1
    }
  fi
  
  viztracer --tracer_entries 1000000 "$script" || {
    log_error "Profiling failed"
    return 1
  }
  log_success "Profile saved to result.html"
}

cmd_memory_profile() {
  local script="${1:-}"
  if [[ -z "$script" ]]; then
    log_error "Usage: ./dev.sh memory-profile <script.py>"
    exit 1
  fi
  
  if [[ ! -f "$script" ]]; then
    log_error "Script not found: $script"
    return 1
  fi
  
  log_info "Memory profiling $script..."
  activate_venv
  if ! python -c "import memory_profiler" 2>/dev/null; then
    log_warning "Installing memory-profiler temporarily..."
    uv add memory-profiler --compile-bytecode || {
      log_error "Failed to install memory-profiler"
      return 1
    }
  fi
  
  python -m memory_profiler "$script" || {
    log_error "Memory profiling failed"
    return 1
  }
}

cmd_jupyter() {
  log_info "Starting Jupyter Lab..."
  activate_venv
  
  # Check if jupyterlab is available
  if ! python -c "import jupyterlab" 2>/dev/null; then
    log_warning "JupyterLab not found, installing temporarily..."
    uv add jupyterlab --compile-bytecode || {
      log_error "Failed to install JupyterLab"
      return 1
    }
  fi
  
  # Generate a secure token if none exists
  if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
    JUPYTER_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
  fi
  
  echo ""
  log_success "🔐 Jupyter Lab will be available at:"
  echo "   http://localhost:8888/lab?token=${JUPYTER_TOKEN}"
  echo ""
  log_info "Starting Jupyter Lab with secure token..."
  
  jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token="${JUPYTER_TOKEN}" \
    --ServerApp.password='' \
    --ServerApp.open_browser=False \
    --ServerApp.allow_remote_access=True
}

cmd_notebook() {
  log_info "Starting Jupyter Notebook..."
  activate_venv
  
  if ! python -c "import notebook" 2>/dev/null; then
    log_warning "Installing Jupyter Notebook temporarily..."
    uv add notebook --compile-bytecode || {
      log_error "Failed to install Jupyter Notebook"
      return 1
    }
  fi
  
  # Generate a secure token if none exists
  if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
    JUPYTER_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
  fi
  
  echo ""
  log_success "🔐 Jupyter Notebook will be available at:"
  echo "   http://localhost:8888/?token=${JUPYTER_TOKEN}"
  echo ""
  log_info "Starting Jupyter Notebook with secure token..."
  
  jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token="${JUPYTER_TOKEN}" \
    --NotebookApp.password='' \
    --NotebookApp.open_browser=False \
    --NotebookApp.allow_remote_access=True
}

# Docker and container management
cmd_docker_stats() {
  log_info "Container resource usage:"
  docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" || {
    log_error "Failed to get Docker stats"
    return 1
  }
}

cmd_docker_cleanup() {
  log_info "Cleaning up Docker resources..."
  docker system prune -f || log_warning "Docker system prune failed"
  docker volume prune -f || log_warning "Docker volume prune failed"
  log_success "Docker cleanup complete"
}

# Environment optimization
cmd_optimize() {
  log_info "Running environment optimizations..."
  
  # Clean caches
  cmd_clean
  
  # Precompile Python files
  activate_venv
  python -m compileall . -q -j 0 2>/dev/null || true
  
  # Optimize uv cache
  uv cache clean 2>/dev/null || true
  
  # Remove unused packages
  uv sync --compile-bytecode || log_warning "Sync during optimization failed"
  
  log_success "Environment optimized"
}

# Git hooks and pre-commit
cmd_install_hooks() {
  log_info "Installing pre-commit hooks..."
  activate_venv
  
  if ! command -v pre-commit &>/dev/null; then
    log_info "Installing pre-commit..."
    uv add pre-commit --compile-bytecode || {
      log_error "Failed to install pre-commit"
      return 1
    }
  fi
  
  pre-commit install || {
    log_error "Failed to install pre-commit hooks"
    return 1
  }
  log_success "Pre-commit hooks installed"
}

cmd_run_hooks() {
  log_info "Running pre-commit hooks on all files..."
  activate_venv
  pre-commit run --all-files || {
    log_error "Pre-commit hooks failed"
    return 1
  }
}

# Dependency analysis
cmd_deps_tree() {
  log_info "Dependency tree:"
  activate_venv
  uv tree || {
    log_error "Failed to show dependency tree"
    return 1
  }
}

cmd_deps_outdated() {
  log_info "Checking for outdated packages..."
  activate_venv
  # Note: uv doesn't have direct outdated command yet
  log_warning "Use './dev.sh lock-update' to update all dependencies"
}

# Security scanning
cmd_security() {
  log_info "Running security scan..."
  activate_venv
  
  if ! python -c "import safety" 2>/dev/null; then
    log_warning "Installing safety temporarily..."
    uv add safety --compile-bytecode || {
      log_error "Failed to install safety"
      return 1
    }
  fi
  
  safety scan || {
    log_warning "Security scan found issues"
    return 1
  }
  log_success "Security scan completed"
}

cmd_select_gpu() {
    if [[ $(nvidia-smi -L | wc -l) -gt 1 ]]; then
        echo "Available GPUs:"
        nvidia-smi -L
        read -p "Select GPU (0-N, or 'all'): " gpu_selection
        export CUDA_VISIBLE_DEVICES="$gpu_selection"
    fi
}

# Benchmarking
cmd_benchmark() {
  log_info "Running environment benchmark..."
  activate_venv
  
  python -c "
import time
import sys
try:
    import torch
    import numpy as np
    
    print('🔥 Environment Benchmark')
    print('=' * 40)
    
    # Python performance
    start = time.time()
    _ = [i**2 for i in range(100000)]
    python_time = time.time() - start
    print(f'Python compute: {python_time:.3f}s')
    
    # NumPy performance
    start = time.time()
    arr = np.random.randn(1000, 1000)
    _ = np.dot(arr, arr)
    numpy_time = time.time() - start
    print(f'NumPy matmul: {numpy_time:.3f}s')
    
    # PyTorch CPU performance
    start = time.time()
    tensor = torch.randn(1000, 1000)
    _ = torch.mm(tensor, tensor)
    torch_cpu_time = time.time() - start
    print(f'PyTorch CPU: {torch_cpu_time:.3f}s')
    
    # PyTorch GPU performance (if available)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        tensor = torch.randn(1000, 1000, device=device)
        torch.cuda.synchronize()
        start = time.time()
        _ = torch.mm(tensor, tensor)
        torch.cuda.synchronize()
        torch_gpu_time = time.time() - start
        print(f'PyTorch GPU: {torch_gpu_time:.3f}s')
        print(f'GPU speedup: {torch_cpu_time/torch_gpu_time:.1f}x')
    else:
        print('CUDA not available')
    
    print('=' * 40)
except ImportError as e:
    print(f'Benchmark failed: {e}')
    sys.exit(1)
" || {
    log_error "Benchmark failed"
    return 1
  }
}

# Help and usage
usage() {
  cat <<'USAGE'
🚀 Optimized Development Script

ENVIRONMENT:
  sync                Create/activate .venv and sync dependencies
  sync-minimal        Sync only core dependencies (no dev tools)
  optimize           Run environment optimizations
  verify-setup       Verify environment and dependencies
  setup-keys         Setup API keys interactively
  cuda-version       Show current CUDA version
  set-cuda <version> Set CUDA version (requires rebuild)
  list-cuda          List supported CUDA versions
  clean              Remove caches and build artifacts

CODE QUALITY:
  format             Format code with ruff
  lint               Run linter checks
  lint-fix           Auto-fix lint issues
  fix-imports        Organize imports
  typecheck          Run mypy type checking
  test               Run pytest
  test-cov           Run tests with coverage
  all-checks         Run all quality checks

PACKAGE MANAGEMENT:
  add-temp <pkg>     Add package temporarily (lost on rebuild)
  add-perm <pkg>     Add package permanently (requires rebuild)
  remove <pkg>       Remove package
  sync-temp          Sync temporary packages only
  deps-tree          Show dependency tree
  deps-outdated      Check for outdated packages
  lock-update        Update lock file and sync

JUPYTER & NOTEBOOKS:
  jupyter            Start Jupyter Lab (with secure token)
  notebook           Start Jupyter Notebook (with secure token)

PERFORMANCE & DEBUGGING:
  profile <script>   Profile script with viztracer
  memory-profile <script>  Memory profiling
  benchmark          Run performance benchmark

DOCKER & CONTAINERS:
  docker-stats       Show container resource usage
  docker-cleanup     Clean Docker resources
  rebuild-image      Show rebuild instructions

DEVELOPMENT TOOLS:
  install-hooks      Install pre-commit hooks
  run-hooks          Run pre-commit on all files
  security           Run security scan

EXAMPLES:
  ./dev.sh sync && ./dev.sh all-checks
  ./dev.sh add-temp wandb && ./dev.sh jupyter
  ./dev.sh profile scripts/train.py
  ./dev.sh benchmark

SECURITY NOTES:
  - Jupyter now uses secure tokens by default
  - API keys should be stored in .env.local (see .env.example)
USAGE
}

# Performance timer wrapper
time_command() {
  local func_name="$1"
  local start_time=$(date +%s.%N 2>/dev/null || date +%s)
  
  shift
  "$func_name" "$@"
  local exit_code=$?
  
  local end_time=$(date +%s.%N 2>/dev/null || date +%s)
  local duration
  if command -v bc >/dev/null 2>&1; then
    duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
  else
    duration=$(( end_time - start_time ))
  fi
  
  if [[ $exit_code -eq 0 ]]; then
    log_success "Command completed in ${duration}s"
  else
    log_error "Command failed after ${duration}s"
  fi
  
  return $exit_code
}

# Main command dispatcher with better error handling
main() {
  local cmd="${1:-}"
  
  # Validate command exists
  case "$cmd" in
    sync|sync-minimal|format|lint|lint-fix|fix-imports|typecheck|test|test-cov|all-checks|lock-update|validate-lock|clean|verify-setup|setup-keys|add-temp|add-perm|remove|rebuild-image|sync-temp|profile|memory-profile|jupyter|notebook|docker-stats|docker-cleanup|optimize|install-hooks|run-hooks|deps-tree|deps-outdated|security|benchmark|cuda-version|set-cuda|list-cuda|lock-update|""|help|-h|--help)
      # Valid command, continue
      ;;
    *)
      log_error "Unknown command: $cmd"
      echo ""
      usage
      exit 1
      ;;
  esac
  
  # Add timing to long-running commands
  case "$cmd" in
    sync|all-checks|test|typecheck|lock-update|benchmark)
      # Replace dashes with underscores in function names
      local func_name="cmd_${cmd//-/_}"
      time_command "$func_name" "${@:2}"
      ;;
    sync-minimal) cmd_sync_minimal ;;
    format) cmd_format ;;
    lint) cmd_lint ;;
    lint-fix) cmd_lint_fix ;;
    fix-imports) cmd_fix_imports ;;
    typecheck) cmd_typecheck ;;
    test) cmd_test ;;
    test-cov) cmd_test_cov ;;
    all-checks) cmd_all_checks ;;
    lock-update) cmd_lock_update ;;
    clean) cmd_clean ;;
    validate-lock) cmd_validate_lock ;;
    verify-setup) cmd_verify_setup ;;
    setup-keys) cmd_setup_keys ;;
    add-temp) cmd_add_temp "${2:-}" ;;
    add-perm) cmd_add_perm "${2:-}" ;;
    remove) cmd_remove "${2:-}" ;;
    rebuild-image) cmd_rebuild_image ;;
    sync-temp) cmd_sync_temp ;;
    profile) cmd_profile "${2:-}" ;;
    memory-profile) cmd_memory_profile "${2:-}" ;;
    jupyter) cmd_jupyter ;;
    notebook) cmd_notebook ;;
    docker-stats) cmd_docker_stats ;;
    docker-cleanup) cmd_docker_cleanup ;;
    optimize) cmd_optimize ;;
    install-hooks) cmd_install_hooks ;;
    run-hooks) cmd_run_hooks ;;
    deps-tree) cmd_deps_tree ;;
    deps-outdated) cmd_deps_outdated ;;
    security) cmd_security ;;
    benchmark) cmd_benchmark ;;
    cuda-version) cmd_cuda_version ;;
    set-cuda) cmd_set_cuda "${2:-}" ;;
    list-cuda) cmd_list_cuda ;;
    ""|help|-h|--help) usage ;;
  esac
}

# Error handling
trap 'log_error "Script interrupted"; exit 130' INT
trap 'log_error "Script terminated"; exit 143' TERM

main "$@"
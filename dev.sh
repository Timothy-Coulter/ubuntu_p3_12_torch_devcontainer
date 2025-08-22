#!/usr/bin/env bash
# Optimized developer helper script with performance improvements
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
  uv python install 3.12
  
  if [[ ! -d "${ROOT_DIR}/.venv" ]]; then
    log_info "Creating virtual environment..."
    uv venv "${ROOT_DIR}/.venv" -p 3.12
  fi
  
  activate_venv
  log_info "Syncing dependencies..."
  uv sync --compile-bytecode
  log_success "Environment setup complete"
}

# Performance-optimized commands
cmd_format() {
  log_info "Formatting code with ruff..."
  activate_venv
  ruff format . --diff --check
  if [[ $? -eq 0 ]]; then
    log_success "Code already formatted"
  else
    ruff format .
    log_success "Code formatted"
  fi
}

cmd_lint() {
  log_info "Running linter..."
  activate_venv
  ruff check . --output-format=github
}

cmd_lint_fix() {
  log_info "Auto-fixing lint issues..."
  activate_venv
  ruff check --fix . --unsafe-fixes
  ruff format .
  log_success "Lint issues fixed"
}

cmd_fix_imports() {
  log_info "Fixing import order..."
  activate_venv
  ruff check --select I --fix .
  log_success "Imports organized"
}

cmd_typecheck() {
  log_info "Type checking with mypy..."
  activate_venv
  mypy . --show-error-codes --pretty
}

cmd_test() {
  log_info "Running tests..."
  activate_venv
  pytest -xvs --tb=short --disable-warnings
}

cmd_test_cov() {
  log_info "Running tests with coverage..."
  activate_venv
  pytest --cov=torch_starter --cov-report=html --cov-report=term-missing
}

cmd_all_checks() {
  log_info "Running all quality checks..."
  local start_time=$(date +%s)
  
  cmd_format
  cmd_lint  
  cmd_fix_imports
  cmd_typecheck
  cmd_test
  
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  log_success "All checks completed in ${duration}s"
}

cmd_sync() {
  log_info "Syncing environment..."
  ensure_uv_and_python
}

cmd_sync_minimal() {
  log_info "Minimal dependency sync..."
  activate_venv
  uv sync --no-dev --compile-bytecode
  log_success "Minimal sync complete"
}

cmd_lock_update() {
  log_info "Updating lock file..."
  activate_venv || true
  uv lock --upgrade
  uv sync --compile-bytecode
  log_success "Dependencies updated and synced"
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
  bash "${ROOT_DIR}/verify_setup/verify_setup.sh" 2>/dev/null || {
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
"
    log_success "Basic verification complete"
  }
}

cmd_setup_keys() {
  if [[ -f "${ROOT_DIR}/verify_setup/setup_api_keys.sh" ]]; then
    bash "${ROOT_DIR}/verify_setup/setup_api_keys.sh"
  else
    log_warning "API key setup script not found"
    log_info "Create .env.local file manually with your API keys"
  fi
}

# Optimized package management
cmd_add_temp() {
  local package="${1:-}"
  if [[ -z "$package" ]]; then
    log_error "Usage: ./dev.sh add-temp <package>"
    exit 1
  fi
  
  log_info "Adding temporary package: $package"
  activate_venv
  uv add "$package" --compile-bytecode
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
  cp pyproject.toml pyproject.toml.bak
  
  # Add to dependencies using uv
  activate_venv 2>/dev/null || true
  uv add "$package"
  
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
  uv remove "$package"
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
  uv sync --no-install-project --compile-bytecode
  log_success "Temporary packages synced"
}

# Performance monitoring
cmd_profile() {
  local script="${1:-}"
  if [[ -z "$script" ]]; then
    log_error "Usage: ./dev.sh profile <script.py>"
    exit 1
  fi
  
  log_info "Profiling $script with viztracer..."
  activate_venv
  if ! python -c "import viztracer" 2>/dev/null; then
    log_warning "Installing viztracer temporarily..."
    uv add viztracer --compile-bytecode
  fi
  
  viztracer --tracer_entries 1000000 "$script"
  log_success "Profile saved to result.html"
}

cmd_memory_profile() {
  local script="${1:-}"
  if [[ -z "$script" ]]; then
    log_error "Usage: ./dev.sh memory-profile <script.py>"
    exit 1
  fi
  
  log_info "Memory profiling $script..."
  activate_venv
  if ! python -c "import memory_profiler" 2>/dev/null; then
    log_warning "Installing memory-profiler temporarily..."
    uv add memory-profiler --compile-bytecode
  fi
  
  python -m memory_profiler "$script"
}

# Jupyter and notebook management
cmd_jupyter() {
  log_info "Starting Jupyter Lab..."
  activate_venv
  
  # Check if jupyterlab is available
  if ! python -c "import jupyterlab" 2>/dev/null; then
    log_warning "JupyterLab not found, installing temporarily..."
    uv add jupyterlab --compile-bytecode
  fi
  
  jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password=''
}

cmd_notebook() {
  log_info "Starting Jupyter Notebook..."
  activate_venv
  
  if ! python -c "import notebook" 2>/dev/null; then
    uv add notebook --compile-bytecode
  fi
  
  jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password=''
}

# Docker and container management
cmd_docker_stats() {
  log_info "Container resource usage:"
  docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"
}

cmd_docker_cleanup() {
  log_info "Cleaning up Docker resources..."
  docker system prune -f
  docker volume prune -f
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
  uv sync --compile-bytecode
  
  log_success "Environment optimized"
}

# Git hooks and pre-commit
cmd_install_hooks() {
  log_info "Installing pre-commit hooks..."
  activate_venv
  
  if ! command -v pre-commit &>/dev/null; then
    uv add pre-commit --compile-bytecode
  fi
  
  pre-commit install
  log_success "Pre-commit hooks installed"
}

cmd_run_hooks() {
  log_info "Running pre-commit hooks on all files..."
  activate_venv
  pre-commit run --all-files
}

# Dependency analysis
cmd_deps_tree() {
  log_info "Dependency tree:"
  activate_venv
  uv tree
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
    uv add safety --compile-bytecode
  fi
  
  safety check
}

# Benchmarking
cmd_benchmark() {
  log_info "Running environment benchmark..."
  activate_venv
  
  python -c "
import time
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
"
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
  jupyter            Start Jupyter Lab
  notebook           Start Jupyter Notebook

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
USAGE
}

# Performance timer wrapper
time_command() {
  local cmd="$1"
  local start_time=$(date +%s.%N)
  
  shift
  "$cmd" "$@"
  local exit_code=$?
  
  local end_time=$(date +%s.%N)
  local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
  
  if [[ $exit_code -eq 0 ]]; then
    log_success "Command completed in ${duration}s"
  else
    log_error "Command failed after ${duration}s"
  fi
  
  return $exit_code
}

# Main command dispatcher
main() {
  local cmd="${1:-}"
  
  # Add timing to long-running commands
  case "$cmd" in
    sync|all-checks|test|typecheck|lock-update)
      time_command "cmd_$cmd" "${@:2}"
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
    ""|help|-h|--help) usage ;;
    *) 
      log_error "Unknown command: $cmd"
      echo ""
      usage
      exit 1
      ;;
  esac
}

# Error handling
trap 'log_error "Script interrupted"; exit 130' INT
trap 'log_error "Script terminated"; exit 143' TERM

main "$@"
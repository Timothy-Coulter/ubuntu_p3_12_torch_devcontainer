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
  
  log_success "Package '$package' added to pyproject.toml
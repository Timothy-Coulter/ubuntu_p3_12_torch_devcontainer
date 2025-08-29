#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Global Config
# ==============================================================================

EXPECTED_CUDA_VERSION="12.6"   # Expected CUDA version for PyTorch & system

# ==============================================================================
# Logging Helpers
# ==============================================================================

COLOR_RESET="\033[0m"
COLOR_RED="\033[0;31m"
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[0;33m"
COLOR_BLUE="\033[0;34m"

log_error()   { echo -e "${COLOR_RED}[ERROR] $*${COLOR_RESET}" >&2; }
log_success() { echo -e "${COLOR_GREEN}[ OK ] $*${COLOR_RESET}"; }
log_warning() { echo -e "${COLOR_YELLOW}[WARN] $*${COLOR_RESET}"; }
log_info()    { echo -e "${COLOR_BLUE}[INFO] $*${COLOR_RESET}"; }

# Aliases for consistency (from verify_setup.sh)
fail()  { log_error "$@"; return 1; }
pass()  { log_success "$@"; }
warn()  { log_warning "$@"; }
info()  { log_info "$@"; }

section() { echo -e "\n${COLOR_BLUE}=== $* ===${COLOR_RESET}"; }

# ==============================================================================
# Core Helpers
# ==============================================================================

ensure_uv_and_python() {
  if ! command -v uv >/dev/null 2>&1; then
    log_error "uv package manager not found. Install from https://astral.sh/uv/"
    exit 1
  fi

  if ! uv python list | grep -q "${PYTHON_VERSION:-3.12}"; then
    log_info "Installing Python ${PYTHON_VERSION:-3.12} with uv..."
    uv python install "${PYTHON_VERSION:-3.12}"
  fi

  if [[ ! -d ".venv" ]]; then
    log_info "Creating virtual environment..."
    uv venv --clear -p "${PYTHON_VERSION:-3.12}"
  fi
}

python_exec() {
  echo ".venv/bin/python"
}

# ==============================================================================
# Commands
# ==============================================================================

cmd_sync() {
  ensure_uv_and_python
  uv sync --compile-bytecode --no-install-project
}

cmd_lock_update() {
  ensure_uv_and_python
  uv lock --upgrade
}

cmd_validate_lock() {
  ensure_uv_and_python
  uv lock --check
}

cmd_clean() {
  log_info "Cleaning environment..."
  rm -rf .venv .pytest_cache __pycache__ */__pycache__ build dist *.egg-info
  find . -type f -name '*.pyc' -delete
}

cmd_verify_setup() {
  section "Verifying Environment Setup"

  ensure_uv_and_python

  local py
  py=$(python_exec)

  # Python version
  section "Python"
  $py --version

  # PyTorch CUDA check
  section "PyTorch / CUDA"
  $py verify_setup/print_torch_info.py

  # CUDA version check
  system_cuda=$(grep -oE "[0-9]+\.[0-9]+" /usr/local/cuda/version.txt | head -n1 || echo "unknown")
  log_info "System CUDA: $system_cuda"
  if [[ "$system_cuda" != "$EXPECTED_CUDA_VERSION" ]]; then
    log_warning "System CUDA version ($system_cuda) does not match expected ($EXPECTED_CUDA_VERSION)"
  fi

  # Linkage check
  torch_lib=$($py verify_setup/get_torch_lib_path.py 2>/dev/null || echo "")
  if [[ -f "$torch_lib" ]]; then
    log_info "Checking CUDA linkage in $torch_lib"
    if ldd "$torch_lib" | grep -q "/usr/local/cuda"; then
      log_success "PyTorch is linked against system CUDA libraries"
    else
      log_warning "PyTorch may be using bundled CUDA libraries"
      ldd "$torch_lib" | grep cuda || true
    fi
  else
    log_warning "Could not find libtorch_cuda.so"
  fi

  # NVIDIA tools
  section "NVIDIA"
  nvidia-smi || log_warning "nvidia-smi not available"
  nvcc --version || log_warning "nvcc not found"
}

cmd_test() {
  ensure_uv_and_python
  .venv/bin/pytest -v
}

cmd_coverage() {
  ensure_uv_and_python
  .venv/bin/coverage run -m pytest
  .venv/bin/coverage report -m
}

cmd_lint() {
  ensure_uv_and_python
  .venv/bin/ruff check .
  .venv/bin/mypy .
}

cmd_all_checks() {
  cmd_lint
  cmd_test
  cmd_coverage
}

cmd_jupyter() {
  ensure_uv_and_python
  local port="${2:-8888}"
  .venv/bin/jupyter lab --ip=0.0.0.0 --port="$port" --no-browser --ServerApp.allow_remote_access=True
}

cmd_notebook() {
  ensure_uv_and_python
  local port="${2:-8888}"
  .venv/bin/jupyter notebook --ip=0.0.0.0 --port="$port" --no-browser --ServerApp.allow_remote_access=True
}

cmd_shell() {
  ensure_uv_and_python
  .venv/bin/bash
}

cmd_list_cuda() {
  log_info "Available CUDA toolkits on system:"
  ls -d /usr/local/cuda* || log_warning "No CUDA directories found"
}

cmd_docker_cleanup() {
  if [[ "${2:-}" != "--force" ]]; then
    read -rp "⚠️  This will remove unused Docker resources. Continue? [y/N] " yn
    [[ "$yn" == "y" || "$yn" == "Y" ]] || { log_info "Aborted."; return; }
  fi
  docker system prune -f
  docker volume prune -f
}

cmd_doctor() {
  section "Doctor: Environment Health Check"
  cmd_verify_setup
  cmd_lint || log_warning "Lint issues found"
  cmd_test || log_warning "Tests failed"
}

cmd_help() {
  cat <<EOF
Usage: $0 <command> [options]

Commands:
  sync           Install dependencies with uv
  lock-update    Update lock file
  validate-lock  Validate lock file
  clean          Remove build/test artifacts
  verify-setup   Verify CUDA, PyTorch, Python
  test           Run tests
  coverage       Run tests with coverage
  lint           Run linters
  all-checks     Lint + test + coverage
  jupyter [p]    Start JupyterLab on port p (default 8888)
  notebook [p]   Start Jupyter Notebook on port p
  shell          Start shell inside venv
  list-cuda      List installed CUDA versions
  docker-clean   Clean Docker resources
  doctor         Run environment health check
  help           Show this message
EOF
}

# ==============================================================================
# Dispatcher
# ==============================================================================

cmd="${1:-help}"
shift || true

case "$cmd" in
  sync)           cmd_sync "$@" ;;
  lock-update)    cmd_lock_update ;;
  validate-lock)  cmd_validate_lock ;;
  clean)          cmd_clean ;;
  verify-setup)   cmd_verify_setup ;;
  test)           cmd_test ;;
  coverage)       cmd_coverage ;;
  lint)           cmd_lint ;;
  all-checks)     cmd_all_checks ;;
  jupyter)        cmd_jupyter "$@" ;;
  notebook)       cmd_notebook "$@" ;;
  shell)          cmd_shell ;;
  list-cuda)      cmd_list_cuda ;;
  docker-clean)   cmd_docker_cleanup "$@" ;;
  doctor)         cmd_doctor ;;
  help|""|-h|--help) cmd_help ;;
  *) log_error "Unknown command: $cmd"; cmd_help; exit 1 ;;
esac

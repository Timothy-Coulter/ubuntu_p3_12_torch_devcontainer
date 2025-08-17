#!/usr/bin/env bash
# Developer helper script for formatting, linting, typechecking, testing, syncing deps, etc.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

activate_venv() {
  if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.venv/bin/activate"
  elif [[ -f "${ROOT_DIR}/.venv/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.venv/Scripts/activate"
  fi
}

ensure_uv_and_python() {
  if ! command -v uv &>/dev/null; then
    echo "uv is required. Install inside devcontainer or from https://astral.sh/uv" >&2
    exit 1
  fi
  uv python install 3.12
  if [[ ! -d "${ROOT_DIR}/.venv" ]]; then
    uv venv "${ROOT_DIR}/.venv" -p 3.12
  fi
  activate_venv
  uv sync
}

cmd_format() {
  activate_venv
  ruff format .
}

cmd_lint() {
  activate_venv
  ruff check .
}

cmd_lint_fix() {
  cmd_format
  activate_venv
  ruff check --fix .
}

cmd_fix_imports() {
  activate_venv
  ruff check --select I --fix .
}

cmd_typecheck() {
  activate_venv
  mypy .
}

cmd_test() {
  activate_venv
  pytest -q
}

cmd_all_checks() {
  cmd_format
  cmd_lint
  cmd_fix_imports
  cmd_typecheck
  cmd_test
}

cmd_sync() {
  ensure_uv_and_python
}

cmd_lock_update() {
  activate_venv || true
  uv lock --upgrade
  uv sync
}

cmd_clean() {
  find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
  find . -type d -name ".pytest_cache" -prune -exec rm -rf {} + || true
  rm -rf .mypy_cache .ruff_cache .coverage htmlcov || true
  find . -type f -name "*.pyc" -delete || true
}

cmd_verify_setup() {
  bash "${ROOT_DIR}/verify_setup/verify_setup.sh"
}

cmd_setup_keys() {
  bash "${ROOT_DIR}/verify_setup/setup_api_keys.sh"
}

usage() {
  cat <<'USAGE'
Usage: ./.dev.sh <command>

Commands:
  sync            Create/activate .venv with uv and install deps
  format          Format code (ruff format)
  lint            Run linter (ruff)
  lint-fix        Format + autofix lint issues (ruff --fix)
  fix-imports     Autofix import ordering (ruff I)
  typecheck       Run mypy in strict mode
  test            Run pytest
  all-checks      Run format, lint, fix-imports, typecheck, and tests
  lock-update     Update lock and sync deps (uv)
  clean           Remove build and test caches
  verify-setup    Run environment verification (CUDA, caches, libs)
  setup-keys      Prompt for and store API keys safely

Examples:
  ./.dev.sh sync
  ./.dev.sh all-checks
  ./.dev.sh verify-setup
USAGE
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    sync) cmd_sync ;;
    format) cmd_format ;;
    lint) cmd_lint ;;
    lint-fix) cmd_lint_fix ;;
    fix-imports) cmd_fix_imports ;;
    typecheck) cmd_typecheck ;;
    test) cmd_test ;;
    all-checks) cmd_all_checks ;;
    lock-update) cmd_lock_update ;;
    clean) cmd_clean ;;
    verify-setup) cmd_verify_setup ;;
    setup-keys) cmd_setup_keys ;;
    ""|help|-h|--help) usage ;;
    *) echo "Unknown command: $cmd" >&2; usage; exit 1 ;;
  esac
}

main "$@"
#!/usr/bin/env bash
# Verifies dev environment, Python/uv toolchain, core libraries, and CUDA availability.

set -euo pipefail

color() { printf "\033[%sm%s\033[0m\n" "$1" "${2:-}"; }
info()  { color "36" "➤ $*"; }
pass()  { color "32" "✔ $*"; }
warn()  { color "33" "⚠ $*"; }
fail()  { color "31" "✘ $*"; }

section() { echo; color "1;34" "=== $* ==="; }

check_bin() {
  if command -v "$1" &>/dev/null; then
    pass "$1 found: $(command -v "$1")"
  else
    fail "$1 not found in PATH"
    return 1
  fi
}

python_exec() {
  # Prefer project venv if present
  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
  elif [[ -x ".venv/Scripts/python.exe" ]]; then
    # Windows virtualenv (if executed on host)
    echo ".venv/Scripts/python.exe"
  else
    if command -v python3 &>/dev/null; then
      echo "python3"
    else
      echo "python"
    fi
  fi
}

check_python_version() {
  local py
  py="$(python_exec)"
  info "Using Python at: ${py}"
  local version
  version="$("$py" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  echo "Python version: ${version}"
  # Require 3.12.x
  if ! "$py" -c 'import sys; sys.exit(0 if (sys.version_info[:2] == (3,12)) else 1)'; then
    fail "Python 3.12 is required"
    return 1
  fi
  pass "Python 3.12 verified"
}

check_imports_and_versions() {
  local py
  py="$(python_exec)"
  "$py" - <<'PY'
import importlib, json

pkgs = ["numpy", "transformers", "datasets", "tqdm", "pandas", "scipy", "sklearn", "accelerate", "matplotlib", "seaborn"]
info = {}
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", "unknown")
        info[p] = {"ok": True, "version": v}
    except Exception as e:
        info[p] = {"ok": False, "error": str(e)}

print(json.dumps(info, indent=2))
PY
  echo

  # Torch and CUDA checks
  "$py" - <<'PY'
import json, os
try:
    import torch
    res = {
        "torch_ok": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_is_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_arch_list": getattr(torch.cuda, "get_arch_list", lambda: [])(),
    }
    if res["cuda_is_available"] and res["cuda_device_count"] > 0:
        devices = []
        for i in range(res["cuda_device_count"]):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_GB": round(props.total_memory / (1024**3), 2),
                "major": props.major,
                "minor": props.minor,
            })
        res["devices"] = devices
    print(json.dumps(res, indent=2))
except Exception as e:
    print(json.dumps({"torch_ok": False, "error": str(e)}, indent=2))
PY
}

check_caches_rw() {
  section "Cache and data directories"
  local hf="${HF_HOME:-/home/vscode/.cache/huggingface}"
  local tf="${TRANSFORMERS_CACHE:-$hf}"
  local th="${TORCH_HOME:-/home/vscode/.cache/torch}"
  local kg="${KAGGLE_CONFIG_DIR:-/home/vscode/.kaggle}"
  local data_dir="${DATA_DIR:-${PWD}/data}"

  echo "HF_HOME=${hf}"
  echo "TRANSFORMERS_CACHE=${tf}"
  echo "TORCH_HOME=${th}"
  echo "KAGGLE_CONFIG_DIR=${kg}"
  echo "DATA_DIR=${data_dir}"

  mkdir -p "$hf" "$tf" "$th" "$kg" "$data_dir" || true

  local ok=0
  for d in "$hf" "$tf" "$th" "$kg" "$data_dir"; do
    if [[ -d "$d" ]] && [[ -w "$d" ]]; then
      pass "RW OK: $d"
      ok=$((ok+1))
    else
      warn "Cannot write: $d"
    fi
  done

  # Write small sentinel files
  echo "ok" > "$hf/.verify" || true
  echo "ok" > "$th/.verify" || true
  echo "ok" > "$data_dir/.verify" || true

  if [[ "$ok" -lt 3 ]]; then
    warn "Some cache/data directories are not writable. Check devcontainer mounts."
  else
    pass "Cache/data directories are writable."
  fi
}

print_cuda_runtime_info() {
  section "CUDA runtime (nvidia-smi)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || warn "nvidia-smi returned non-zero"
  else
    warn "nvidia-smi not found in PATH (host GPU/runtime passthrough may be missing)"
  fi
}

summary() {
  echo
  color "1;32" "Environment verification complete."
  echo "If CUDA is expected but unavailable:"
  echo " - Ensure host NVIDIA driver and nvidia-container-toolkit are installed."
  echo " - Ensure the container is started with --gpus all (devcontainer sets this)."
  echo " - Rebuild container after driver/toolkit updates."
}

main() {
  section "Tooling"
  check_bin uv || exit 1
  check_bin git || exit 1

  section "Python"
  check_python_version || exit 1

  section "Libraries"
  if ! check_imports_and_versions; then
    warn "Some libraries failed to import"
  fi

  section "CUDA"
  print_cuda_runtime_info

  check_caches_rw

  summary
}

if [[ "${BASH_SOURCE[0]:-}" == "$0" ]]; then
  main "$@"
fi
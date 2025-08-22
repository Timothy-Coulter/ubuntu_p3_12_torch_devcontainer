#!/usr/bin/env bash
# Enhanced verification script with performance tests and better error handling
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
    local version=""
    case "$1" in
      "uv") version="$(uv --version 2>/dev/null | cut -d' ' -f2 || echo "unknown")" ;;
      "git") version="$(git --version 2>/dev/null | cut -d' ' -f3 || echo "unknown")" ;;
      "docker") version="$(docker --version 2>/dev/null | cut -d' ' -f3 | sed 's/,//' || echo "unknown")" ;;
    esac
    pass "$1 found: $(command -v "$1") ${version:+(v$version)}"
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
  
  if [[ ! -x "$py" ]]; then
    fail "Python executable not found: $py"
    return 1
  fi
  
  info "Using Python at: ${py}"
  
  local version
  version="$("$py" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null)" || {
    fail "Cannot get Python version"
    return 1
  }
  
  echo "Python version: ${version}"
  
  # Require 3.12.x
  if ! "$py" -c 'import sys; sys.exit(0 if (sys.version_info[:2] == (3,12)) else 1)' 2>/dev/null; then
    fail "Python 3.12 is required, found: ${version}"
    return 1
  fi
  
  pass "Python 3.12 verified"
  
  # Check if it's in virtual environment
  if "$py" -c 'import sys; sys.exit(0 if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix) else 1)' 2>/dev/null; then
    pass "Running in virtual environment"
  else
    warn "Not running in virtual environment"
  fi
}

check_imports_and_versions() {
  local py
  py="$(python_exec)"
  
  section "Core Dependencies"
  
  # Check core packages with detailed output
  "$py" - <<'PY'
import importlib
import json
import sys
import time

# Define packages to check
core_pkgs = [
    ("numpy", "Scientific computing"),
    ("pandas", "Data manipulation"),
    ("scipy", "Scientific algorithms"),
    ("sklearn", "Machine learning"),
    ("matplotlib", "Plotting")
]

ml_pkgs = [
    ("transformers", "Transformer models"),
    ("datasets", "ML datasets"),
    ("accelerate", "Distributed training"),
    ("tqdm", "Progress bars"),
    ("safetensors", "Safe tensor storage")
]

optional_pkgs = [
    ("seaborn", "Statistical plots"),
    ("plotly", "Interactive plots"),
    ("jupyterlab", "Jupyter environment"),
    ("ipykernel", "Jupyter kernel")
]

def check_package_group(packages, group_name):
    print(f"\n{group_name}:")
    failed = []
    
    for pkg_name, description in packages:
        try:
            start_time = time.time()
            module = importlib.import_module(pkg_name)
            import_time = time.time() - start_time
            
            version = getattr(module, "__version__", "unknown")
            print(f"  ✓ {pkg_name:<15} v{version:<12} ({description}) - {import_time:.3f}s")
            
        except ImportError as e:
            print(f"  ✗ {pkg_name:<15} MISSING      ({description})")
            failed.append(pkg_name)
        except Exception as e:
            print(f"  ⚠ {pkg_name:<15} ERROR        ({str(e)[:50]}...)")
            failed.append(pkg_name)
    
    return failed

# Check all package groups
core_failed = check_package_group(core_pkgs, "Core Scientific Computing")
ml_failed = check_package_group(ml_pkgs, "Machine Learning")
opt_failed = check_package_group(optional_pkgs, "Optional Packages")

# Summary
total_failed = len(core_failed) + len(ml_failed) + len(opt_failed)
if total_failed == 0:
    print(f"\n✓ All packages imported successfully!")
else:
    print(f"\n⚠ {total_failed} packages had issues")
    if core_failed:
        print(f"  Core failures: {', '.join(core_failed)}")
    if ml_failed:
        print(f"  ML failures: {', '.join(ml_failed)}")

sys.exit(1 if (core_failed or ml_failed) else 0)
PY

  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    pass "Package verification completed successfully"
  else
    fail "Some critical packages are missing"
    return 1
  fi
}

check_pytorch_cuda() {
  local py
  py="$(python_exec)"
  
  section "PyTorch and CUDA"
  
  "$py" - <<'PY'
import json
import sys
import time

try:
    print("Testing PyTorch import...")
    start_time = time.time()
    import torch
    import_time = time.time() - start_time
    
    info = {
        "torch_ok": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "import_time": round(import_time, 3),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else "unknown",
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() and torch.backends.cudnn.is_available() else None,
    }
    
    print(f"✓ PyTorch v{info['torch_version']} imported in {info['import_time']}s")
    print(f"  CUDA available: {info['cuda_available']}")
    print(f"  CUDA devices: {info['cuda_device_count']}")
    
    if info['cuda_available']:
        print(f"  CUDA version: {info['cuda_version']}")
        if info['cudnn_version']:
            print(f"  cuDNN version: {info['cudnn_version']}")
            
        # Get device information
        devices = []
        for i in range(info['cuda_device_count']):
            try:
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "index": i,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessors": props.multi_processor_count
                }
                devices.append(device_info)
                print(f"  Device {i}: {device_info['name']} ({device_info['memory_gb']}GB, CC {device_info['compute_capability']})")
            except Exception as e:
                print(f"  Device {i}: Error getting properties - {e}")
        
        # Quick CUDA functionality test
        try:
            print("\nTesting CUDA functionality...")
            device = torch.device('cuda:0')
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            torch.cuda.synchronize()
            
            start_time = time.time()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            cuda_time = time.time() - start_time
            
            print(f"✓ CUDA matrix multiplication test: {cuda_time:.4f}s")
            
        except Exception as e:
            print(f"✗ CUDA functionality test failed: {e}")
    
    # Check additional torch components
    try:
        import torchvision
        print(f"✓ torchvision v{torchvision.__version__}")
    except ImportError:
        print("⚠ torchvision not available")
    
    try:
        import torchaudio
        print(f"✓ torchaudio v{torchaudio.__version__}")
    except ImportError:
        print("⚠ torchaudio not available")

except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ PyTorch verification failed: {e}")
    sys.exit(1)

print("\n✓ PyTorch verification completed")
PY

  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    pass "PyTorch verification completed"
  else
    fail "PyTorch verification failed"
    return 1
  fi
}

check_caches_rw() {
  section "Cache and Data Directories"
  
  local hf="${HF_HOME:-/home/vscode/.cache/huggingface}"
  local tf="${TRANSFORMERS_CACHE:-$hf}"
  local th="${TORCH_HOME:-/home/vscode/.cache/torch}"
  local kg="${KAGGLE_CONFIG_DIR:-/home/vscode/.kaggle}"
  local uv_cache="${HOME}/.cache/uv"
  local data_dir="${DATA_DIR:-${PWD}/data}"
  
  echo "Environment variables:"
  echo "  HF_HOME=${hf}"
  echo "  TRANSFORMERS_CACHE=${tf}"
  echo "  TORCH_HOME=${th}"
  echo "  KAGGLE_CONFIG_DIR=${kg}"
  echo "  UV_CACHE=${uv_cache}"
  echo "  DATA_DIR=${data_dir}"
  echo
  
  # Create directories
  local dirs=("$hf" "$tf" "$th" "$kg" "$uv_cache" "$data_dir")
  for d in "${dirs[@]}"; do
    mkdir -p "$d" 2>/dev/null || true
  done
  
  local ok=0
  local total=${#dirs[@]}
  
  for d in "${dirs[@]}"; do
    if [[ -d "$d" ]] && [[ -w "$d" ]]; then
      # Test actual write capability
      local test_file="${d}/.write_test_$$"
      if echo "test" > "$test_file" 2>/dev/null && rm "$test_file" 2>/dev/null; then
        pass "RW OK: $d"
        ok=$((ok+1))
      else
        warn "Write test failed: $d"
      fi
    else
      warn "Not accessible: $d"
    fi
  done
  
  # Write verification sentinel files
  echo "$(date)" > "$hf/.verify" 2>/dev/null || true
  echo "$(date)" > "$th/.verify" 2>/dev/null || true
  echo "$(date)" > "$data_dir/.verify" 2>/dev/null || true
  
  if [[ "$ok" -ge 4 ]]; then
    pass "Cache/data directories verification passed (${ok}/${total})"
  else
    warn "Some cache/data directories have issues (${ok}/${total} accessible)"
    info "Check devcontainer volume mounts if directories are not writable"
  fi
}

print_cuda_runtime_info() {
  section "CUDA Runtime Environment"
  
  if command -v nvidia-smi >/dev/null 2>&1; then
    info "NVIDIA System Management Interface:"
    if nvidia-smi 2>/dev/null; then
      pass "nvidia-smi executed successfully"
    else
      warn "nvidia-smi returned non-zero exit code"
    fi
  else
    warn "nvidia-smi not found in PATH"
    info "This is normal if running without GPU or if NVIDIA drivers are not installed"
  fi
  
  # Check for Docker GPU runtime
  if command -v docker >/dev/null 2>&1; then
    if docker info 2>/dev/null | grep -q "nvidia"; then
      pass "Docker NVIDIA runtime detected"
    else
      info "Docker NVIDIA runtime not detected (may be normal)"
    fi
  fi
}

check_performance() {
  section "Performance Benchmarks"
  
  local py
  py="$(python_exec)"
  
  "$py" - <<'PY'
import time
import sys

def benchmark_python():
    """Basic Python performance test"""
    start = time.time()
    result = sum(i**2 for i in range(100000))
    duration = time.time() - start
    print(f"Python compute (sum of squares): {duration:.3f}s")
    return duration

def benchmark_numpy():
    """NumPy performance test"""
    try:
        import numpy as np
        
        # Matrix multiplication test
        start = time.time()
        a = np.random.randn(1000, 1000)
        b = np.random.randn(1000, 1000)
        c = np.dot(a, b)
        duration = time.time() - start
        print(f"NumPy matrix multiply (1000x1000): {duration:.3f}s")
        
        return duration
    except ImportError:
        print("NumPy not available for benchmarking")
        return None

def benchmark_torch_cpu():
    """PyTorch CPU performance test"""
    try:
        import torch
        
        start = time.time()
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        c = torch.mm(a, b)
        duration = time.time() - start
        print(f"PyTorch CPU matrix multiply (1000x1000): {duration:.3f}s")
        
        return duration
    except ImportError:
        print("PyTorch not available for CPU benchmarking")
        return None

def benchmark_torch_gpu():
    """PyTorch GPU performance test"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("CUDA not available for GPU benchmarking")
            return None
            
        device = torch.device('cuda:0')
        
        # Warm up
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)
        torch.cuda.synchronize()
        _ = torch.mm(a, b)
        torch.cuda.synchronize()
        
        # Actual benchmark
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        torch.cuda.synchronize()
        
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        duration = time.time() - start
        
        print(f"PyTorch GPU matrix multiply (1000x1000): {duration:.3f}s")
        
        return duration
    except Exception as e:
        print(f"GPU benchmarking failed: {e}")
        return None

# Run benchmarks
print("🔥 Performance Benchmarks")
print("=" * 50)

python_time = benchmark_python()
numpy_time = benchmark_numpy()
torch_cpu_time = benchmark_torch_cpu()
torch_gpu_time = benchmark_torch_gpu()

print("=" * 50)

# Performance analysis
if numpy_time and python_time:
    speedup = python_time / numpy_time
    print(f"NumPy speedup over Python: {speedup:.1f}x")

if torch_cpu_time and torch_gpu_time:
    speedup = torch_cpu_time / torch_gpu_time
    print(f"GPU speedup over CPU: {speedup:.1f}x")

if torch_cpu_time and numpy_time:
    ratio = torch_cpu_time / numpy_time
    print(f"PyTorch/NumPy CPU ratio: {ratio:.2f}")

print("\nBenchmark completed successfully ✓")
PY

  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    pass "Performance benchmarks completed"
  else
    warn "Some performance tests failed"
  fi
}

check_environment_config() {
  section "Environment Configuration"
  
  # Check important environment variables
  local env_vars=(
    "PYTHONDONTWRITEBYTECODE:Python bytecode generation"
    "PYTHONUNBUFFERED:Python output buffering"
    "HF_HOME:Hugging Face cache"
    "TORCH_HOME:PyTorch cache"
    "CUDA_VISIBLE_DEVICES:CUDA device visibility"
  )
  
  for var_desc in "${env_vars[@]}"; do
    local var="${var_desc%:*}"
    local desc="${var_desc#*:}"
    local value="${!var:-}"
    
    if [[ -n "$value" ]]; then
      pass "$var = $value ($desc)"
    else
      info "$var not set ($desc)"
    fi
  done
  
  # Check if .env.local exists
  if [[ -f ".env.local" ]]; then
    local key_count
    key_count=$(grep -c "^[A-Z].*=" ".env.local" 2>/dev/null || echo "0")
    pass ".env.local found with $key_count environment variables"
  else
    info ".env.local not found (API keys not configured)"
    info "Run './dev.sh setup-keys' to configure API keys"
  fi
}

summary() {
  echo
  color "1;32" "🎉 Environment Verification Summary"
  echo "=" * 50
  
  local issues_found=false
  
  # Check if major components are working
  if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    fail "Python not found"
    issues_found=true
  fi
  
  if ! python_exec -c "import torch" 2>/dev/null; then
    fail "PyTorch not importable"
    issues_found=true
  fi
  
  if [[ "$issues_found" == "true" ]]; then
    echo
    color "1;31" "⚠️  Issues Found - Troubleshooting Steps:"
    echo "1. Ensure virtual environment is activated: source .venv/bin/activate"
    echo "2. Sync dependencies: ./dev.sh sync"
    echo "3. If CUDA issues: check GPU passthrough and rebuild container"
    echo "4. For import errors: clear caches with ./dev.sh clean"
    return 1
  else
    echo
    color "1;32" "✅ Environment verification completed successfully!"
    echo
    echo "Your development environment is ready for:"
    echo "  • Deep learning with PyTorch"
    echo "  • Machine learning with transformers"
    echo "  • Data science with NumPy/Pandas"
    echo "  • Interactive development with Jupyter"
    echo
    echo "Next steps:"
    echo "  • Run './dev.sh jupyter' to start Jupyter Lab"
    echo "  • Run './dev.sh benchmark' for detailed performance tests"
    echo "  • Run './dev.sh setup-keys' to configure API keys"
    return 0
  fi
}

main() {
  local start_time
  start_time=$(date +%s)
  
  color "1;36" "🔍 Environment Verification Script"
  color "1;36" "=================================="
  
  section "System Tools"
  local tools_ok=true
  check_bin uv || tools_ok=false
  check_bin git || tools_ok=false
  command -v docker >/dev/null && check_bin docker
  
  if [[ "$tools_ok" == "false" ]]; then
    fail "Critical tools missing"
    exit 1
  fi
  
  section "Python Environment"
  if ! check_python_version; then
    fail "Python environment check failed"
    exit 1
  fi
  
  # Continue with other checks
  check_imports_and_versions || warn "Package verification had issues"
  check_pytorch_cuda || warn "PyTorch/CUDA verification had issues"
  check_caches_rw
  print_cuda_runtime_info
  check_performance
  check_environment_config
  
  local end_time
  end_time=$(date +%s)
  local duration=$((end_time - start_time))
  
  echo
  info "Verification completed in ${duration}s"
  
  summary
}

# Trap for cleanup
trap 'echo; warn "Verification interrupted"' INT TERM

if [[ "${BASH_SOURCE[0]:-}" == "$0" ]]; then
  main "$@"
fi
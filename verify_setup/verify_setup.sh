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
  
  # Require at least Python 3.10
  if ! "$py" -c 'import sys; sys.exit(0 if (sys.version_info.major == 3 and sys.version_info.minor >= 10) else 1)' 2>/dev/null; then
    fail "Python >=3.10 is required, found: ${version}"
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

check_cuda_compatibility() {
    local pytorch_cuda_version
    pytorch_cuda_version=$(python "$(dirname "${BASH_SOURCE[0]}")/get_pytorch_cuda.py" 2>/dev/null || echo "none")
    
    if [[ "$pytorch_cuda_version" == "none" ]]; then
        fail "PyTorch was not compiled with CUDA support"
        info "Reinstall with: uv sync --extra torch-cu128"
        return 1
    fi
    
    info "PyTorch CUDA version: $pytorch_cuda_version"
    
    # Verify CUDA 12.8.1 compatibility
    if [[ "$pytorch_cuda_version" != "12.8.1" ]]; then
        warn "PyTorch CUDA version ($pytorch_cuda_version) differs from expected CUDA 12.8.1"
        info "This container is optimized for CUDA 12.8.1"
    else
        pass "PyTorch CUDA 12.8.1 compatibility confirmed"
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
import os

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
    
    # Enhanced CUDA diagnostics
    if not info['cuda_available']:
        print("\n🔍 CUDA Diagnostics:")
        
        # Check CUDA environment variables
        cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'NVIDIA_VISIBLE_DEVICES']
        for var in cuda_vars:
            value = os.environ.get(var, 'Not set')
            print(f"  {var}: {value}")
        
        # Check if PyTorch was compiled with CUDA
        print(f"  PyTorch CUDA version: {info['cuda_version']}")
        if info['cuda_version'] == "unknown" or info['cuda_version'] is None:
            print("  ⚠ PyTorch was not compiled with CUDA support")
            print("  💡 Reinstall PyTorch with CUDA: uv sync --extra torch-cu128")
        else:
            print("  ✓ PyTorch was compiled with CUDA support")
            if info['cuda_version'] != "12.8.1":
                print(f"  ⚠ Expected CUDA 12.8.1, found: {info['cuda_version']}")
            print("  ⚠ CUDA runtime libraries may be missing or incompatible")
            print("  💡 Check CUDA runtime installation in container")
    
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
        
        # Enhanced CUDA functionality test
        try:
            print("\nTesting CUDA functionality...")
            device = torch.device('cuda:0')
            
            # Memory allocation test
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            torch.cuda.synchronize()
            
            # Computation test
            start_time = time.time()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            cuda_time = time.time() - start_time
            
            print(f"✓ CUDA matrix multiplication: {cuda_time:.4f}s")
            
            # Memory management test
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_cached = torch.cuda.memory_reserved(0) / 1024**2  # MB
            print(f"✓ GPU memory - Allocated: {memory_allocated:.1f}MB, Cached: {memory_cached:.1f}MB")
            
            # Cleanup
            del x, y, z
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ CUDA functionality test failed: {e}")
            print("💡 This may indicate CUDA driver/runtime incompatibility")
    
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
    print("💡 Run: uv sync --extra torch-cu128")
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

check_cuda_linkage() {
  section "CUDA Library Linkage"

  local py
  py="$(python_exec)"
  local torch_lib
  torch_lib=$($py -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_cuda.so'))" 2>/dev/null || echo "")

  if [[ -f "$torch_lib" ]]; then
    info "Checking linkage for: $torch_lib"
    if ldd "$torch_lib" | grep -q "/usr/local/cuda"; then
      pass "PyTorch is linked against system CUDA libraries"
    else
      warn "PyTorch may be using bundled CUDA libraries!"
      info "ldd output:"
      ldd "$torch_lib" | grep cuda || true
    fi
  else
    warn "Could not locate libtorch_cuda.so — skipping linkage check"
  fi
}

check_caches_rw() {
  section "Cache and Data Directories"
  
  local hf="${HF_HOME:-/home/ubuntu/.cache/huggingface}"
  local tf="${TRANSFORMERS_CACHE:-$hf}"
  local th="${TORCH_HOME:-/home/ubuntu/.cache/torch}"
  local kg="${KAGGLE_CONFIG_DIR:-/home/ubuntu/.kaggle}"
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
  
  # Check CUDA library paths
  info "CUDA library path checks:"
  local cuda_paths=("/usr/local/cuda-12.8.1/lib64" "/usr/local/cuda/lib64" "/usr/lib/x86_64-linux-gnu")
  for path in "${cuda_paths[@]}"; do
    if [[ -d "$path" ]]; then
      local cuda_libs=$(find "$path" -name "libcuda*" 2>/dev/null | wc -l)
      if [[ "$cuda_libs" -gt 0 ]]; then
        pass "CUDA libraries found in: $path ($cuda_libs files)"
      else
        info "Directory exists but no CUDA libraries: $path"
      fi
    else
      info "CUDA path not found: $path"
    fi
  done
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
        
        # Warm up GPU
        for _ in range(3):
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
        
        # Cleanup
        del a, b, c
        torch.cuda.empty_cache()
        
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
    if speedup < 2:
        print("  ⚠ GPU speedup is low - check GPU utilization")
    elif speedup > 50:
        print("  🚀 Excellent GPU acceleration!")

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
    "NVIDIA_VISIBLE_DEVICES:NVIDIA device visibility"
    "NVIDIA_DRIVER_CAPABILITIES:NVIDIA driver capabilities"
    "LD_LIBRARY_PATH:Library search path"
    "CUDA_HOME:CUDA 12.8.1 installation path"
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
    
    # Check file permissions
    local perms
    perms=$(stat -c "%a" ".env.local" 2>/dev/null || echo "unknown")
    if [[ "$perms" == "600" ]]; then
      pass ".env.local has secure permissions (600)"
    else
      warn ".env.local permissions: $perms (should be 600)"
      info "Fix with: chmod 600 .env.local"
    fi
  else
    info ".env.local not found (API keys not configured)"
    info "Run './dev.sh setup-keys' to configure API keys"
  fi
}

check_container_health() {
  section "Container Health"
  
  # Check available resources
  if command -v free >/dev/null 2>&1; then
    local memory_info
    memory_info=$(free -h | grep "Mem:")
    info "Memory: $memory_info"
  fi
  
  if command -v df >/dev/null 2>&1; then
    local disk_usage
    disk_usage=$(df -h /workspaces 2>/dev/null | tail -n 1 || echo "N/A")
    info "Workspace disk usage: $disk_usage"
  fi
  
  # Check network connectivity
  if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    pass "Network connectivity OK"
  else
    warn "Network connectivity issues detected"
  fi
  
  # Check if running in container
  if [[ -f "/.dockerenv" ]] || grep -q "docker\|lxc" /proc/1/cgroup 2>/dev/null; then
    pass "Running in container environment"
  else
    info "Not running in container (native environment)"
  fi
}

summary() {
  echo
  color "1;32" "🎉 Environment Verification Summary"
  echo "=" * 50
  
  local issues_found=false
  local warnings_found=false
  
  # Check if major components are working
  if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    fail "Python not found"
    issues_found=true
  fi
  
  if ! python_exec -c "import torch" 2>/dev/null; then
    fail "PyTorch not importable"
    issues_found=true
  fi
  
  # Check CUDA availability
  local cuda_available
  cuda_available=$(python_exec -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
  
  if [[ "$cuda_available" == "False" ]]; then
    warn "CUDA not available in PyTorch"
    warnings_found=true
    echo
    color "1;33" "🔧 CUDA Troubleshooting:"
    echo "1. Ensure GPU passthrough: Check Docker Desktop GPU settings"
    echo "2. Verify NVIDIA runtime: docker info | grep nvidia"
    echo "3. Rebuild container with CUDA runtime libraries"
    echo "4. Check CUDA compatibility: nvidia-smi vs PyTorch CUDA version"
  fi
  
  if [[ "$issues_found" == "true" ]]; then
    echo
    color "1;31" "⚠️  Critical Issues Found - Troubleshooting Steps:"
    echo "1. Ensure virtual environment is activated: source .venv/bin/activate"
    echo "2. Sync dependencies: ./dev.sh sync"
    echo "3. If import errors: clear caches with ./dev.sh clean"
    echo "4. Rebuild container: Ctrl+Shift+P → 'Dev Container: Rebuild Container'"
    return 1
  elif [[ "$warnings_found" == "true" ]]; then
    echo
    color "1;33" "⚠️  Warnings Found - Environment Partially Ready"
    echo
    echo "Your environment is functional but has some issues:"
    echo "  • Basic development capabilities available"
    echo "  • CPU-based ML/DL training supported"
    echo "  • GPU acceleration not available"
    echo
    echo "For GPU support, address the CUDA issues above."
    return 0
  else
    echo
    color "1;32" "✅ Environment verification completed successfully!"
    echo
    echo "Your development environment is ready for:"
    echo "  • Deep learning with PyTorch${cuda_available:+ (GPU accelerated)}"
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
  
  color "1;36" "🔍 Enhanced Environment Verification"
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
  
  # Continue with other checks (don't exit on warnings)
  check_imports_and_versions || warn "Package verification had issues"
  check_pytorch_cuda || warn "PyTorch/CUDA verification had issues"
  check_cuda_compatibility || warn "CUDA compatibility check had issues"
  check_cuda_linkage
  check_caches_rw
  print_cuda_runtime_info
  check_performance
  check_environment_config
  check_container_health
  
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
import torch
import os

print("=== PyTorch GPU Diagnostics ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Check environment variables
print("\n[Environment Variables]")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"NVIDIA_DRIVER_CAPABILITIES: {os.environ.get('NVIDIA_DRIVER_CAPABILITIES', 'Not set')}")

# Additional diagnostics when CUDA is not available
if not torch.cuda.is_available():
    print("\n[Detailed Diagnostics]")
    print(f"CUDA Build Info: {torch._C._cuda_getCompiledVersion() if hasattr(torch._C, '_cuda_getCompiledVersion') else 'N/A'}")
    print(f"CUDA Runtime Version: {torch._C._cuda_getRuntimeVersion() if hasattr(torch._C, '_cuda_getRuntimeVersion') else 'N/A'}")
    print(f"CUDA Driver Version: {torch._C._cuda_getDriverVersion() if hasattr(torch._C, '_cuda_getDriverVersion') else 'N/A'}")
    
    # Check library paths
    print("\n[Library Paths]")
    try:
        import ctypes
        cudart = ctypes.CDLL('libcudart.so')
        print("✅ libcudart.so found")
    except Exception as e:
        print(f"❌ libcudart.so not found: {e}")

if torch.cuda.is_available():
    print("\n[GPU Details]")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
    
    # Test GPU operations
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = x * 2
        print(f"GPU Tensor Operation: {y}")
        print("✅ GPU tensor operations successful!")
    except Exception as e:
        print(f"❌ GPU operations failed: {e}")

print("\n=== Diagnostics Complete ===")
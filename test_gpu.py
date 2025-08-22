import torch

print("=== PyTorch GPU Test ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
    
    # Test creating a tensor on GPU
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"GPU Tensor: {x}")
        print("✅ GPU tensor creation successful!")
    except Exception as e:
        print(f"❌ GPU tensor creation failed: {e}")
else:
    print("❌ CUDA is not available")

print("=== Test Complete ===")
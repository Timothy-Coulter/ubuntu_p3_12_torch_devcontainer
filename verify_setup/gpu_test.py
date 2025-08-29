import torch
import sys

def test_gpu_functionality():
    try:
        if not torch.cuda.is_available():
            print('⚠️  CUDA not available - will run in CPU mode')
            return True
        device_count = torch.cuda.device_count()
        print(f'🖥️  Found {device_count} CUDA device(s)')
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f'   Device {i}: {props.name} ({props.total_memory // 1024**3}GB)')
        print('🧪 Testing GPU tensor operations...')
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.mm(x, y)
        print(f'✅ GPU tensor test passed - Result shape: {z.shape}')
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(0) // 1024**2
        print(f'📊 GPU memory allocated: {memory_allocated}MB')
        return True
    except Exception as e:
        print(f'❌ GPU test failed: {e}')
        return False

if __name__ == "__main__":
    success = test_gpu_functionality()
    sys.exit(0 if success else 1)
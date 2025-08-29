import torch
print(f"PyTorch {torch.__version__}")
print(f"Built with CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
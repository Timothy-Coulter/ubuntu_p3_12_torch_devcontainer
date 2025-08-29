# tests/test_gpu.py
import pytest
import torch


@pytest.mark.gpu
def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA should be available"


@pytest.mark.gpu
def test_gpu_computation():
    if torch.cuda.is_available():
        x = torch.randn(100, 100).cuda()
        y = torch.mm(x, x.t())
        assert y.device.type == "cuda"

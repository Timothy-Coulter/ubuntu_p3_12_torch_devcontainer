# Jupyter Notebook Setup Guide

## ✅ Current Status: Fully Configured!

The Jupyter environment is now properly configured and ready to use. Here's everything you need to know:

## Available Kernels

Two Jupyter kernels are available:

1. **`Python 3.12 (torch_starter)`** ← **RECOMMENDED**
   - Full project environment with all dependencies
   - Located at: `/home/vscode/.local/share/jupyter/kernels/torch_starter-3.12`
   - Python path: `/workspaces/associative_wavelets/.venv/bin/python`

2. **`python3`** 
   - Basic environment
   - Located at: `/workspaces/associative_wavelets/.venv/share/jupyter/kernels/python3`

## How to Use Notebooks in VS Code

### Method 1: VS Code Jupyter Extension (Recommended)
1. Open the notebook file (`notebooks/dtdwt_demonstration.ipynb`)
2. Click on the kernel selector in the top-right corner
3. Choose **"Python 3.12 (torch_starter)"** from the dropdown
4. Start running cells!

### Method 2: Jupyter Lab
```bash
# In terminal
source .venv/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```
Then open the provided URL in your browser.

## Verified Dependencies

All required packages are installed and working:
- ✅ `ipykernel>=6.29` - Jupyter kernel support
- ✅ `PyWavelets` - Wavelet transforms
- ✅ `scikit-image` - Image processing
- ✅ `NumPy, Matplotlib, Pandas` - Scientific computing
- ✅ `PyTorch + CUDA 12.4` - Deep learning framework

## Testing the Setup

Run this in a notebook cell to verify everything works:

```python
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage import data

# Test basic functionality
print("✅ All imports successful!")
print(f"PyWavelets version: {pywt.__version__}")

# Test image processing
camera = data.camera()[:64, :64]
coeffs = pywt.dwt2(camera, 'db4')
print("✅ Basic wavelet transform working!")

# Test plotting
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(camera, cmap='gray')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(coeffs[0], cmap='gray')
plt.title('Approximation')
plt.show()
print("✅ Plotting working!")
```

## Troubleshooting

### Issue: "Running cells with 'Python' requires the ipykernel package"
**Solution:** ✅ **FIXED** - Kernel was reinstalled with correct path

### Issue: Kernel not found
**Solution:** Select "Python 3.12 (torch_starter)" kernel in VS Code

### Issue: Import errors
**Solution:** Ensure virtual environment is activated:
```bash
source .venv/bin/activate
```

## Ready to Run

The DTDWT demonstration notebook is fully functional and ready to run:
- **Location:** [`notebooks/dtdwt_demonstration.ipynb`](dtdwt_demonstration.ipynb)
- **Kernel:** Python 3.12 (torch_starter)
- **Status:** ✅ All dependencies installed and tested

Enjoy exploring the Dual-Tree Discrete Wavelet Transform! 🌊
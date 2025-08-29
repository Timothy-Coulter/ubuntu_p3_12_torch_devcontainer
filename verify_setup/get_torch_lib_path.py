import torch
import os
print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_cuda.so'))
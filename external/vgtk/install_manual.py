import os
import subprocess
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Set environment variables
os.environ['TORCH_CUDA_ARCH_LIST'] = "6.0;6.1;7.0;7.5;8.0"

# Build the extension manually
extension = CUDAExtension(
    name='vgtk.cuda.gathering',
    sources=[
        'vgtk/cuda/gathering_cuda.cpp',
        'vgtk/cuda/gathering_cuda_kernel.cu'
    ],
    extra_compile_args={
        'cxx': ['-std=c++14', '-D_GLIBCXX_USE_CXX11_ABI=0'],
        'nvcc': ['-std=c++14', '--expt-relaxed-constexpr']
    }
)

builder = BuildExtension()
# This approach might work better

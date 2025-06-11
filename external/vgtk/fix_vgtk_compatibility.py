import os
import re
import glob

def patch_gathering_cuda():
    """Fix the deprecated tensor.type() calls"""
    file_path = "vgtk/cuda/gathering_cuda.cpp"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace deprecated calls
    content = re.sub(r'x\.type\(\)\.is_cuda\(\)', 'x.is_cuda()', content)
    content = re.sub(r'AT_ASSERTM\(([^,]+)\.type\(\)\.is_cuda\(\)', r'AT_ASSERTM(\1.is_cuda()', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Patched {file_path}")

def patch_setup_py():
    """Add compatibility flags to setup.py"""
    if not os.path.exists("setup.py"):
        return
        
    with open("setup.py", 'r') as f:
        content = f.read()
    
    # Add compatibility flags
    if 'extra_compile_args' not in content:
        content = content.replace(
            'CUDAExtension(',
            '''CUDAExtension(
            extra_compile_args={
                'cxx': ['-std=c++14', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': ['-std=c++14']
            },'''
        )
        
        with open("setup.py", 'w') as f:
            f.write(content)
        print("Patched setup.py")

def patch_cuda_kernel():
    """Fix CUDA kernel compatibility issues"""
    kernel_file = "vgtk/cuda/gathering_cuda_kernel.cu"
    if not os.path.exists(kernel_file):
        return
        
    with open(kernel_file, 'r') as f:
        content = f.read()
    
    # Add compatibility headers at the top
    if '#include <cuda_runtime.h>' not in content:
        content = '#include <cuda_runtime.h>\n' + content
    
    with open(kernel_file, 'w') as f:
        f.write(content)
    print(f"Patched {kernel_file}")

# Run all patches
patch_gathering_cuda()
patch_setup_py()
patch_cuda_kernel()
print("All patches applied!")

import re

with open("setup.py", 'r') as f:
    content = f.read()

# Replace the entire cuda_extension function with a more compatible version
new_function = '''def cuda_extension(package_name, ext):
    import os
    ext_name = f"{package_name}.cuda.{ext}"
    ext_cpp = f"{package_name}/cuda/{ext}_cuda.cpp"
    ext_cu = f"{package_name}/cuda/{ext}_cuda_kernel.cu"
    
    return CUDAExtension(
        ext_name, 
        [ext_cpp, ext_cu],
        extra_compile_args={
            'cxx': ['-g', '-std=c++14', '-D_GLIBCXX_USE_CXX11_ABI=0', '-Wno-error', '-fpermissive', '-fno-strict-aliasing'],
            'nvcc': ['-O2', '-std=c++14', '--expt-relaxed-constexpr', 
                     '-gencode=arch=compute_86,code=sm_86',
                     '--compiler-options', '-std=c++14,-fpermissive,-fno-strict-aliasing']
        }
    )'''

# Find and replace the cuda_extension function
pattern = r'def cuda_extension\(package_name, ext\):.*?return CUDAExtension\(.*?\)'
content = re.sub(pattern, new_function, content, flags=re.DOTALL)

with open("setup.py", 'w') as f:
    f.write(content)

print("Applied aggressive compatibility patch to setup.py")

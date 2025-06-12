import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def cuda_extension(package_name, ext):
    ext_name = f"{package_name}.cuda.{ext}"
    ext_cpp = f"{package_name}/cuda/{ext}_cuda.cpp"
    ext_cu = f"{package_name}/cuda/{ext}_cuda_kernel.cu"
    
    return CUDAExtension(
        ext_name, 
        [ext_cpp, ext_cu], 
        extra_compile_args={'cxx': ['-g', '-std=c++14']}
    )

setup(
    name='vgtk',
    version='0.1.0',
    author='vgtk',
    packages=find_packages(),
    ext_modules=[
        cuda_extension('vgtk', 'gathering'),
        cuda_extension('vgtk', 'grouping'),
        cuda_extension('vgtk', 'zpconv'),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
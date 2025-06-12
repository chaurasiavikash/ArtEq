Fixing vgtk CUDA Extension Compilation on Modern Toolchains: A Detailed Report

This document outlines the diagnosis and solution for a critical compilation failure when installing the vgtk library on a modern system with GCC 11 and CUDA 11.5.
1. Problem Statement

When attempting to install the vgtk library, which is a dependency for the ArtEq project, the build process would fail during the compilation of its CUDA extensions (gathering, grouping, zpconv).

Environment:

    OS: Ubuntu 22.04

    Compiler: GCC 11 (g++-11)

    CUDA Toolkit: 11.5

    PyTorch: 1.12.1 (built with CUDA 11.3)

Initial Error:
The compilation would consistently fail with the following error message, originating from the C++ standard library headers:

/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’

2. Root Cause Analysis

The error stems from a fundamental incompatibility between the CUDA toolkit's compiler (nvcc) and the C++ standard library provided by newer versions of GCC.

    Compiler Mismatch: nvcc from CUDA 11.5 uses its own internal C++ compiler front-end to parse .cu files. This front-end is not fully compliant with the C++17 standard.

    Modern C++ Headers: GCC 11's standard library headers (like <functional>, where std::function is defined) use modern C++17 features (specifically, if constexpr with parameter packs).

    The Conflict: When nvcc processes a .cu file that includes PyTorch headers (torch/torch.h or ATen/ATen.h), it transitively includes GCC 11's <functional> header. The CUDA compiler front-end cannot parse the C++17 syntax within this header, leading to the "parameter packs not expanded" error.

Initial attempts to solve this by passing compiler flags were unsuccessful because the vgtk source code structure forced nvcc to see the problematic headers.
3. The Solution: Code Refactoring and Separation

The correct and most robust solution is to refactor the source code to enforce a clean separation between the C++ code (which uses PyTorch Tensors) and the pure CUDA code. This ensures that nvcc never sees the complex PyTorch C++ headers it cannot parse.

This was achieved by applying the following three-step pattern to each CUDA module (gathering, grouping, zpconv):

Step 1: Create a C-style Interface Header (.h)
A new header file (e.g., gathering_cuda_kernel.h) was created for each module. This file defines the "bridge" between the C++ and CUDA code. It contains only C-style function declarations that use basic data types and raw pointers (float*, int*, etc.), and it does not include any PyTorch headers.

Step 2: Isolate the CUDA Kernel File (.cu)
The CUDA implementation file (e.g., gathering_cuda_kernel.cu) was modified to:

    Remove all PyTorch-related #include directives.

    Include only the new C-style header (gathering_cuda_kernel.h) and standard CUDA headers.

    Contain the __global__ CUDA kernel implementations, which operate on the raw pointers.

    Implement "launcher" functions (e.g., gather_points_forward_launcher) which are C-style functions that take raw pointers and are responsible for configuring and calling the CUDA kernels.

Step 3: Update the C++ Wrapper File (.cpp)
The C++ file (e.g., gathering_cuda.cpp) was modified to:

    Include the new C-style header (gathering_cuda_kernel.h).

    Keep all the PyTorch-related logic.

    In the functions exposed to Python, extract the raw data pointers from the at::Tensor objects using .data_ptr<T>().

    Call the C-style launcher functions from the .cu file, passing the tensor dimensions and raw pointers as arguments.

This refactoring guarantees that g++-11 compiles the Tensor logic and nvcc compiles the CUDA kernels, with a clean C-style interface between them, completely avoiding the compiler incompatibility.
4. Final Build Configuration (setup.py)

With the code correctly structured, the setup.py file was simplified significantly. The final cuda_extension function is clean and clear:

def cuda_extension(package_name, ext):
    ext_name = f"{package_name}.cuda.{ext


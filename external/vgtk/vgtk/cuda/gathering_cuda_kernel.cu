#include "gathering_cuda_kernel.h"
#include <cuda_runtime.h>

#if __CUDA_ARCH__ < 600 && defined(__CUDA_ARCH__)
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

namespace { // Anonymous namespace for the kernels

// The __global__ CUDA kernel for the forward pass
__global__ void gather_points_forward_kernel_impl(
    const float* __restrict__ points,
    const int* __restrict__ idx,
    float* __restrict__ out,
    int num_points, int num_channels, int num_indices)
{
    const int bn = blockIdx.y;
    const int aidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pn = aidx / num_channels;
    const int ci = aidx % num_channels;

    if (pn < num_indices && ci < num_channels) {
        const int np = num_points;
        const int nc = num_channels;
        const int nm = num_indices;
        const int a = idx[bn * nm + pn];
        out[(bn * nc + ci) * nm + pn] = points[(bn * nc + ci) * np + a];
    }
}

// The __global__ CUDA kernel for the backward pass
__global__ void gather_points_backward_kernel_impl(
    const float* __restrict__ grad_out,
    const int* __restrict__ idx,
    float* __restrict__ grad_points,
    int num_points, int num_channels, int num_indices)
{
    const int bn = blockIdx.y;
    const int aidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pn = aidx / num_channels;
    const int ci = aidx % num_channels;

    if (pn < num_indices) {
        const int np = num_points;
        const int nc = num_channels;
        const int nm = num_indices;
        const int a = idx[bn * nm + pn];
        atomicAdd(grad_points + (bn * nc + ci) * np + a, grad_out[(bn * nc + ci) * nm + pn]);
    }
}

} // end anonymous namespace

// C-style launcher function for the forward pass
void gather_points_forward_launcher(
    int b, int c, int n, int m,
    const float* points, const int* indices, float* out)
{
    const int threads = 1024;
    const dim3 blocks((m * c + threads - 1) / threads, b);
    gather_points_forward_kernel_impl<<<blocks, threads>>>(points, indices, out, n, c, m);
}

// C-style launcher function for the backward pass
void gather_points_backward_launcher(
    int b, int c, int n, int m,
    const float* grad_out, const int* indices, float* grad_points)
{
    const int threads = 1024;
    const dim3 blocks((n * c + threads - 1) / threads, b);
    gather_points_backward_kernel_impl<<<blocks, threads>>>(grad_out, indices, grad_points, n, c, m);
}
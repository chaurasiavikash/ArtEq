#include <torch/torch.h>
#include <vector>
#include "grouping_cuda_kernel.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor ball_query(
    at::Tensor new_xyz, at::Tensor xyz, const float radius, const int nsample
) {
    CHECK_INPUT(new_xyz);
    CHECK_INPUT(xyz);
    at::Tensor idx = torch::zeros({new_xyz.size(0), new_xyz.size(2), nsample},
                                  at::device(new_xyz.device()).dtype(at::ScalarType::Int));
    
    ball_query_launcher(
        xyz.size(0), xyz.size(2), new_xyz.size(2), radius, nsample,
        new_xyz.data_ptr<float>(), xyz.data_ptr<float>(), idx.data_ptr<int>()
    );
    return idx;
}

at::Tensor furthest_point_sampling(
    at::Tensor source_xyz, const int m
) {
    CHECK_INPUT(source_xyz);
    at::Tensor tmp = torch::full({source_xyz.size(0), source_xyz.size(2)}, 1e10,
                                 at::device(source_xyz.device()).dtype(source_xyz.dtype()));
    at::Tensor sampled_idx = torch::zeros({source_xyz.size(0), m},
                                          at::device(source_xyz.device()).dtype(at::ScalarType::Int));

    furthest_point_sampling_launcher(
        source_xyz.size(0), source_xyz.size(2), m,
        source_xyz.data_ptr<float>(), tmp.data_ptr<float>(), sampled_idx.data_ptr<int>()
    );
    return sampled_idx;
}

// The anchor query functions are not being compiled as their kernels are missing.
// If you need them, they must be refactored in the same way.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query", &ball_query, "ball query (CUDA)");
    m.def("furthest_point_sampling", &furthest_point_sampling, "furthest point sampling (CUDA)");
}
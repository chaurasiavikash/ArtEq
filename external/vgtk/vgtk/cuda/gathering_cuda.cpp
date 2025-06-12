#include <torch/torch.h>
#include <vector>
#include "gathering_cuda_kernel.h" // Include our new C-style interface

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor gather_points_forward(
    at::Tensor support_points, // [nb, c_in, nsupport]
    at::Tensor grouped_indices // [nb, nsample]
) {
    CHECK_INPUT(support_points);
    CHECK_INPUT(grouped_indices);

    at::Tensor output = torch::zeros({support_points.size(0), support_points.size(1), grouped_indices.size(1)},
        at::device(support_points.device()).dtype(at::ScalarType::Float));

    gather_points_forward_launcher(
        support_points.size(0),   // batch size
        support_points.size(1),   // channels
        support_points.size(2),   // num_points
        grouped_indices.size(1),  // num_indices
        support_points.data_ptr<float>(),
        grouped_indices.data_ptr<int>(),
        output.data_ptr<float>()
    );

    return output;
}

at::Tensor gather_points_backward(
    at::Tensor grad_out,        // [nb, c_in, nsample]
    at::Tensor grouped_indices, // [nb, nsample]
    int npoint
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(grouped_indices);

    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), npoint},
        at::device(grad_out.device()).dtype(grad_out.dtype()));

    gather_points_backward_launcher(
        grad_out.size(0),         // batch size
        grad_out.size(1),         // channels
        npoint,                   // num_points
        grad_out.size(2),         // num_indices
        grad_out.data_ptr<float>(),
        grouped_indices.data_ptr<int>(),
        output.data_ptr<float>()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_points_forward", &gather_points_forward, "gathering forward (CUDA)");
    m.def("gather_points_backward", &gather_points_backward, "gathering backward (CUDA)");
}
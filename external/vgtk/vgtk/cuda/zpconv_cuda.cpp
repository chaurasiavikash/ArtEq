#include <torch/torch.h>
#include <vector>
#include "zpconv_cuda_kernel.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor inter_zpconv_forward(
    at::Tensor anchor_neighbors, at::Tensor anchor_weights, at::Tensor support_point_feats
) {
    CHECK_INPUT(anchor_neighbors);
    CHECK_INPUT(anchor_weights);
    CHECK_INPUT(support_point_feats);

    at::Tensor anchor_feats = torch::zeros({anchor_neighbors.size(0), support_point_feats.size(1), anchor_neighbors.size(3), 
                                            anchor_neighbors.size(1), anchor_neighbors.size(2)}, 
                                           at::device(support_point_feats.device()).dtype(support_point_feats.dtype()));

    inter_zpconv_forward_launcher(
        anchor_neighbors.size(0), support_point_feats.size(1), anchor_neighbors.size(1),
        support_point_feats.size(2), anchor_neighbors.size(2), anchor_neighbors.size(3),
        anchor_neighbors.size(4),
        anchor_neighbors.data_ptr<int>(), anchor_weights.data_ptr<float>(),
        support_point_feats.data_ptr<float>(), anchor_feats.data_ptr<float>()
    );
    return anchor_feats;
}

at::Tensor inter_zpconv_backward(
    at::Tensor anchor_neighbors, at::Tensor anchor_weights, at::Tensor grad_anchor_feats, const int npoint
) {
    CHECK_INPUT(anchor_neighbors);
    CHECK_INPUT(anchor_weights);
    CHECK_INPUT(grad_anchor_feats);

    at::Tensor grad_support_point_feats = torch::zeros({anchor_neighbors.size(0), grad_anchor_feats.size(1),
                                                        npoint, anchor_neighbors.size(2)}, 
                                                       at::device(grad_anchor_feats.device()).dtype(grad_anchor_feats.dtype()));

    inter_zpconv_backward_launcher(
        anchor_neighbors.size(0), grad_anchor_feats.size(1), anchor_neighbors.size(1),
        npoint, anchor_neighbors.size(2), anchor_neighbors.size(3),
        anchor_neighbors.size(4),
        anchor_neighbors.data_ptr<int>(), anchor_weights.data_ptr<float>(),
        grad_anchor_feats.data_ptr<float>(), grad_support_point_feats.data_ptr<float>()
    );
    return grad_support_point_feats;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inter_zpconv_forward", &inter_zpconv_forward, "inter conv forward (CUDA)");
    m.def("inter_zpconv_backward", &inter_zpconv_backward, "inter conv backward (CUDA)");
}
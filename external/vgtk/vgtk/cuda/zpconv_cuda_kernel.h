#pragma once

void inter_zpconv_forward_launcher(
    int b, int c_in, int np, int nq, int na, int ks, int ann,
    const int* anchor_neighbors, const float* anchor_weights,
    const float* support_point_feats, float* anchor_feats
);

void inter_zpconv_backward_launcher(
    int b, int c_in, int np, int nq, int na, int ks, int ann,
    const int* anchor_neighbors, const float* anchor_weights,
    const float* grad_anchor_feats, float* grad_support_point_feats
);
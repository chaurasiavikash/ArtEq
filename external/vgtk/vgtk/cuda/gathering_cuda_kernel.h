#pragma once

// C-style function declarations. No PyTorch types are used here.
void gather_points_forward_launcher(
    int b, int c, int n, int m,
    const float* points,
    const int* indices,
    float* out
);

void gather_points_backward_launcher(
    int b, int c, int n, int m,
    const float* grad_out,
    const int* indices,
    float* grad_points
);
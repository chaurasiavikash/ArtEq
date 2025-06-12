#pragma once

// C-style function declarations. No PyTorch types are used here.
void ball_query_launcher(int b, int n, int m, float radius, int nsample,
                         const float* new_xyz, const float* xyz, int* idx);

void furthest_point_sampling_launcher(int b, int n, int m,
                                      const float* dataset, float* temp, int* idx);

// Note: The anchor query functions from the original repo are complex and their kernels
// are not fully defined in the provided code. I have stubbed them out here.
// If they are needed, their implementations will also have to be refactored.
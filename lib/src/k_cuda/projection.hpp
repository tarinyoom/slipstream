#pragma once

namespace slipstream::k_cuda {

void compute_projection(int nx, int ny, const float* obstacle,
                        float* vx, float* vy,
                        float* pressure, float* rhs_scratch,
                        int max_iterations, float tolerance);

} // namespace slipstream::k_cuda

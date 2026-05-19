#pragma once

namespace slipstream::k_cuda {

void compute_projection(int nx, int ny, int nz, const float* obstacle,
                        float* vx, float* vy, float* vz,
                        float* pressure, float* rhs_scratch, float* residuals,
                        int max_iterations, float tolerance);

} // namespace slipstream::k_cuda

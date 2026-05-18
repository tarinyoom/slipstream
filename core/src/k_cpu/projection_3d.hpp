#pragma once

namespace slipstream::k_cpu_3d {

void compute_projection(int nx, int ny, int nz, const float* obstacle,
                        float* vx, float* vy, float* vz,
                        float* pressure, float* rhs_scratch,
                        int max_iterations, float tolerance);

void compute_red_black_gs(int nx, int ny, int nz, const float* obstacle,
                          const float* rhs, float* pressure,
                          int max_iterations, float tolerance);

} // namespace slipstream::k_cpu_3d

#pragma once

namespace slipstream::gpu {

void compute_projection(int nx, int ny, const float* obstacle,
                        float* vx, float* vy,
                        float* pressure, float* rhs_scratch,
                        int max_iterations, float tolerance);

} // namespace slipstream::gpu

#pragma once

namespace slipstream::k_cpu {

void compute_buoyancy(int nx, int ny, float buoyancy, float dt,
                      const float* temperature, float* vx);

} // namespace slipstream::k_cpu

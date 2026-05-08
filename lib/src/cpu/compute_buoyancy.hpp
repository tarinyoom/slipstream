#pragma once

namespace slipstream::cpu {

void compute_buoyancy(int nx, int ny, float buoyancy, float dt,
                      const float* temperature, float* vx);

} // namespace slipstream::cpu

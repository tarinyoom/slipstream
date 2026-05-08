#pragma once

namespace slipstream::gpu {

void apply_buoyancy(int nx, int ny, float buoyancy, float dt,
                    const float* temperature, float* vx);

} // namespace slipstream::gpu

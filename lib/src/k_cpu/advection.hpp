#pragma once

namespace slipstream::k_cpu {

void compute_scalar_advection(int nx, int ny,
                              const float* vx, const float* vy,
                              const float* field_in, float* field_out, float dt);

void compute_velocity_advection(int nx, int ny,
                                float* vx, float* vy,
                                float* scratch, float dt);

} // namespace slipstream::k_cpu

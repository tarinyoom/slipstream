#pragma once

namespace slipstream::k_cuda_3d {

void compute_scalar_advection(int nx, int ny, int nz,
                              const float* vx, const float* vy, const float* vz,
                              const float* field_in, float* field_out, float dt);

void compute_velocity_advection(int nx, int ny, int nz,
                                float* vx, float* vy, float* vz,
                                float* scratch, float dt);

} // namespace slipstream::k_cuda_3d

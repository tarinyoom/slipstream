#pragma once

namespace slipstream::cpu {

void advect_scalar(int nx, int ny,
                   const float* vx, const float* vy,
                   const float* field_in, float* field_out, float dt);

void advect_velocity(int nx, int ny,
                     float* vx, float* vy,
                     float* scratch, float dt);

} // namespace slipstream::cpu

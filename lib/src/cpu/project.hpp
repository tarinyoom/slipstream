#pragma once

namespace slipstream::cpu {

void project(int nx, int ny, const float* obstacle,
             float* vx, float* vy,
             float* pressure, float* rhs_scratch,
             int max_iterations, float tolerance);

void red_black_gs(int nx, int ny, const float* obstacle,
                  const float* rhs, float* pressure,
                  int max_iterations, float tolerance);

} // namespace slipstream::cpu

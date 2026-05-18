#pragma once

#include "state.hpp"

namespace slipstream {

void step_cpu(State& s, float dt,
              int max_iterations = 100, float tolerance = 1e-3f);

void step_3d_cpu(State& s, float dt,
                 int max_iterations = 100, float tolerance = 1e-3f);

#ifdef SLIPSTREAM_HAS_CUDA

void step_cuda(State& s, float dt,
               int max_iterations = 100, float tolerance = 1e-3f);

void step_3d_cuda(State& s, float dt,
                  int max_iterations = 100, float tolerance = 1e-3f);

#endif

} // namespace slipstream

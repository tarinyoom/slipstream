#pragma once

#ifdef SLIPSTREAM_HAS_CUDA

#include "../state.hpp"

namespace slipstream::gpu {

void step(State& s, float dt,
          int max_iterations = 100, float tolerance = 1e-3f);

} // namespace slipstream::gpu

#endif

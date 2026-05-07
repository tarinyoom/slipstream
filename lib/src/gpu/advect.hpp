#pragma once

#include "state.hpp"

namespace slipstream::gpu {

void advect_scalar(State s, float* d_scratch, float dt);

} // namespace slipstream::gpu

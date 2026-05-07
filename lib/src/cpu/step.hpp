#pragma once

#include "state.hpp"

namespace slipstream::cpu {

void step(PersistentState& s, const ScratchState& sc, float dt,
          int max_iterations = 100, float tolerance = 1e-3f);

} // namespace slipstream::cpu

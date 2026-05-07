#pragma once

#include "state.hpp"

namespace slipstream::cpu {

void advect_velocity(const PersistentState& s, float* scratch, float dt);

void advect_scalar(const PersistentState& s, const float* field, float* scratch, float dt);

} // namespace slipstream::cpu

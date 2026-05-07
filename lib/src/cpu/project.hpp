#pragma once

#include "state.hpp"

namespace slipstream::cpu {

void project(const PersistentState& s, float* pressure, int max_iterations, float tolerance);

void red_black_gs(const PersistentState& s, const float* rhs, float* pressure,
                  int max_iterations, float tolerance);

} // namespace slipstream::cpu

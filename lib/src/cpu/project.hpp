#pragma once

/* CPU projection: divergence computation, pressure solve, gradient subtraction. */

#include <span>
#include <vector>

namespace slipstream::cpu {

void project(std::span<const int>           shape,
             std::vector<std::span<float>>& velocity,
             std::span<const bool>          obstacle,
             std::span<float>               pressure,
             int                            max_iterations,
             float                          tolerance);

// Exposed for testing. The red-black Gauss-Seidel solve is otherwise an internal
// step of project(), but declaring it here allows tests to exercise it in isolation
// without going through the full projection pipeline.
//
// Unlike project(), red_black_gs() does NOT zero pressure on entry — callers using
// it directly can warm-start by supplying a non-zero initial guess.
void red_black_gs(std::span<const int>   shape,
                  std::span<const bool>  obstacle,
                  std::span<const float> rhs,
                  std::span<float>       pressure,
                  int                    max_iterations,
                  float                  tolerance);

} // namespace slipstream::cpu

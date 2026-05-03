#pragma once

#include <span>
#include <vector>

namespace slipstream::cpu {

void advect_velocity(std::span<const int>           shape,
                     std::vector<std::span<float>>& velocity,
                     std::span<float>               scratch,
                     float                          dt);

} // namespace slipstream::cpu

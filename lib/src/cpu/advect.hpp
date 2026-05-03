#pragma once

#include <span>
#include <vector>

namespace slipstream::cpu {

void advect_velocity(std::span<const int>           shape,
                     std::vector<std::span<float>>& velocity,
                     std::span<float>               scratch,
                     float                          dt);

void advect_scalar(std::span<const int>                       shape,
                   std::span<const float>                     field,
                   std::span<float>                           scratch,
                   const std::vector<std::span<float>>&       velocity,
                   float                                      dt);

} // namespace slipstream::cpu

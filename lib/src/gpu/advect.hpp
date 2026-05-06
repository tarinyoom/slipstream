#pragma once

#include <span>
#include <vector>

namespace slipstream::gpu {

void advect_scalar(std::span<const int>                 shape,
                   std::span<const float>               field,
                   std::span<float>                     scratch,
                   const std::vector<std::span<float>>& velocity,
                   float                                dt);

} // namespace slipstream::gpu

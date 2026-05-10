#pragma once

namespace slipstream::k_cuda {

void compute_cooling(int total, float cooling, float dt, float* temperature);

} // namespace slipstream::k_cuda

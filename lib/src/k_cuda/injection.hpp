#pragma once

namespace slipstream::k_cuda {

void compute_injection(int n_emitters, int total,
                       const float* masks,
                       const float* emitter_densities,
                       const float* emitter_temperatures,
                       float* density, float* temperature);

} // namespace slipstream::k_cuda

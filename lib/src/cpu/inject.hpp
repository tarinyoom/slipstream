#pragma once

namespace slipstream::cpu {

void inject_emitters(int n_emitters, int total,
                     const float* masks,
                     const float* emitter_densities,
                     const float* emitter_temperatures,
                     float* density, float* temperature);

} // namespace slipstream::cpu

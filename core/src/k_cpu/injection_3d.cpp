#include "injection_3d.hpp"

namespace slipstream::k_cpu_3d {

void compute_injection(int n_emitters, int total,
                       const float* masks,
                       const float* emitter_densities,
                       const float* emitter_temperatures,
                       float* density, float* temperature)
{
    for (int e = 0; e < n_emitters; ++e) {
        for (int c = 0; c < total; ++c) {
            if (masks[e * total + c] != 0.0f) {
                density[c] = emitter_densities[e];
                if (temperature && emitter_temperatures)
                    temperature[c] = emitter_temperatures[e];
            }
        }
    }
}

} // namespace slipstream::k_cpu_3d

#include "step.hpp"
#include "advect.hpp"
#include "project.hpp"

#include <algorithm>

namespace slipstream::cpu {

void step(PersistentState& s, const ScratchState& sc, float dt,
          int max_iterations, float tolerance)
{
    const int Nx    = s.nx;
    const int Ny    = s.ny;
    const int total = Nx * Ny;

    if (s.n_emitters > 0 && s.emitter_masks) {
        for (int e = 0; e < s.n_emitters; ++e) {
            for (int c = 0; c < total; ++c) {
                if (s.emitter_masks[e * total + c] != 0.0f) {
                    s.density[c] = s.emitter_densities[e];
                    if (s.temperature && s.emitter_temperatures)
                        s.temperature[c] = s.emitter_temperatures[e];
                }
            }
        }
    }

    if (s.velocity) {
        advect_scalar(s, s.density, sc.tmp, dt);
        std::copy(sc.tmp, sc.tmp + total, s.density);

        if (s.temperature) {
            advect_scalar(s, s.temperature, sc.tmp, dt);
            std::copy(sc.tmp, sc.tmp + total, s.temperature);
        }

        advect_velocity(s, sc.tmp, dt);

        if (s.buoyancy != 0.0f && s.temperature) {
            for (int i = 1; i < Nx; ++i)
                for (int j = 0; j < Ny; ++j) {
                    float t_avg = 0.5f * (s.temperature[(i - 1) * Ny + j]
                                        + s.temperature[ i      * Ny + j]);
                    s.velocity[i * Ny + j] += s.buoyancy * t_avg * dt;
                }
        }

        if (s.cooling != 0.0f && s.temperature) {
            for (int c = 0; c < total; ++c)
                s.temperature[c] *= (1.0f - s.cooling * dt);
        }

        project(s, sc.pressure, max_iterations, tolerance);
    }
}

} // namespace slipstream::cpu

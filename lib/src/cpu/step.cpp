#include "step.hpp"
#include "compute_injection.hpp"
#include "compute_advection.hpp"
#include "compute_buoyancy.hpp"
#include "compute_cooling.hpp"
#include "compute_projection.hpp"

#include <algorithm>

namespace slipstream::cpu {

void step(PersistentState& s, const ScratchState& sc, float dt,
          int max_iterations, float tolerance)
{
    const int Nx    = s.nx;
    const int Ny    = s.ny;
    const int total = Nx * Ny;
    float* vx = s.velocity;
    float* vy = s.velocity + (Nx + 1) * Ny;

    compute_injection(s.n_emitters, total, s.emitter_masks,
                      s.emitter_densities, s.emitter_temperatures,
                      s.density, s.temperature);

    compute_scalar_advection(Nx, Ny, vx, vy, s.density, sc.tmp, dt);
    std::copy(sc.tmp, sc.tmp + total, s.density);

    compute_scalar_advection(Nx, Ny, vx, vy, s.temperature, sc.tmp, dt);
    std::copy(sc.tmp, sc.tmp + total, s.temperature);

    compute_velocity_advection(Nx, Ny, vx, vy, sc.tmp, dt);

    compute_buoyancy(Nx, Ny, s.buoyancy, dt, s.temperature, vx);

    compute_cooling(total, s.cooling, dt, s.temperature);

    compute_projection(Nx, Ny, s.obstacle, vx, vy,
                       sc.pressure, sc.tmp, max_iterations, tolerance);
}

} // namespace slipstream::cpu

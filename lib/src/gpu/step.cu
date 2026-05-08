#include "step.hpp"
#include "inject.hpp"
#include "advect.hpp"
#include "buoyancy.hpp"
#include "cooling.hpp"
#include "project.hpp"

#include <cuda_runtime.h>
#include <cstddef>

namespace slipstream::gpu {

void step(PersistentState& s, const ScratchState& sc, float dt,
          int max_iterations, float tolerance)
{
    const int Nx    = s.nx;
    const int Ny    = s.ny;
    const int total = Nx * Ny;
    float* vx = s.velocity;
    float* vy = s.velocity + (Nx + 1) * Ny;

    inject_emitters(s.n_emitters, total, s.emitter_masks,
                    s.emitter_densities, s.emitter_temperatures,
                    s.density, s.temperature);

    advect_scalar(Nx, Ny, vx, vy, s.density, sc.tmp, dt);
    cudaMemcpy(s.density, sc.tmp, (std::size_t)total * sizeof(float),
               cudaMemcpyDeviceToDevice);

    advect_scalar(Nx, Ny, vx, vy, s.temperature, sc.tmp, dt);
    cudaMemcpy(s.temperature, sc.tmp, (std::size_t)total * sizeof(float),
               cudaMemcpyDeviceToDevice);

    advect_velocity(Nx, Ny, vx, vy, sc.tmp, dt);

    apply_buoyancy(Nx, Ny, s.buoyancy, dt, s.temperature, vx);

    apply_cooling(total, s.cooling, dt, s.temperature);

    project(Nx, Ny, s.obstacle, vx, vy,
            sc.pressure, sc.tmp, max_iterations, tolerance);
}

} // namespace slipstream::gpu

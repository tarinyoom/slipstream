#include "step.hpp"

#include "k_cpu/injection.hpp"
#include "k_cpu/advection.hpp"
#include "k_cpu/buoyancy.hpp"
#include "k_cpu/cooling.hpp"
#include "k_cpu/projection.hpp"

#include <algorithm>

namespace slipstream {

void step_cpu(State& s, float dt, int max_iterations, float tolerance)
{
    using namespace k_cpu;
    const int Nx    = s.nx;
    const int Ny    = s.ny;
    const int total = Nx * Ny;
    float* vx = s.velocity;
    float* vy = s.velocity + (Nx + 1) * Ny;

    compute_injection(s.n_emitters, total, s.emitter_masks,
                      s.emitter_densities, s.emitter_temperatures,
                      s.density, s.temperature);

    compute_scalar_advection(Nx, Ny, vx, vy, s.density, s.tmp, dt);
    std::copy(s.tmp, s.tmp + total, s.density);

    compute_scalar_advection(Nx, Ny, vx, vy, s.temperature, s.tmp, dt);
    std::copy(s.tmp, s.tmp + total, s.temperature);

    compute_velocity_advection(Nx, Ny, vx, vy, s.tmp, dt);

    compute_buoyancy(Nx, Ny, s.buoyancy, dt, s.temperature, vx);

    compute_cooling(total, s.cooling, dt, s.temperature);

    compute_projection(Nx, Ny, s.obstacle, vx, vy,
                       s.pressure, s.tmp, max_iterations, tolerance);
}

#ifdef SLIPSTREAM_HAS_CUDA

} // namespace slipstream

#include "k_cuda/injection.hpp"
#include "k_cuda/advection.hpp"
#include "k_cuda/buoyancy.hpp"
#include "k_cuda/cooling.hpp"
#include "k_cuda/projection.hpp"

#include <cuda_runtime.h>
#include <cstddef>

namespace slipstream {

void step_cuda(State& s, float dt, int max_iterations, float tolerance)
{
    using namespace k_cuda;
    const int Nx    = s.nx;
    const int Ny    = s.ny;
    const int total = Nx * Ny;
    float* vx = s.velocity;
    float* vy = s.velocity + (Nx + 1) * Ny;

    compute_injection(s.n_emitters, total, s.emitter_masks,
                      s.emitter_densities, s.emitter_temperatures,
                      s.density, s.temperature);

    compute_scalar_advection(Nx, Ny, vx, vy, s.density, s.tmp, dt);
    cudaMemcpy(s.density, s.tmp, (std::size_t)total * sizeof(float),
               cudaMemcpyDeviceToDevice);

    compute_scalar_advection(Nx, Ny, vx, vy, s.temperature, s.tmp, dt);
    cudaMemcpy(s.temperature, s.tmp, (std::size_t)total * sizeof(float),
               cudaMemcpyDeviceToDevice);

    compute_velocity_advection(Nx, Ny, vx, vy, s.tmp, dt);

    compute_buoyancy(Nx, Ny, s.buoyancy, dt, s.temperature, vx);

    compute_cooling(total, s.cooling, dt, s.temperature);

    compute_projection(Nx, Ny, s.obstacle, vx, vy,
                       s.pressure, s.tmp, max_iterations, tolerance);
}

#endif // SLIPSTREAM_HAS_CUDA

} // namespace slipstream

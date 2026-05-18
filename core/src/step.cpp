#include "step.hpp"

#include "k_cpu/injection.hpp"
#include "k_cpu/advection.hpp"
#include "k_cpu/buoyancy.hpp"
#include "k_cpu/cooling.hpp"
#include "k_cpu/projection.hpp"
#include "k_cpu/injection_3d.hpp"
#include "k_cpu/advection_3d.hpp"
#include "k_cpu/buoyancy_3d.hpp"
#include "k_cpu/cooling_3d.hpp"
#include "k_cpu/projection_3d.hpp"

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

    compute_scalar_advection(Nx, Ny, vx, vy, s.density, s.scratch, dt);
    std::copy(s.scratch, s.scratch + total, s.density);

    compute_scalar_advection(Nx, Ny, vx, vy, s.temperature, s.scratch, dt);
    std::copy(s.scratch, s.scratch + total, s.temperature);

    compute_velocity_advection(Nx, Ny, vx, vy, s.scratch, dt);

    compute_buoyancy(Nx, Ny, s.buoyancy, dt, s.temperature, vx);

    compute_cooling(total, s.cooling, dt, s.temperature);

    compute_projection(Nx, Ny, s.obstacle, vx, vy,
                       s.pressure, s.scratch, max_iterations, tolerance);
}

void step_3d_cpu(State& s, float dt, int max_iterations, float tolerance)
{
    using namespace k_cpu_3d;
    const int Nx    = s.nx;
    const int Ny    = s.ny;
    const int Nz    = s.nz;
    const int total = Nx * Ny * Nz;
    float* vx = s.velocity;
    float* vy = vx + (Nx + 1) * Ny * Nz;
    float* vz = vy + Nx * (Ny + 1) * Nz;

    compute_injection(s.n_emitters, total, s.emitter_masks,
                      s.emitter_densities, s.emitter_temperatures,
                      s.density, s.temperature);

    compute_scalar_advection(Nx, Ny, Nz, vx, vy, vz, s.density, s.scratch, dt);
    std::copy(s.scratch, s.scratch + total, s.density);

    compute_scalar_advection(Nx, Ny, Nz, vx, vy, vz, s.temperature, s.scratch, dt);
    std::copy(s.scratch, s.scratch + total, s.temperature);

    compute_velocity_advection(Nx, Ny, Nz, vx, vy, vz, s.scratch, dt);

    compute_buoyancy(Nx, Ny, Nz, s.buoyancy, dt, s.temperature, vx);

    compute_cooling(total, s.cooling, dt, s.temperature);

    compute_projection(Nx, Ny, Nz, s.obstacle, vx, vy, vz,
                       s.pressure, s.scratch, max_iterations, tolerance);
}

#ifdef SLIPSTREAM_HAS_CUDA

} // namespace slipstream

#include "k_cuda/injection.hpp"
#include "k_cuda/advection.hpp"
#include "k_cuda/buoyancy.hpp"
#include "k_cuda/cooling.hpp"
#include "k_cuda/projection.hpp"
#include "k_cuda/injection_3d.hpp"
#include "k_cuda/advection_3d.hpp"
#include "k_cuda/buoyancy_3d.hpp"
#include "k_cuda/cooling_3d.hpp"
#include "k_cuda/projection_3d.hpp"

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

    compute_scalar_advection(Nx, Ny, vx, vy, s.density, s.scratch, dt);
    cudaMemcpy(s.density, s.scratch, (std::size_t)total * sizeof(float),
               cudaMemcpyDeviceToDevice);

    compute_scalar_advection(Nx, Ny, vx, vy, s.temperature, s.scratch, dt);
    cudaMemcpy(s.temperature, s.scratch, (std::size_t)total * sizeof(float),
               cudaMemcpyDeviceToDevice);

    compute_velocity_advection(Nx, Ny, vx, vy, s.scratch, dt);

    compute_buoyancy(Nx, Ny, s.buoyancy, dt, s.temperature, vx);

    compute_cooling(total, s.cooling, dt, s.temperature);

    compute_projection(Nx, Ny, s.obstacle, vx, vy,
                       s.pressure, s.scratch, s.scratch + total,
                       max_iterations, tolerance);
}

void step_3d_cuda(State& s, float dt, int max_iterations, float tolerance)
{
    using namespace k_cuda_3d;
    const int Nx    = s.nx;
    const int Ny    = s.ny;
    const int Nz    = s.nz;
    const int total = Nx * Ny * Nz;
    float* vx = s.velocity;
    float* vy = vx + (Nx + 1) * Ny * Nz;
    float* vz = vy + Nx * (Ny + 1) * Nz;

    compute_injection(s.n_emitters, total, s.emitter_masks,
                      s.emitter_densities, s.emitter_temperatures,
                      s.density, s.temperature);

    compute_scalar_advection(Nx, Ny, Nz, vx, vy, vz, s.density, s.scratch, dt);
    cudaMemcpy(s.density, s.scratch, (std::size_t)total * sizeof(float),
               cudaMemcpyDeviceToDevice);

    compute_scalar_advection(Nx, Ny, Nz, vx, vy, vz, s.temperature, s.scratch, dt);
    cudaMemcpy(s.temperature, s.scratch, (std::size_t)total * sizeof(float),
               cudaMemcpyDeviceToDevice);

    compute_velocity_advection(Nx, Ny, Nz, vx, vy, vz, s.scratch, dt);

    compute_buoyancy(Nx, Ny, Nz, s.buoyancy, dt, s.temperature, vx);

    compute_cooling(total, s.cooling, dt, s.temperature);

    compute_projection(Nx, Ny, Nz, s.obstacle, vx, vy, vz,
                       s.pressure, s.scratch, s.scratch + total,
                       max_iterations, tolerance);
}

#endif // SLIPSTREAM_HAS_CUDA

} // namespace slipstream

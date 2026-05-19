#include "state.hpp"

#include <algorithm>
#include <stdexcept>

namespace slipstream {

namespace {

std::size_t scratch_floats(int nx, int ny, int nz)
{
    const std::size_t total = (std::size_t)nx * ny * nz;
    const std::size_t f_x   = (std::size_t)(nx + 1) * ny * nz;
    const std::size_t f_y   = (std::size_t)nx * (ny + 1) * nz;
    const std::size_t f_z   = (std::size_t)nx * ny * (nz + 1);
    const std::size_t max_faces = std::max({f_x, f_y, f_z});
#ifdef SLIPSTREAM_HAS_CUDA
    return std::max(max_faces, 2 * total);
#else
    return max_faces;
#endif
}

} // anonymous namespace

std::size_t required_state_bytes(int nx, int ny, int nz,
                                 int n_emitters,
                                 bool allocate_solver_workspace)
{
    const std::size_t total  = (std::size_t)nx * ny * nz;
    const std::size_t v_size = (std::size_t)(nx + 1) * ny * nz
                             + (std::size_t)nx * (ny + 1) * nz
                             + (std::size_t)nx * ny * (nz + 1);

    std::size_t sz = 0;
    sz += total  * sizeof(float);
    sz += v_size * sizeof(float);
    sz += total  * sizeof(float);
    sz += total  * sizeof(float);
    if (n_emitters > 0) {
        sz += (std::size_t)n_emitters * total * sizeof(float);
        sz += (std::size_t)n_emitters         * sizeof(float);
        sz += (std::size_t)n_emitters         * sizeof(float);
    }
    if (allocate_solver_workspace) {
        sz += total * sizeof(float);
        sz += scratch_floats(nx, ny, nz) * sizeof(float);
    }
    return sz;
}

void init_state(State& state,
                void* buf, std::size_t len,
                int nx, int ny, int nz,
                int n_emitters,
                bool allocate_solver_workspace)
{
    const std::size_t need = required_state_bytes(nx, ny, nz, n_emitters,
                                                   allocate_solver_workspace);
    if (len < need)
        throw std::runtime_error("init_state: buffer too small for requested layout");

    const std::size_t total = (std::size_t)nx * ny * nz;
    const std::size_t v_size = (std::size_t)(nx + 1) * ny * nz
                             + (std::size_t)nx * (ny + 1) * nz
                             + (std::size_t)nx * ny * (nz + 1);

    auto* p = static_cast<char*>(buf);
    auto next = [&](std::size_t count) -> float* {
        float* ptr = reinterpret_cast<float*>(p);
        p += count * sizeof(float);
        return ptr;
    };

    state.nx          = nx;
    state.ny          = ny;
    state.nz          = nz;
    state.n_emitters  = n_emitters;
    state.density     = next(total);
    state.velocity    = next(v_size);
    state.temperature = next(total);
    state.obstacle    = next(total);

    if (n_emitters > 0) {
        state.emitter_masks        = next((std::size_t)n_emitters * total);
        state.emitter_densities    = next(n_emitters);
        state.emitter_temperatures = next(n_emitters);
    } else {
        state.emitter_masks        = nullptr;
        state.emitter_densities    = nullptr;
        state.emitter_temperatures = nullptr;
    }

    if (allocate_solver_workspace) {
        state.pressure = next(total);
        state.scratch  = next(scratch_floats(nx, ny, nz));
    } else {
        state.pressure = nullptr;
        state.scratch  = nullptr;
    }

    state.viscosity = 0.0f;
    state.buoyancy  = 0.0f;
    state.cooling   = 0.0f;
    state.vorticity = 0.0f;
}

} // namespace slipstream

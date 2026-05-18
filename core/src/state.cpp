#include "state.hpp"

#include <algorithm>
#include <stdexcept>

namespace slipstream {

static std::size_t scratch_floats(int nx, int ny)
{
    const std::size_t total     = (std::size_t)nx * ny;
    const std::size_t max_faces = total + (std::size_t)std::max(nx, ny);
#ifdef SLIPSTREAM_HAS_CUDA
    return std::max(max_faces, 2 * total);
#else
    return max_faces;
#endif
}

std::size_t required_state_bytes(std::span<const int> dims,
                                 int n_emitters,
                                 bool allocate_solver_workspace)
{
    const int nx = dims[0];
    const int ny = dims[1];
    const int total  = nx * ny;
    const int v_size = (nx + 1) * ny + nx * (ny + 1);

    std::size_t sz = 0;
    sz += (std::size_t)total  * sizeof(float);
    sz += (std::size_t)v_size * sizeof(float);
    sz += (std::size_t)total  * sizeof(float);
    sz += (std::size_t)total  * sizeof(float);
    if (n_emitters > 0) {
        sz += (std::size_t)n_emitters * total * sizeof(float);
        sz += (std::size_t)n_emitters         * sizeof(float);
        sz += (std::size_t)n_emitters         * sizeof(float);
    }
    if (allocate_solver_workspace) {
        sz += (std::size_t)total * sizeof(float);
        sz += scratch_floats(nx, ny) * sizeof(float);
    }
    return sz;
}

void init_state(State& state,
                void* buf, std::size_t len,
                std::span<const int> dims,
                int n_emitters,
                bool allocate_solver_workspace)
{
    const std::size_t need = required_state_bytes(dims, n_emitters, allocate_solver_workspace);
    if (len < need)
        throw std::runtime_error("init_state: buffer too small for requested layout");

    const int nx      = dims[0];
    const int ny      = dims[1];
    const int total   = nx * ny;
    const int v_size  = (nx + 1) * ny + nx * (ny + 1);

    auto* p = static_cast<char*>(buf);
    auto next = [&](std::size_t count) -> float* {
        float* ptr = reinterpret_cast<float*>(p);
        p += count * sizeof(float);
        return ptr;
    };

    state.nx          = nx;
    state.ny          = ny;
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
        state.scratch  = next(scratch_floats(nx, ny));
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

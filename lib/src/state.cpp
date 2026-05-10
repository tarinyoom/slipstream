#include "state.hpp"

#include <algorithm>
#include <stdexcept>

namespace slipstream {

std::size_t required_state_bytes(std::span<const int> dims,
                                 int n_emitters,
                                 bool allocate_scratch)
{
    const int nx = dims[0];
    const int ny = dims[1];
    const int total     = nx * ny;
    const int v_size    = (nx + 1) * ny + nx * (ny + 1);
    const int max_faces = std::max((nx + 1) * ny, nx * (ny + 1));

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
    if (allocate_scratch) {
        sz += (std::size_t)total     * sizeof(float);
        sz += (std::size_t)max_faces * sizeof(float);
    }
    return sz;
}

void init_state(State& state,
                void* buf, std::size_t len,
                std::span<const int> dims,
                int n_emitters,
                bool allocate_scratch)
{
    const std::size_t need = required_state_bytes(dims, n_emitters, allocate_scratch);
    if (len < need)
        throw std::runtime_error("init_state: buffer too small for requested layout");

    const int nx        = dims[0];
    const int ny        = dims[1];
    const int total     = nx * ny;
    const int v_size    = (nx + 1) * ny + nx * (ny + 1);
    const int max_faces = std::max((nx + 1) * ny, nx * (ny + 1));

    auto* p = static_cast<char*>(buf);
    auto next = [&](int count) -> float* {
        float* ptr = reinterpret_cast<float*>(p);
        p += (std::size_t)count * sizeof(float);
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
        state.emitter_masks        = next(n_emitters * total);
        state.emitter_densities    = next(n_emitters);
        state.emitter_temperatures = next(n_emitters);
    } else {
        state.emitter_masks        = nullptr;
        state.emitter_densities    = nullptr;
        state.emitter_temperatures = nullptr;
    }

    if (allocate_scratch) {
        state.pressure = next(total);
        state.tmp      = next(max_faces);
    } else {
        state.pressure = nullptr;
        state.tmp      = nullptr;
    }

    state.viscosity = 0.0f;
    state.buoyancy  = 0.0f;
    state.cooling   = 0.0f;
    state.vorticity = 0.0f;
}

} // namespace slipstream

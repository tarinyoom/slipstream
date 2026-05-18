#include "state.hpp"

#include <algorithm>
#include <stdexcept>

namespace slipstream {

namespace {

std::size_t scratch_floats_2d(int nx, int ny)
{
    const std::size_t total     = (std::size_t)nx * ny;
    const std::size_t max_faces = total + (std::size_t)std::max(nx, ny);
#ifdef SLIPSTREAM_HAS_CUDA
    return std::max(max_faces, 2 * total);
#else
    return max_faces;
#endif
}

std::size_t scratch_floats_3d(int nx, int ny, int nz)
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

std::size_t required_state_bytes_2d(int nx, int ny, int n_emitters, bool ws)
{
    const std::size_t total  = (std::size_t)nx * ny;
    const std::size_t v_size = (std::size_t)(nx + 1) * ny + (std::size_t)nx * (ny + 1);

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
    if (ws) {
        sz += total * sizeof(float);
        sz += scratch_floats_2d(nx, ny) * sizeof(float);
    }
    return sz;
}

std::size_t required_state_bytes_3d(int nx, int ny, int nz, int n_emitters, bool ws)
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
    if (ws) {
        sz += total * sizeof(float);
        sz += scratch_floats_3d(nx, ny, nz) * sizeof(float);
    }
    return sz;
}

} // anonymous namespace

std::size_t required_state_bytes(std::span<const int> dims,
                                 int n_emitters,
                                 bool allocate_solver_workspace)
{
    if (dims.size() == 2)
        return required_state_bytes_2d(dims[0], dims[1], n_emitters, allocate_solver_workspace);
    if (dims.size() == 3)
        return required_state_bytes_3d(dims[0], dims[1], dims[2], n_emitters, allocate_solver_workspace);
    throw std::runtime_error("required_state_bytes: dims must have size 2 or 3");
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

    const bool is_3d = dims.size() == 3;
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = is_3d ? dims[2] : 0;

    const std::size_t total = is_3d
        ? (std::size_t)nx * ny * nz
        : (std::size_t)nx * ny;
    const std::size_t v_size = is_3d
        ? (std::size_t)(nx + 1) * ny * nz
          + (std::size_t)nx * (ny + 1) * nz
          + (std::size_t)nx * ny * (nz + 1)
        : (std::size_t)(nx + 1) * ny
          + (std::size_t)nx * (ny + 1);

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
        state.scratch  = next(is_3d ? scratch_floats_3d(nx, ny, nz)
                                    : scratch_floats_2d(nx, ny));
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

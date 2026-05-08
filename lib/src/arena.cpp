#include "arena.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

#ifdef SLIPSTREAM_HAS_CUDA
#include "gpu/memory.hpp"
#include "gpu/transfer.hpp"
#endif

namespace slipstream {

static std::size_t arena_buf_size(int nx, int ny, int n_emitters, bool allocate_scratch) {
    const int total  = nx * ny;
    const int v_size = (nx + 1) * ny + nx * (ny + 1);
    const int max_faces = std::max((nx + 1) * ny, nx * (ny + 1));

    std::size_t sz = 0;
    sz += (std::size_t)total   * sizeof(float); // density
    sz += (std::size_t)v_size  * sizeof(float); // velocity
    sz += (std::size_t)total   * sizeof(float); // temperature
    sz += (std::size_t)total   * sizeof(float); // obstacle
    if (n_emitters > 0) {
        sz += (std::size_t)n_emitters * total * sizeof(float); // emitter_masks
        sz += (std::size_t)n_emitters         * sizeof(float); // emitter_densities
        sz += (std::size_t)n_emitters         * sizeof(float); // emitter_temperatures
    }
    if (allocate_scratch) {
        sz += (std::size_t)total     * sizeof(float); // pressure
        sz += (std::size_t)max_faces * sizeof(float); // tmp
    }
    return sz;
}

CalculationArena::CalculationArena(Backend b, [[maybe_unused]] int n_dims, const int* dims,
                                   int n_emitters, bool allocate_scratch)
    : buf(nullptr), backend(b)
{
    assert(n_dims == 2);

    const int nx = dims[0];
    const int ny = dims[1];
    const int total    = nx * ny;
    const int v_size   = (nx + 1) * ny + nx * (ny + 1);
    const int max_faces = std::max((nx + 1) * ny, nx * (ny + 1));

    std::size_t sz = arena_buf_size(nx, ny, n_emitters, allocate_scratch);

    if (b == Backend::GPU) {
#ifdef SLIPSTREAM_HAS_CUDA
        buf = gpu::gpu_alloc(sz);
#else
        throw std::runtime_error("GPU CalculationArena not supported in this build");
#endif
    } else {
        buf = new char[sz]();
    }

    char* p = buf;
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

    state.viscosity  = 0.0f;
    state.buoyancy   = 0.0f;
    state.cooling    = 0.0f;
    state.vorticity  = 0.0f;

    if (allocate_scratch) {
        ScratchState sc{};
        sc.pressure = next(total);
        sc.tmp      = next(max_faces);
        scratch     = sc;
    }
}

CalculationArena::~CalculationArena() {
#ifdef SLIPSTREAM_HAS_CUDA
    if (backend == Backend::GPU)
        gpu::gpu_free(buf);
    else
        delete[] buf;
#else
    delete[] buf;
#endif
}

void copy(const CalculationArena& src, CalculationArena& dst) {
    const PersistentState& s = src.state;
    PersistentState&       d = dst.state;
    assert(s.nx == d.nx && s.ny == d.ny && s.n_emitters == d.n_emitters);

    const int total  = s.nx * s.ny;
    const int v_size = (s.nx + 1) * s.ny + s.nx * (s.ny + 1);

    auto do_copy = [&](void* dp, const void* sp, std::size_t sz) {
#ifdef SLIPSTREAM_HAS_CUDA
        if (src.backend == Backend::CPU && dst.backend == Backend::GPU)
            gpu::memcpy_h2d(dp, sp, sz);
        else if (src.backend == Backend::GPU && dst.backend == Backend::CPU)
            gpu::memcpy_d2h(dp, sp, sz);
        else if (src.backend == Backend::GPU && dst.backend == Backend::GPU)
            gpu::memcpy_d2d(dp, sp, sz);
        else
            std::memcpy(dp, sp, sz);
#else
        std::memcpy(dp, sp, sz);
#endif
    };

    do_copy(d.density,     s.density,     (std::size_t)total  * sizeof(float));
    do_copy(d.velocity,    s.velocity,    (std::size_t)v_size * sizeof(float));
    do_copy(d.temperature, s.temperature, (std::size_t)total  * sizeof(float));
    do_copy(d.obstacle,    s.obstacle,    (std::size_t)total  * sizeof(float));

    if (s.n_emitters > 0 && s.emitter_masks) {
        do_copy(d.emitter_masks,        s.emitter_masks,
                (std::size_t)s.n_emitters * total * sizeof(float));
        do_copy(d.emitter_densities,    s.emitter_densities,
                (std::size_t)s.n_emitters * sizeof(float));
        do_copy(d.emitter_temperatures, s.emitter_temperatures,
                (std::size_t)s.n_emitters * sizeof(float));
    }

    d.viscosity  = s.viscosity;
    d.buoyancy   = s.buoyancy;
    d.cooling    = s.cooling;
    d.vorticity  = s.vorticity;
}

} // namespace slipstream

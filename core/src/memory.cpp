#include "memory.hpp"

#ifdef SLIPSTREAM_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace slipstream {

void* host_alloc(std::size_t sz) {
    return new char[sz]();
}

void host_free(void* p) {
    delete[] static_cast<char*>(p);
}

#ifdef SLIPSTREAM_HAS_CUDA

void* device_alloc(std::size_t sz) {
    void* p = nullptr;
    if (cudaMalloc(&p, sz) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    if (cudaMemset(p, 0, sz) != cudaSuccess) {
        cudaFree(p);
        throw std::runtime_error("cudaMemset (device_alloc) failed");
    }
    return p;
}

void device_free(void* p) {
    cudaFree(p);
}

namespace {

void memcpy_h2d(void* dst, const void* src, std::size_t sz) {
    if (cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy H2D failed");
}

void memcpy_d2h(void* dst, const void* src, std::size_t sz) {
    if (cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy D2H failed");
}

} // anonymous namespace

void upload(const State& s, State& d) {
    assert(s.nx == d.nx && s.ny == d.ny && s.n_emitters == d.n_emitters);

    const int total  = s.nx * s.ny;
    const int v_size = (s.nx + 1) * s.ny + s.nx * (s.ny + 1);

    memcpy_h2d(d.density,     s.density,     (std::size_t)total  * sizeof(float));
    memcpy_h2d(d.velocity,    s.velocity,    (std::size_t)v_size * sizeof(float));
    memcpy_h2d(d.temperature, s.temperature, (std::size_t)total  * sizeof(float));
    memcpy_h2d(d.obstacle,    s.obstacle,    (std::size_t)total  * sizeof(float));

    if (s.n_emitters > 0) {
        memcpy_h2d(d.emitter_masks,        s.emitter_masks,
                   (std::size_t)s.n_emitters * total * sizeof(float));
        memcpy_h2d(d.emitter_densities,    s.emitter_densities,
                   (std::size_t)s.n_emitters * sizeof(float));
        memcpy_h2d(d.emitter_temperatures, s.emitter_temperatures,
                   (std::size_t)s.n_emitters * sizeof(float));
    }

    d.viscosity = s.viscosity;
    d.buoyancy  = s.buoyancy;
    d.cooling   = s.cooling;
    d.vorticity = s.vorticity;
}

void download(const State& s, State& d) {
    assert(s.nx == d.nx && s.ny == d.ny && s.n_emitters == d.n_emitters);

    const int total  = s.nx * s.ny;
    const int v_size = (s.nx + 1) * s.ny + s.nx * (s.ny + 1);

    memcpy_d2h(d.density,     s.density,     (std::size_t)total  * sizeof(float));
    memcpy_d2h(d.velocity,    s.velocity,    (std::size_t)v_size * sizeof(float));
    memcpy_d2h(d.temperature, s.temperature, (std::size_t)total  * sizeof(float));
    memcpy_d2h(d.obstacle,    s.obstacle,    (std::size_t)total  * sizeof(float));

    if (s.n_emitters > 0) {
        memcpy_d2h(d.emitter_masks,        s.emitter_masks,
                   (std::size_t)s.n_emitters * total * sizeof(float));
        memcpy_d2h(d.emitter_densities,    s.emitter_densities,
                   (std::size_t)s.n_emitters * sizeof(float));
        memcpy_d2h(d.emitter_temperatures, s.emitter_temperatures,
                   (std::size_t)s.n_emitters * sizeof(float));
    }

    d.viscosity = s.viscosity;
    d.buoyancy  = s.buoyancy;
    d.cooling   = s.cooling;
    d.vorticity = s.vorticity;
}

void download_field(const State& s, State& d, Field field) {
    assert(s.nx == d.nx && s.ny == d.ny);

    const int total  = s.nx * s.ny;
    const int v_size = (s.nx + 1) * s.ny + s.nx * (s.ny + 1);

    switch (field) {
    case Field::Density:
        memcpy_d2h(d.density,  s.density,  (std::size_t)total  * sizeof(float));
        break;
    case Field::Velocity:
        memcpy_d2h(d.velocity, s.velocity, (std::size_t)v_size * sizeof(float));
        break;
    }
}

#endif

} // namespace slipstream

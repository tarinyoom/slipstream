#include "state.hpp"

#include <cstring>

namespace slipstream {

void upload(const CpuState& src, GpuState& dst) {
    const State& h = src.s;
    const State& d = dst.s;

    int total  = 1;
    for (int i = 0; i < h.n_dims; ++i) total *= h.dims[i];
    int Nx = h.dims[0], Ny = h.dims[1];
    int v_size = (Nx + 1) * Ny + Nx * (Ny + 1);

    cudaMemcpy(d.density,     h.density,     total  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d.v,           h.v,           v_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d.temperature, h.temperature, total  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(const_cast<bool*>(d.obstacle), h.obstacle,
               total * sizeof(bool), cudaMemcpyHostToDevice);

    if (h.n_emitters > 0 && h.emitter_masks) {
        cudaMemcpy(const_cast<bool* >(d.emitter_masks),
                   h.emitter_masks,
                   h.n_emitters * total * sizeof(bool),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<float*>(d.emitter_densities),
                   h.emitter_densities,
                   h.n_emitters * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(const_cast<float*>(d.emitter_temperatures),
                   h.emitter_temperatures,
                   h.n_emitters * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    dst.s.n_emitters = h.n_emitters;
    dst.s.viscosity  = h.viscosity;
    dst.s.buoyancy   = h.buoyancy;
    dst.s.cooling    = h.cooling;
    dst.s.vorticity  = h.vorticity;
}

void download(const GpuState& src, CpuState& dst) {
    const State& d = src.s;
    const State& h = dst.s;

    int total  = 1;
    for (int i = 0; i < h.n_dims; ++i) total *= h.dims[i];
    int Nx = h.dims[0], Ny = h.dims[1];
    int v_size = (Nx + 1) * Ny + Nx * (Ny + 1);

    cudaMemcpy(h.density,     d.density,     total  * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h.v,           d.v,           v_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h.temperature, d.temperature, total  * sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace slipstream

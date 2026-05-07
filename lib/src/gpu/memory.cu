#include "state.hpp"

#include <cassert>

namespace slipstream {

State alloc_gpu_state(int n_dims, const int* dims, int n_emitters) {
    assert(n_dims == 2);
    int Nx = dims[0], Ny = dims[1];
    int total  = Nx * Ny;
    int v_size = (Nx + 1) * Ny + Nx * (Ny + 1);

    State s{};
    s.n_dims     = n_dims;
    s.n_emitters = n_emitters;

    cudaMalloc(&s.dims, n_dims * sizeof(int));
    cudaMemcpy(s.dims, dims, n_dims * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&s.density,     total  * sizeof(float));
    cudaMalloc(&s.v,           v_size * sizeof(float));
    cudaMalloc(&s.temperature, total  * sizeof(float));

    bool* obs;
    cudaMalloc(&obs, total * sizeof(bool));
    cudaMemset(obs, 0, total * sizeof(bool));
    s.obstacle = obs;

    cudaMemset(s.density,     0, total  * sizeof(float));
    cudaMemset(s.v,           0, v_size * sizeof(float));
    cudaMemset(s.temperature, 0, total  * sizeof(float));

    if (n_emitters > 0) {
        bool*  em;
        float* ed;
        float* et;
        cudaMalloc(&em, n_emitters * total    * sizeof(bool));
        cudaMalloc(&ed, n_emitters            * sizeof(float));
        cudaMalloc(&et, n_emitters            * sizeof(float));
        cudaMemset(em, 0, n_emitters * total  * sizeof(bool));
        cudaMemset(ed, 0, n_emitters          * sizeof(float));
        cudaMemset(et, 0, n_emitters          * sizeof(float));
        s.emitter_masks        = em;
        s.emitter_densities    = ed;
        s.emitter_temperatures = et;
    }

    return s;
}

void free_gpu_state(State& s) {
    cudaFree(s.density);
    cudaFree(s.v);
    cudaFree(s.temperature);
    cudaFree(s.dims);
    cudaFree(const_cast<bool* >(s.obstacle));
    cudaFree(const_cast<bool* >(s.emitter_masks));
    cudaFree(const_cast<float*>(s.emitter_densities));
    cudaFree(const_cast<float*>(s.emitter_temperatures));
    s = State{};
}

} // namespace slipstream

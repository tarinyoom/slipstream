#include "state.hpp"

#include <cassert>
#include <cstring>

namespace slipstream {

State alloc_cpu_state(int n_dims, const int* dims, int n_emitters) {
    assert(n_dims == 2);
    int Nx = dims[0], Ny = dims[1];
    int total  = Nx * Ny;
    int v_size = (Nx + 1) * Ny + Nx * (Ny + 1);

    State s{};
    s.n_dims     = n_dims;
    s.n_emitters = n_emitters;

    s.dims = new int[n_dims];
    std::memcpy(s.dims, dims, n_dims * sizeof(int));

    s.density     = new float[total]();
    s.v           = new float[v_size]();
    s.temperature = new float[total]();
    s.obstacle    = new bool[total]();

    if (n_emitters > 0) {
        s.emitter_masks        = new bool [n_emitters * total]();
        s.emitter_densities    = new float[n_emitters]();
        s.emitter_temperatures = new float[n_emitters]();
    }

    return s;
}

void free_cpu_state(State& s) {
    delete[] s.density;
    delete[] s.v;
    delete[] s.temperature;
    delete[] s.dims;
    delete[] const_cast<bool* >(s.obstacle);
    delete[] const_cast<bool* >(s.emitter_masks);
    delete[] const_cast<float*>(s.emitter_densities);
    delete[] const_cast<float*>(s.emitter_temperatures);
    s = State{};
}

} // namespace slipstream

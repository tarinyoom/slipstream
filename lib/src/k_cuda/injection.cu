#include "injection.hpp"

#include <cuda_runtime.h>

namespace slipstream::k_cuda {

__global__ static void inject_emitters_kernel(int n_emitters, int total,
                                               const float* masks,
                                               const float* emitter_densities,
                                               const float* emitter_temperatures,
                                               float* density, float* temperature)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n_emitters * total) return;
    int e = idx / total;
    int c = idx % total;
    if (masks[e * total + c] != 0.0f) {
        density[c] = emitter_densities[e];
        if (temperature && emitter_temperatures)
            temperature[c] = emitter_temperatures[e];
    }
}

void compute_injection(int n_emitters, int total,
                       const float* masks,
                       const float* emitter_densities,
                       const float* emitter_temperatures,
                       float* density, float* temperature)
{
    if (n_emitters == 0) return;
    int threads = 256;
    int blocks  = (n_emitters * total + threads - 1) / threads;
    inject_emitters_kernel<<<blocks, threads>>>(n_emitters, total, masks,
                                                emitter_densities, emitter_temperatures,
                                                density, temperature);
    cudaDeviceSynchronize();
}

} // namespace slipstream::k_cuda

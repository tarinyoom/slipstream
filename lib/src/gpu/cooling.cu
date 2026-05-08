#include "cooling.hpp"

#include <cuda_runtime.h>

namespace slipstream::gpu {

__global__ static void apply_cooling_kernel(int total, float factor, float* temperature) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= total) return;
    temperature[c] *= factor;
}

void apply_cooling(int total, float cooling, float dt, float* temperature) {
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    apply_cooling_kernel<<<blocks, threads>>>(total, 1.0f - cooling * dt, temperature);
    cudaDeviceSynchronize();
}

} // namespace slipstream::gpu

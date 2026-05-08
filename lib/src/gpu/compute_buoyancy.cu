#include "compute_buoyancy.hpp"

#include <cuda_runtime.h>

namespace slipstream::gpu {

__global__ static void apply_buoyancy_kernel(int nx, int ny, float buoyancy, float dt,
                                              const float* temperature, float* vx)
{
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y) + 1;
    if (i >= nx || j >= ny) return;
    float t_avg = 0.5f * (temperature[(i - 1) * ny + j]
                        + temperature[ i      * ny + j]);
    vx[i * ny + j] += buoyancy * t_avg * dt;
}

void compute_buoyancy(int nx, int ny, float buoyancy, float dt,
                      const float* temperature, float* vx)
{
    if (nx < 2) return;
    dim3 block(16, 16);
    dim3 grid((ny + 15) / 16, (nx - 1 + 15) / 16);
    apply_buoyancy_kernel<<<grid, block>>>(nx, ny, buoyancy, dt, temperature, vx);
    cudaDeviceSynchronize();
}

} // namespace slipstream::gpu

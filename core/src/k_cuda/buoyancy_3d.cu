#include "buoyancy_3d.hpp"

#include <cuda_runtime.h>

namespace slipstream::k_cuda_3d {

__global__ static void apply_buoyancy_kernel(int nx, int ny, int nz,
                                              float buoyancy, float dt,
                                              const float* temperature, float* vx)
{
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int i = (int)(blockIdx.z * blockDim.z + threadIdx.z) + 1;
    if (i >= nx || j >= ny || k >= nz) return;
    float t_avg = 0.5f * (temperature[((i - 1) * ny + j) * nz + k]
                        + temperature[( i      * ny + j) * nz + k]);
    vx[(i * ny + j) * nz + k] += buoyancy * t_avg * dt;
}

void compute_buoyancy(int nx, int ny, int nz, float buoyancy, float dt,
                      const float* temperature, float* vx)
{
    if (nx < 2) return;
    dim3 block(8, 8, 8);
    dim3 grid((nz + 7) / 8, (ny + 7) / 8, (nx - 1 + 7) / 8);
    apply_buoyancy_kernel<<<grid, block>>>(nx, ny, nz, buoyancy, dt, temperature, vx);
    cudaDeviceSynchronize();
}

} // namespace slipstream::k_cuda_3d

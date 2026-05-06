#include "advect.hpp"

#include <span>
#include <vector>

namespace slipstream::gpu {

struct DeviceAdvectArgs {
    const float* density;
    const float* vx;
    const float* vy;
    float*       scratch;
    int   Nx, Ny;
    float dt;
};

__device__ float bilinear_sample_dev(const float* v, int fs0, int fs1, float px, float py) {
    int i0 = (int)floorf(px);
    int j0 = (int)floorf(py);
    int i1 = i0 + 1;
    int j1 = j0 + 1;
    float tx = px - (float)i0;
    float ty = py - (float)j0;

    i0 = min(max(i0, 0), fs0 - 1);
    i1 = min(max(i1, 0), fs0 - 1);
    j0 = min(max(j0, 0), fs1 - 1);
    j1 = min(max(j1, 0), fs1 - 1);

    float v00 = v[i0 * fs1 + j0];
    float v10 = v[i1 * fs1 + j0];
    float v01 = v[i0 * fs1 + j1];
    float v11 = v[i1 * fs1 + j1];

    return (1.0f - tx) * (1.0f - ty) * v00
         +          tx * (1.0f - ty) * v10
         + (1.0f - tx) *          ty * v01
         +          tx *          ty * v11;
}

__global__ void advect_scalar_kernel(DeviceAdvectArgs a) {
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= a.Nx || j >= a.Ny) return;

    int Ny = a.Ny;
    float vx_c = 0.5f * (a.vx[ i      * Ny + j] + a.vx[(i + 1) * Ny + j]);
    float vy_c = 0.5f * (a.vy[ i * (Ny + 1) + j] + a.vy[ i * (Ny + 1) + (j + 1)]);

    float bx = fminf(fmaxf((float)i - a.dt * vx_c, 0.0f), (float)(a.Nx - 1));
    float by = fminf(fmaxf((float)j - a.dt * vy_c, 0.0f), (float)(a.Ny - 1));

    a.scratch[i * Ny + j] = bilinear_sample_dev(a.density, a.Nx, a.Ny, bx, by);
}

void advect_scalar(std::span<const int>                 shape,
                   std::span<const float>               field,
                   std::span<float>                     scratch,
                   const std::vector<std::span<float>>& velocity,
                   float                                dt)
{
    const int Nx = shape[0];
    const int Ny = shape[1];

    const size_t density_bytes = (size_t)Nx * Ny * sizeof(float);
    const size_t vx_bytes      = (size_t)(Nx + 1) * Ny * sizeof(float);
    const size_t vy_bytes      = (size_t)Nx * (Ny + 1) * sizeof(float);

    float *d_density, *d_vx, *d_vy, *d_scratch;
    cudaMalloc(&d_density, density_bytes);
    cudaMalloc(&d_vx,      vx_bytes);
    cudaMalloc(&d_vy,      vy_bytes);
    cudaMalloc(&d_scratch, density_bytes);

    cudaMemcpy(d_density, field.data(),        density_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx,      velocity[0].data(),  vx_bytes,      cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy,      velocity[1].data(),  vy_bytes,      cudaMemcpyHostToDevice);

    DeviceAdvectArgs args{ d_density, d_vx, d_vy, d_scratch, Nx, Ny, dt };

    dim3 block(16, 16);
    dim3 grid((Ny + 15) / 16, (Nx + 15) / 16);
    advect_scalar_kernel<<<grid, block>>>(args);
    cudaDeviceSynchronize();

    cudaMemcpy(scratch.data(), d_scratch, density_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_density);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_scratch);
}

} // namespace slipstream::gpu

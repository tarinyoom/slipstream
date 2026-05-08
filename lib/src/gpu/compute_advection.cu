#include "compute_advection.hpp"

#include <cuda_runtime.h>
#include <cstddef>

namespace slipstream::gpu {

__device__ static float bilinear_sample_dev(const float* v, int fs0, int fs1,
                                             float px, float py)
{
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

__device__ static float safe_get_dev(const float* v, int fs0, int fs1, int a, int b) {
    if (a < 0 || a >= fs0 || b < 0 || b >= fs1) return 0.0f;
    return v[a * fs1 + b];
}

__global__ static void advect_scalar_kernel(int nx, int ny,
                                             const float* vx, const float* vy,
                                             const float* field_in, float* field_out,
                                             float dt)
{
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= nx || j >= ny) return;

    float vx_c = 0.5f * (vx[ i      * ny + j] + vx[(i + 1) * ny + j]);
    float vy_c = 0.5f * (vy[ i * (ny + 1) + j] + vy[ i * (ny + 1) + (j + 1)]);

    float bx = fminf(fmaxf((float)i - dt * vx_c, 0.0f), (float)(nx - 1));
    float by = fminf(fmaxf((float)j - dt * vy_c, 0.0f), (float)(ny - 1));

    field_out[i * ny + j] = bilinear_sample_dev(field_in, nx, ny, bx, by);
}

void compute_scalar_advection(int nx, int ny,
                              const float* vx, const float* vy,
                              const float* field_in, float* field_out, float dt)
{
    dim3 block(16, 16);
    dim3 grid((ny + 15) / 16, (nx + 15) / 16);
    advect_scalar_kernel<<<grid, block>>>(nx, ny, vx, vy, field_in, field_out, dt);
    cudaDeviceSynchronize();
}

__global__ static void advect_vx_kernel(int nx, int ny,
                                         const float* vx, const float* vy,
                                         float* scratch, float dt)
{
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= nx + 1 || j >= ny) return;

    float vd_here = vx[i * ny + j];
    float vo_here = 0.25f * (safe_get_dev(vy, nx, ny + 1, i - 1, j    )
                            + safe_get_dev(vy, nx, ny + 1, i,     j    )
                            + safe_get_dev(vy, nx, ny + 1, i - 1, j + 1)
                            + safe_get_dev(vy, nx, ny + 1, i,     j + 1));

    float bx = fminf(fmaxf((float)i - dt * vd_here, 0.0f), (float)(nx));
    float by = fminf(fmaxf((float)j - dt * vo_here, 0.0f), (float)(ny - 1));

    scratch[i * ny + j] = bilinear_sample_dev(vx, nx + 1, ny, bx, by);
}

__global__ static void advect_vy_kernel(int nx, int ny,
                                         const float* vx, const float* vy,
                                         float* scratch, float dt)
{
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (i >= nx || j >= ny + 1) return;

    float vd_here = vy[i * (ny + 1) + j];
    float vo_here = 0.25f * (safe_get_dev(vx, nx + 1, ny, i,     j - 1)
                            + safe_get_dev(vx, nx + 1, ny, i + 1, j - 1)
                            + safe_get_dev(vx, nx + 1, ny, i,     j    )
                            + safe_get_dev(vx, nx + 1, ny, i + 1, j    ));

    float bx = fminf(fmaxf((float)i - dt * vo_here, 0.0f), (float)(nx - 1));
    float by = fminf(fmaxf((float)j - dt * vd_here, 0.0f), (float)(ny));

    scratch[i * (ny + 1) + j] = bilinear_sample_dev(vy, nx, ny + 1, bx, by);
}

void compute_velocity_advection(int nx, int ny, float* vx, float* vy, float* scratch, float dt) {
    dim3 block(16, 16);

    dim3 grid_vx((ny + 15) / 16, (nx + 1 + 15) / 16);
    advect_vx_kernel<<<grid_vx, block>>>(nx, ny, vx, vy, scratch, dt);
    cudaDeviceSynchronize();
    cudaMemcpy(vx, scratch, (std::size_t)(nx + 1) * ny * sizeof(float),
               cudaMemcpyDeviceToDevice);

    dim3 grid_vy((ny + 1 + 15) / 16, (nx + 15) / 16);
    advect_vy_kernel<<<grid_vy, block>>>(nx, ny, vx, vy, scratch, dt);
    cudaDeviceSynchronize();
    cudaMemcpy(vy, scratch, (std::size_t)nx * (ny + 1) * sizeof(float),
               cudaMemcpyDeviceToDevice);
}

} // namespace slipstream::gpu

#include "advection_3d.hpp"

#include <cuda_runtime.h>
#include <cstddef>

namespace slipstream::k_cuda_3d {

__device__ static float trilinear_sample_dev(const float* v, int fs0, int fs1, int fs2,
                                              float px, float py, float pz)
{
    int i0 = (int)floorf(px);
    int j0 = (int)floorf(py);
    int k0 = (int)floorf(pz);
    int i1 = i0 + 1;
    int j1 = j0 + 1;
    int k1 = k0 + 1;
    float tx = px - (float)i0;
    float ty = py - (float)j0;
    float tz = pz - (float)k0;

    i0 = min(max(i0, 0), fs0 - 1);
    i1 = min(max(i1, 0), fs0 - 1);
    j0 = min(max(j0, 0), fs1 - 1);
    j1 = min(max(j1, 0), fs1 - 1);
    k0 = min(max(k0, 0), fs2 - 1);
    k1 = min(max(k1, 0), fs2 - 1);

    auto at = [&](int a, int b, int c) {
        return v[(a * fs1 + b) * fs2 + c];
    };

    float v000 = at(i0, j0, k0);
    float v100 = at(i1, j0, k0);
    float v010 = at(i0, j1, k0);
    float v110 = at(i1, j1, k0);
    float v001 = at(i0, j0, k1);
    float v101 = at(i1, j0, k1);
    float v011 = at(i0, j1, k1);
    float v111 = at(i1, j1, k1);

    float w000 = (1.0f - tx) * (1.0f - ty) * (1.0f - tz);
    float w100 =          tx * (1.0f - ty) * (1.0f - tz);
    float w010 = (1.0f - tx) *          ty * (1.0f - tz);
    float w110 =          tx *          ty * (1.0f - tz);
    float w001 = (1.0f - tx) * (1.0f - ty) *          tz;
    float w101 =          tx * (1.0f - ty) *          tz;
    float w011 = (1.0f - tx) *          ty *          tz;
    float w111 =          tx *          ty *          tz;

    return w000 * v000 + w100 * v100 + w010 * v010 + w110 * v110
         + w001 * v001 + w101 * v101 + w011 * v011 + w111 * v111;
}

__device__ static float safe_get_dev(const float* v, int fs0, int fs1, int fs2,
                                      int a, int b, int c)
{
    if (a < 0 || a >= fs0 || b < 0 || b >= fs1 || c < 0 || c >= fs2) return 0.0f;
    return v[(a * fs1 + b) * fs2 + c];
}

__global__ static void advect_scalar_kernel(int nx, int ny, int nz,
                                             const float* vx, const float* vy, const float* vz,
                                             const float* field_in, float* field_out,
                                             float dt)
{
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int i = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx || j >= ny || k >= nz) return;

    float vx_c = 0.5f * (vx[( i      * ny + j) * nz + k]
                       + vx[((i + 1) * ny + j) * nz + k]);
    float vy_c = 0.5f * (vy[(i * (ny + 1) +  j     ) * nz + k]
                       + vy[(i * (ny + 1) + (j + 1)) * nz + k]);
    float vz_c = 0.5f * (vz[(i * ny + j) * (nz + 1) +  k     ]
                       + vz[(i * ny + j) * (nz + 1) + (k + 1)]);

    float bx = fminf(fmaxf((float)i - dt * vx_c, 0.0f), (float)(nx - 1));
    float by = fminf(fmaxf((float)j - dt * vy_c, 0.0f), (float)(ny - 1));
    float bz = fminf(fmaxf((float)k - dt * vz_c, 0.0f), (float)(nz - 1));

    field_out[(i * ny + j) * nz + k] = trilinear_sample_dev(field_in, nx, ny, nz, bx, by, bz);
}

void compute_scalar_advection(int nx, int ny, int nz,
                              const float* vx, const float* vy, const float* vz,
                              const float* field_in, float* field_out, float dt)
{
    dim3 block(8, 8, 8);
    dim3 grid((nz + 7) / 8, (ny + 7) / 8, (nx + 7) / 8);
    advect_scalar_kernel<<<grid, block>>>(nx, ny, nz, vx, vy, vz, field_in, field_out, dt);
    cudaDeviceSynchronize();
}

__global__ static void advect_vx_kernel(int nx, int ny, int nz,
                                         const float* vx, const float* vy, const float* vz,
                                         float* out, float dt)
{
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int i = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx + 1 || j >= ny || k >= nz) return;

    float vd   = vx[(i * ny + j) * nz + k];
    float vo_y = 0.25f * (safe_get_dev(vy, nx, ny + 1, nz, i - 1, j,     k)
                        + safe_get_dev(vy, nx, ny + 1, nz, i,     j,     k)
                        + safe_get_dev(vy, nx, ny + 1, nz, i - 1, j + 1, k)
                        + safe_get_dev(vy, nx, ny + 1, nz, i,     j + 1, k));
    float vo_z = 0.25f * (safe_get_dev(vz, nx, ny, nz + 1, i - 1, j, k    )
                        + safe_get_dev(vz, nx, ny, nz + 1, i,     j, k    )
                        + safe_get_dev(vz, nx, ny, nz + 1, i - 1, j, k + 1)
                        + safe_get_dev(vz, nx, ny, nz + 1, i,     j, k + 1));

    float bx = fminf(fmaxf((float)i - dt * vd,   0.0f), (float)(nx));
    float by = fminf(fmaxf((float)j - dt * vo_y, 0.0f), (float)(ny - 1));
    float bz = fminf(fmaxf((float)k - dt * vo_z, 0.0f), (float)(nz - 1));

    out[(i * ny + j) * nz + k] = trilinear_sample_dev(vx, nx + 1, ny, nz, bx, by, bz);
}

__global__ static void advect_vy_kernel(int nx, int ny, int nz,
                                         const float* vx, const float* vy, const float* vz,
                                         float* out, float dt)
{
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int i = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx || j >= ny + 1 || k >= nz) return;

    float vd   = vy[(i * (ny + 1) + j) * nz + k];
    float vo_x = 0.25f * (safe_get_dev(vx, nx + 1, ny, nz, i,     j - 1, k)
                        + safe_get_dev(vx, nx + 1, ny, nz, i + 1, j - 1, k)
                        + safe_get_dev(vx, nx + 1, ny, nz, i,     j,     k)
                        + safe_get_dev(vx, nx + 1, ny, nz, i + 1, j,     k));
    float vo_z = 0.25f * (safe_get_dev(vz, nx, ny, nz + 1, i, j - 1, k    )
                        + safe_get_dev(vz, nx, ny, nz + 1, i, j,     k    )
                        + safe_get_dev(vz, nx, ny, nz + 1, i, j - 1, k + 1)
                        + safe_get_dev(vz, nx, ny, nz + 1, i, j,     k + 1));

    float bx = fminf(fmaxf((float)i - dt * vo_x, 0.0f), (float)(nx - 1));
    float by = fminf(fmaxf((float)j - dt * vd,   0.0f), (float)(ny));
    float bz = fminf(fmaxf((float)k - dt * vo_z, 0.0f), (float)(nz - 1));

    out[(i * (ny + 1) + j) * nz + k] = trilinear_sample_dev(vy, nx, ny + 1, nz, bx, by, bz);
}

__global__ static void advect_vz_kernel(int nx, int ny, int nz,
                                         const float* vx, const float* vy, const float* vz,
                                         float* out, float dt)
{
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int i = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx || j >= ny || k >= nz + 1) return;

    float vd   = vz[(i * ny + j) * (nz + 1) + k];
    float vo_x = 0.25f * (safe_get_dev(vx, nx + 1, ny, nz, i,     j, k - 1)
                        + safe_get_dev(vx, nx + 1, ny, nz, i + 1, j, k - 1)
                        + safe_get_dev(vx, nx + 1, ny, nz, i,     j, k    )
                        + safe_get_dev(vx, nx + 1, ny, nz, i + 1, j, k    ));
    float vo_y = 0.25f * (safe_get_dev(vy, nx, ny + 1, nz, i, j,     k - 1)
                        + safe_get_dev(vy, nx, ny + 1, nz, i, j + 1, k - 1)
                        + safe_get_dev(vy, nx, ny + 1, nz, i, j,     k    )
                        + safe_get_dev(vy, nx, ny + 1, nz, i, j + 1, k    ));

    float bx = fminf(fmaxf((float)i - dt * vo_x, 0.0f), (float)(nx - 1));
    float by = fminf(fmaxf((float)j - dt * vo_y, 0.0f), (float)(ny - 1));
    float bz = fminf(fmaxf((float)k - dt * vd,   0.0f), (float)(nz));

    out[(i * ny + j) * (nz + 1) + k] = trilinear_sample_dev(vz, nx, ny, nz + 1, bx, by, bz);
}

void compute_velocity_advection(int nx, int ny, int nz,
                                float* vx, float* vy, float* vz,
                                float* scratch, float dt)
{
    dim3 block(8, 8, 8);

    dim3 grid_vx((nz + 7) / 8, (ny + 7) / 8, (nx + 1 + 7) / 8);
    advect_vx_kernel<<<grid_vx, block>>>(nx, ny, nz, vx, vy, vz, scratch, dt);
    cudaDeviceSynchronize();
    cudaMemcpy(vx, scratch, (std::size_t)(nx + 1) * ny * nz * sizeof(float),
               cudaMemcpyDeviceToDevice);

    dim3 grid_vy((nz + 7) / 8, (ny + 1 + 7) / 8, (nx + 7) / 8);
    advect_vy_kernel<<<grid_vy, block>>>(nx, ny, nz, vx, vy, vz, scratch, dt);
    cudaDeviceSynchronize();
    cudaMemcpy(vy, scratch, (std::size_t)nx * (ny + 1) * nz * sizeof(float),
               cudaMemcpyDeviceToDevice);

    dim3 grid_vz((nz + 1 + 7) / 8, (ny + 7) / 8, (nx + 7) / 8);
    advect_vz_kernel<<<grid_vz, block>>>(nx, ny, nz, vx, vy, vz, scratch, dt);
    cudaDeviceSynchronize();
    cudaMemcpy(vz, scratch, (std::size_t)nx * ny * (nz + 1) * sizeof(float),
               cudaMemcpyDeviceToDevice);
}

} // namespace slipstream::k_cuda_3d

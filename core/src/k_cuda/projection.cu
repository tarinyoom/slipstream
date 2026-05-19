#include "projection.hpp"

#include <cuda_runtime.h>
#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace slipstream::k_cuda {

__device__ static void flat_to_ijk(int ny, int nz, int flat,
                                    int& i, int& j, int& k)
{
    k = flat % nz;
    int rest = flat / nz;
    j = rest % ny;
    i = rest / ny;
}

__global__ static void zero_obstacle_faces_kernel(int nx, int ny, int nz,
                                                   const float* obstacle,
                                                   float* vx, float* vy, float* vz)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny * nz) return;
    if (obstacle[flat] == 0.0f) return;
    int ci, cj, ck;
    flat_to_ijk(ny, nz, flat, ci, cj, ck);
    vx[(ci     * ny + cj    ) * nz + ck    ] = 0.0f;
    vx[((ci+1) * ny + cj    ) * nz + ck    ] = 0.0f;
    vy[(ci * (ny + 1) + cj    ) * nz + ck    ] = 0.0f;
    vy[(ci * (ny + 1) + cj + 1) * nz + ck    ] = 0.0f;
    vz[(ci * ny + cj) * (nz + 1) + ck    ] = 0.0f;
    vz[(ci * ny + cj) * (nz + 1) + ck + 1] = 0.0f;
}

__global__ static void compute_divergence_kernel(int nx, int ny, int nz,
                                                  const float* obstacle,
                                                  const float* vx, const float* vy, const float* vz,
                                                  float* rhs)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny * nz) return;
    if (obstacle && obstacle[flat] != 0.0f) { rhs[flat] = 0.0f; return; }
    int ci, cj, ck;
    flat_to_ijk(ny, nz, flat, ci, cj, ck);
    rhs[flat] = vx[((ci + 1) * ny + cj    ) * nz + ck    ] - vx[(ci * ny + cj) * nz + ck]
              + vy[( ci      * (ny + 1) + cj + 1) * nz + ck    ] - vy[(ci * (ny + 1) + cj) * nz + ck]
              + vz[( ci      * ny + cj    ) * (nz + 1) + ck + 1] - vz[(ci * ny + cj) * (nz + 1) + ck];
}

__global__ static void red_black_gs_kernel(int nx, int ny, int nz,
                                            const float* obstacle,
                                            const float* rhs, float* pressure, int color)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny * nz) return;
    int ci, cj, ck;
    flat_to_ijk(ny, nz, flat, ci, cj, ck);
    if ((ci + cj + ck) % 2 != color) return;
    if (obstacle && obstacle[flat] != 0.0f) return;

    float sum_nbr = 0.0f;
    int   num_nbr = 0;
    const int di[6] = {-1, +1,  0,  0,  0,  0};
    const int dj[6] = { 0,  0, -1, +1,  0,  0};
    const int dk[6] = { 0,  0,  0,  0, -1, +1};
    for (int n = 0; n < 6; ++n) {
        int ni = ci + di[n];
        int nj = cj + dj[n];
        int nk = ck + dk[n];
        if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue;
        int nc = (ni * ny + nj) * nz + nk;
        if (obstacle && obstacle[nc] != 0.0f) continue;
        sum_nbr += pressure[nc];
        ++num_nbr;
    }
    if (num_nbr > 0)
        pressure[flat] = (sum_nbr - rhs[flat]) / (float)num_nbr;
}

__global__ static void compute_residual_kernel(int nx, int ny, int nz,
                                                const float* obstacle,
                                                const float* rhs, const float* pressure,
                                                float* residuals)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny * nz) return;
    if (obstacle && obstacle[flat] != 0.0f) { residuals[flat] = 0.0f; return; }
    int ci, cj, ck;
    flat_to_ijk(ny, nz, flat, ci, cj, ck);

    float sum_nbr = 0.0f;
    int   num_nbr = 0;
    const int di[6] = {-1, +1,  0,  0,  0,  0};
    const int dj[6] = { 0,  0, -1, +1,  0,  0};
    const int dk[6] = { 0,  0,  0,  0, -1, +1};
    for (int n = 0; n < 6; ++n) {
        int ni = ci + di[n];
        int nj = cj + dj[n];
        int nk = ck + dk[n];
        if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue;
        int nc = (ni * ny + nj) * nz + nk;
        if (obstacle && obstacle[nc] != 0.0f) continue;
        sum_nbr += pressure[nc];
        ++num_nbr;
    }
    residuals[flat] = fabsf((float)num_nbr * pressure[flat] - sum_nbr + rhs[flat]);
}

__global__ static void gradient_subtract_kernel(int nx, int ny, int nz,
                                                  const float* obstacle,
                                                  const float* pressure,
                                                  float* vx, float* vy, float* vz)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny * nz) return;
    if (obstacle && obstacle[flat] != 0.0f) return;
    int ci, cj, ck;
    flat_to_ijk(ny, nz, flat, ci, cj, ck);
    if (ci > 0) {
        int lc = ((ci - 1) * ny + cj) * nz + ck;
        if (!obstacle || obstacle[lc] == 0.0f)
            vx[(ci * ny + cj) * nz + ck] -= pressure[flat] - pressure[lc];
    }
    if (cj > 0) {
        int lc = (ci * ny + (cj - 1)) * nz + ck;
        if (!obstacle || obstacle[lc] == 0.0f)
            vy[(ci * (ny + 1) + cj) * nz + ck] -= pressure[flat] - pressure[lc];
    }
    if (ck > 0) {
        int lc = (ci * ny + cj) * nz + (ck - 1);
        if (!obstacle || obstacle[lc] == 0.0f)
            vz[(ci * ny + cj) * (nz + 1) + ck] -= pressure[flat] - pressure[lc];
    }
}

void compute_projection(int nx, int ny, int nz, const float* obstacle,
                        float* vx, float* vy, float* vz,
                        float* pressure, float* rhs_scratch, float* residuals,
                        int max_iterations, float tolerance)
{
    const int total   = nx * ny * nz;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    if (obstacle)
        zero_obstacle_faces_kernel<<<blocks, threads>>>(nx, ny, nz, obstacle, vx, vy, vz);

    compute_divergence_kernel<<<blocks, threads>>>(nx, ny, nz, obstacle, vx, vy, vz, rhs_scratch);
    cudaDeviceSynchronize();

    for (int iter = 0; iter < max_iterations; ++iter) {
        red_black_gs_kernel<<<blocks, threads>>>(nx, ny, nz, obstacle, rhs_scratch, pressure, 0);
        red_black_gs_kernel<<<blocks, threads>>>(nx, ny, nz, obstacle, rhs_scratch, pressure, 1);

        if ((iter + 1) % 10 == 0) {
            compute_residual_kernel<<<blocks, threads>>>(nx, ny, nz, obstacle,
                                                         rhs_scratch, pressure, residuals);
            float residual = thrust::reduce(thrust::device,
                                            residuals, residuals + total,
                                            0.0f, cuda::maximum<float>{});
            if (residual < tolerance) break;
        }
    }
    cudaDeviceSynchronize();

    gradient_subtract_kernel<<<blocks, threads>>>(nx, ny, nz, obstacle, pressure, vx, vy, vz);
    cudaDeviceSynchronize();
}

} // namespace slipstream::k_cuda

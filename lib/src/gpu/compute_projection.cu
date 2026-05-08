#include "compute_projection.hpp"

#include <cuda_runtime.h>
#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace slipstream::gpu {

__global__ static void zero_obstacle_faces_kernel(int nx, int ny, const float* obstacle,
                                                   float* vx, float* vy)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny) return;
    if (obstacle[flat] == 0.0f) return;
    int ci = flat / ny;
    int cj = flat % ny;
    vx[ ci      * ny + cj] = 0.0f;
    vx[(ci + 1) * ny + cj] = 0.0f;
    vy[ci * (ny + 1) + cj    ] = 0.0f;
    vy[ci * (ny + 1) + cj + 1] = 0.0f;
}

__global__ static void compute_divergence_kernel(int nx, int ny, const float* obstacle,
                                                  const float* vx, const float* vy,
                                                  float* rhs)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny) return;
    if (obstacle && obstacle[flat] != 0.0f) { rhs[flat] = 0.0f; return; }
    int ci = flat / ny;
    int cj = flat % ny;
    rhs[flat] = vx[(ci + 1) * ny + cj    ] - vx[ci * ny + cj]
              + vy[ ci      * (ny + 1) + cj + 1] - vy[ci * (ny + 1) + cj];
}

__global__ static void red_black_gs_kernel(int nx, int ny, const float* obstacle,
                                            const float* rhs, float* pressure, int color)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny) return;
    int ci = flat / ny;
    int cj = flat % ny;
    if ((ci + cj) % 2 != color) return;
    if (obstacle && obstacle[flat] != 0.0f) return;

    float sum_nbr = 0.0f;
    int   num_nbr = 0;
    const int di[4] = {-1, +1,  0,  0};
    const int dj[4] = { 0,  0, -1, +1};
    for (int k = 0; k < 4; ++k) {
        int ni = ci + di[k];
        int nj = cj + dj[k];
        if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
        int nc = ni * ny + nj;
        if (obstacle && obstacle[nc] != 0.0f) continue;
        sum_nbr += pressure[nc];
        ++num_nbr;
    }
    if (num_nbr > 0)
        pressure[flat] = (sum_nbr - rhs[flat]) / (float)num_nbr;
}

__global__ static void compute_residual_kernel(int nx, int ny, const float* obstacle,
                                                const float* rhs, const float* pressure,
                                                float* residuals)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny) return;
    if (obstacle && obstacle[flat] != 0.0f) { residuals[flat] = 0.0f; return; }
    int ci = flat / ny;
    int cj = flat % ny;

    float sum_nbr = 0.0f;
    int   num_nbr = 0;
    const int di[4] = {-1, +1,  0,  0};
    const int dj[4] = { 0,  0, -1, +1};
    for (int k = 0; k < 4; ++k) {
        int ni = ci + di[k];
        int nj = cj + dj[k];
        if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
        int nc = ni * ny + nj;
        if (obstacle && obstacle[nc] != 0.0f) continue;
        sum_nbr += pressure[nc];
        ++num_nbr;
    }
    residuals[flat] = fabsf((float)num_nbr * pressure[flat] - sum_nbr + rhs[flat]);
}

__global__ static void gradient_subtract_kernel(int nx, int ny, const float* obstacle,
                                                  const float* pressure,
                                                  float* vx, float* vy)
{
    int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= nx * ny) return;
    if (obstacle && obstacle[flat] != 0.0f) return;
    int ci = flat / ny;
    int cj = flat % ny;
    if (ci > 0) {
        int lc = (ci - 1) * ny + cj;
        if (!obstacle || obstacle[lc] == 0.0f)
            vx[ci * ny + cj] -= pressure[flat] - pressure[lc];
    }
    if (cj > 0) {
        int lc = ci * ny + (cj - 1);
        if (!obstacle || obstacle[lc] == 0.0f)
            vy[ci * (ny + 1) + cj] -= pressure[flat] - pressure[lc];
    }
}

void compute_projection(int nx, int ny, const float* obstacle,
                        float* vx, float* vy,
                        float* pressure, float* rhs_scratch,
                        int max_iterations, float tolerance)
{
    const int total   = nx * ny;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    if (obstacle)
        zero_obstacle_faces_kernel<<<blocks, threads>>>(nx, ny, obstacle, vx, vy);

    cudaMemset(pressure, 0, (std::size_t)total * sizeof(float));
    compute_divergence_kernel<<<blocks, threads>>>(nx, ny, obstacle, vx, vy, rhs_scratch);
    cudaDeviceSynchronize();

    float* d_residuals;
    cudaMalloc(&d_residuals, (std::size_t)total * sizeof(float));

    for (int iter = 0; iter < max_iterations; ++iter) {
        red_black_gs_kernel<<<blocks, threads>>>(nx, ny, obstacle, rhs_scratch, pressure, 0);
        red_black_gs_kernel<<<blocks, threads>>>(nx, ny, obstacle, rhs_scratch, pressure, 1);

        if ((iter + 1) % 10 == 0) {
            compute_residual_kernel<<<blocks, threads>>>(nx, ny, obstacle,
                                                         rhs_scratch, pressure, d_residuals);
            float residual = thrust::reduce(thrust::device,
                                            d_residuals, d_residuals + total,
                                            0.0f, cuda::maximum<float>{});
            if (residual < tolerance) break;
        }
    }
    cudaDeviceSynchronize();
    cudaFree(d_residuals);

    gradient_subtract_kernel<<<blocks, threads>>>(nx, ny, obstacle, pressure, vx, vy);
    cudaDeviceSynchronize();
}

} // namespace slipstream::gpu

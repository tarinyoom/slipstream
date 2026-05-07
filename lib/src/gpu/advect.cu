#include "advect.hpp"

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

__global__ static void advect_scalar_kernel(PersistentState s, float* scratch, float dt) {
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int Nx = s.nx;
    int Ny = s.ny;
    if (i >= Nx || j >= Ny) return;

    const float* vx = s.velocity;
    const float* vy = s.velocity + (Nx + 1) * Ny;

    float vx_c = 0.5f * (vx[ i      * Ny + j] + vx[(i + 1) * Ny + j]);
    float vy_c = 0.5f * (vy[ i * (Ny + 1) + j] + vy[ i * (Ny + 1) + (j + 1)]);

    float bx = fminf(fmaxf((float)i - dt * vx_c, 0.0f), (float)(Nx - 1));
    float by = fminf(fmaxf((float)j - dt * vy_c, 0.0f), (float)(Ny - 1));

    scratch[i * Ny + j] = bilinear_sample_dev(s.density, Nx, Ny, bx, by);
}

void advect_scalar(PersistentState s, float* d_scratch, float dt) {
    int Nx = s.nx;
    int Ny = s.ny;

    dim3 block(16, 16);
    dim3 grid((Ny + 15) / 16, (Nx + 15) / 16);
    advect_scalar_kernel<<<grid, block>>>(s, d_scratch, dt);
    cudaDeviceSynchronize();
}

} // namespace slipstream::gpu

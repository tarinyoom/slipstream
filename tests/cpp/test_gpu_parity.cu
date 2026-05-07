#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

#include "arena.hpp"
#include "cpu/advect.hpp"
#include "gpu/advect.hpp"

using namespace slipstream;

TEST(GpuParity, AdvectScalar) {
    constexpr int   Nx = 32;
    constexpr int   Ny = 32;
    constexpr float dt = 0.1f;
    int dims[] = {Nx, Ny};

    CalculationArena cpu_arena(Backend::CPU, 2, dims, 0, false);
    PersistentState& cs = cpu_arena.state;
    {
        float* vx = cs.velocity;
        float* vy = cs.velocity + (Nx + 1) * Ny;

        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j) {
                float dx = (float)i - 15.5f;
                float dy = (float)j - 15.5f;
                cs.density[i * Ny + j] = expf(-(dx * dx + dy * dy) / 32.0f);
            }

        for (int i = 0; i <= Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                vx[i * Ny + j] = 0.5f * sinf((float)j * 3.14159f / (float)Ny);

        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j <= Ny; ++j)
                vy[i * (Ny + 1) + j] = 0.5f * sinf((float)i * 3.14159f / (float)Nx);
    }

    std::vector<float> cpu_scratch(Nx * Ny, 0.0f);
    cpu::advect_scalar(cs, cs.density, cpu_scratch.data(), dt);

    // GPU path (deferred: copy() not yet implemented)
    // CalculationArena gpu_arena(Backend::GPU, 2, dims, 0, false);
    // copy(cpu_arena, gpu_arena);
    // float* d_scratch;
    // cudaMalloc(&d_scratch, Nx * Ny * sizeof(float));
    // gpu::advect_scalar(gpu_arena.state, d_scratch, dt);
    // std::vector<float> gpu_result(Nx * Ny, 0.0f);
    // cudaMemcpy(gpu_result.data(), d_scratch, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaFree(d_scratch);
    // float max_diff = 0.0f;
    // for (int k = 0; k < Nx * Ny; ++k)
    //     max_diff = std::max(max_diff, std::abs(cpu_scratch[k] - gpu_result[k]));
    // EXPECT_LT(max_diff, 1e-4f) << "max delta: " << max_diff;

    GTEST_SKIP() << "GPU path deferred pending CalculationArena GPU implementation";
}

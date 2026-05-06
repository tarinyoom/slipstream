#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "cpu/advect.hpp"
#include "gpu/advect.hpp"

using namespace slipstream;

TEST(GpuParity, AdvectScalar) {
    constexpr int   Nx = 32;
    constexpr int   Ny = 32;
    constexpr float dt = 0.1f;

    std::vector<float> density((size_t)Nx * Ny);
    std::vector<float> vx((size_t)(Nx + 1) * Ny);
    std::vector<float> vy((size_t)Nx * (Ny + 1));

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            float dx = (float)i - 15.5f;
            float dy = (float)j - 15.5f;
            density[i * Ny + j] = expf(-(dx * dx + dy * dy) / 32.0f);
        }

    for (int i = 0; i <= Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            vx[i * Ny + j] = 0.5f * sinf((float)j * 3.14159f / (float)Ny);

    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j <= Ny; ++j)
            vy[i * (Ny + 1) + j] = 0.5f * sinf((float)i * 3.14159f / (float)Nx);

    std::vector<std::span<float>> velocity = {
        std::span<float>(vx),
        std::span<float>(vy)
    };

    int shape[] = {Nx, Ny};
    std::span<const int>   shape_span(shape, 2);
    std::span<const float> field_span(density);

    std::vector<float> cpu_scratch((size_t)Nx * Ny);
    std::vector<float> gpu_scratch((size_t)Nx * Ny);

    cpu::advect_scalar(shape_span, field_span, std::span<float>(cpu_scratch), velocity, dt);
    gpu::advect_scalar(shape_span, field_span, std::span<float>(gpu_scratch), velocity, dt);

    float max_diff = 0.0f;
    for (int k = 0; k < Nx * Ny; ++k)
        max_diff = std::max(max_diff, std::abs(cpu_scratch[k] - gpu_scratch[k]));

    EXPECT_LT(max_diff, 1e-4f) << "max delta: " << max_diff;
}

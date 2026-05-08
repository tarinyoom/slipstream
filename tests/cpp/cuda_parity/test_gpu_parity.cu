#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "arena.hpp"
#include "cpu/step.hpp"
#include "gpu/step.hpp"

using namespace slipstream;

namespace {

std::vector<float> d2h(const float* d_ptr, int n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d_ptr, (std::size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

float max_diff(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int k = 0; k < n; ++k) d = std::max(d, std::abs(a[k] - b[k]));
    return d;
}

} // namespace

TEST(CudaParity, SmokePlume) {
    constexpr int   Nx    = 32;
    constexpr int   Ny    = 32;
    constexpr int   STEPS = 20;
    constexpr float DT    = 0.04f;
    constexpr float TOL   = 1e-3f;

    int dims[] = {Nx, Ny};

    CalculationArena cpu_arena(Backend::CPU, 2, dims, 1, true);
    PersistentState& cs = cpu_arena.state;

    const int cx = Nx / 2;
    for (int i = 0; i < 4; ++i)
        for (int j = cx - 2; j < cx + 2; ++j)
            cs.emitter_masks[i * Ny + j] = 1.0f;
    cs.emitter_densities[0]    = 1.0f;
    cs.emitter_temperatures[0] = 200.0f;
    cs.buoyancy = 5.0f;
    cs.cooling  = 0.3f;

    CalculationArena gpu_arena(Backend::GPU, 2, dims, 1, true);
    copy(cpu_arena, gpu_arena);
    PersistentState& gs = gpu_arena.state;
    gs.buoyancy = cs.buoyancy;
    gs.cooling  = cs.cooling;

    for (int k = 0; k < STEPS; ++k) {
        cpu::step(cs, *cpu_arena.scratch, DT);
        gpu::step(gs, *gpu_arena.scratch, DT);
    }

    const int total  = Nx * Ny;
    const int v_size = (Nx + 1) * Ny + Nx * (Ny + 1);

    auto h_density     = d2h(gs.density,     total);
    auto h_temperature = d2h(gs.temperature, total);
    auto h_velocity    = d2h(gs.velocity,    v_size);

    float md_d = max_diff(cs.density,     h_density.data(),     total);
    float md_t = max_diff(cs.temperature, h_temperature.data(), total);
    float md_v = max_diff(cs.velocity,    h_velocity.data(),    v_size);

    EXPECT_LT(md_d, TOL) << "density max delta: "     << md_d;
    EXPECT_LT(md_t, TOL) << "temperature max delta: " << md_t;
    EXPECT_LT(md_v, TOL) << "velocity max delta: "    << md_v;
}

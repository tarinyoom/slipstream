#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include "arena.hpp"
#include "cpu/compute_advection.hpp"
#include "cpu/compute_injection.hpp"
#include "cpu/compute_buoyancy.hpp"
#include "cpu/compute_cooling.hpp"
#include "cpu/compute_projection.hpp"
#include "cpu/step.hpp"
#include "gpu/compute_advection.hpp"
#include "gpu/compute_injection.hpp"
#include "gpu/compute_buoyancy.hpp"
#include "gpu/compute_cooling.hpp"
#include "gpu/compute_projection.hpp"
#include "gpu/step.hpp"

using namespace slipstream;

static void fill_velocity(PersistentState& s) {
    const int Nx = s.nx, Ny = s.ny;
    float* vx = s.velocity;
    float* vy = s.velocity + (Nx + 1) * Ny;
    for (int i = 0; i <= Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            vx[i * Ny + j] = 0.5f * sinf((float)j * 3.14159f / (float)Ny);
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j <= Ny; ++j)
            vy[i * (Ny + 1) + j] = 0.5f * sinf((float)i * 3.14159f / (float)Nx);
}

static void fill_gaussian(float* field, int nx, int ny) {
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            float dx = (float)i - (float)(nx / 2) + 0.5f;
            float dy = (float)j - (float)(ny / 2) + 0.5f;
            field[i * ny + j] = expf(-(dx * dx + dy * dy) / 32.0f);
        }
}

static float max_diff(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int k = 0; k < n; ++k)
        d = std::max(d, std::abs(a[k] - b[k]));
    return d;
}

// Helper: copy device buffer to host vector
static std::vector<float> d2h(const float* d_ptr, int n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d_ptr, (std::size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

TEST(GpuParity, AdvectScalar) {
    constexpr int Nx = 32, Ny = 32;
    constexpr float dt = 0.1f;
    int dims[] = {Nx, Ny};

    CalculationArena cpu_arena(Backend::CPU, 2, dims, 0, true);
    PersistentState& cs = cpu_arena.state;
    fill_gaussian(cs.density, Nx, Ny);
    fill_velocity(cs);

    float* vx = cs.velocity;
    float* vy = cs.velocity + (Nx + 1) * Ny;

    std::vector<float> cpu_result(Nx * Ny);
    cpu::compute_scalar_advection(Nx, Ny, vx, vy, cs.density, cpu_result.data(), dt);

    CalculationArena gpu_arena(Backend::GPU, 2, dims, 0, true);
    copy(cpu_arena, gpu_arena);

    float* gvx = gpu_arena.state.velocity;
    float* gvy = gpu_arena.state.velocity + (Nx + 1) * Ny;
    gpu::compute_scalar_advection(Nx, Ny, gvx, gvy, gpu_arena.state.density,
                       gpu_arena.scratch->tmp, dt);

    auto gpu_result = d2h(gpu_arena.scratch->tmp, Nx * Ny);

    float md = max_diff(cpu_result.data(), gpu_result.data(), Nx * Ny);
    EXPECT_LT(md, 1e-4f) << "advect_scalar max delta: " << md;
}

TEST(GpuParity, AdvectVelocity) {
    constexpr int Nx = 32, Ny = 32;
    constexpr float dt = 0.1f;
    int dims[] = {Nx, Ny};

    CalculationArena cpu_arena(Backend::CPU, 2, dims, 0, true);
    fill_velocity(cpu_arena.state);

    // Save original velocity before CPU advection
    const int vx_size = (Nx + 1) * Ny;
    const int vy_size = Nx * (Ny + 1);
    std::vector<float> vx0(cpu_arena.state.velocity, cpu_arena.state.velocity + vx_size);
    std::vector<float> vy0(cpu_arena.state.velocity + vx_size,
                           cpu_arena.state.velocity + vx_size + vy_size);

    float* cvx = cpu_arena.state.velocity;
    float* cvy = cpu_arena.state.velocity + vx_size;
    std::vector<float> scratch(std::max(vx_size, vy_size));
    cpu::compute_velocity_advection(Nx, Ny, cvx, cvy, scratch.data(), dt);

    // GPU: restore original velocity, then advect
    CalculationArena gpu_arena(Backend::GPU, 2, dims, 0, true);
    // Upload original velocity
    cudaMemcpy(gpu_arena.state.velocity, vx0.data(),
               (std::size_t)vx_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_arena.state.velocity + vx_size, vy0.data(),
               (std::size_t)vy_size * sizeof(float), cudaMemcpyHostToDevice);

    float* gvx = gpu_arena.state.velocity;
    float* gvy = gpu_arena.state.velocity + vx_size;
    gpu::compute_velocity_advection(Nx, Ny, gvx, gvy, gpu_arena.scratch->tmp, dt);

    auto gpu_vx = d2h(gvx, vx_size);
    auto gpu_vy = d2h(gvy, vy_size);

    float md_vx = max_diff(cvx, gpu_vx.data(), vx_size);
    float md_vy = max_diff(cvy, gpu_vy.data(), vy_size);
    EXPECT_LT(md_vx, 1e-4f) << "advect_velocity vx max delta: " << md_vx;
    EXPECT_LT(md_vy, 1e-4f) << "advect_velocity vy max delta: " << md_vy;
}

TEST(GpuParity, InjectEmitters) {
    constexpr int Nx = 16, Ny = 16;
    int dims[] = {Nx, Ny};

    CalculationArena cpu_arena(Backend::CPU, 2, dims, 1, false);
    PersistentState& cs = cpu_arena.state;
    cs.emitter_masks[4 * Ny + 4] = 1.0f;
    cs.emitter_masks[8 * Ny + 8] = 1.0f;
    cs.emitter_densities[0]    = 2.5f;
    cs.emitter_temperatures[0] = 100.0f;

    cpu::compute_injection(1, Nx * Ny, cs.emitter_masks,
                         cs.emitter_densities, cs.emitter_temperatures,
                         cs.density, cs.temperature);

    CalculationArena gpu_arena(Backend::GPU, 2, dims, 1, false);
    copy(cpu_arena, gpu_arena);
    // Reset density/temperature on GPU to zeros, then inject
    cudaMemset(gpu_arena.state.density,     0, (std::size_t)Nx * Ny * sizeof(float));
    cudaMemset(gpu_arena.state.temperature, 0, (std::size_t)Nx * Ny * sizeof(float));

    PersistentState& gs = gpu_arena.state;
    gpu::compute_injection(1, Nx * Ny, gs.emitter_masks,
                         gs.emitter_densities, gs.emitter_temperatures,
                         gs.density, gs.temperature);

    auto gpu_d = d2h(gs.density,     Nx * Ny);
    auto gpu_t = d2h(gs.temperature, Nx * Ny);

    float md_d = max_diff(cs.density,     gpu_d.data(), Nx * Ny);
    float md_t = max_diff(cs.temperature, gpu_t.data(), Nx * Ny);
    EXPECT_LT(md_d, 1e-6f) << "inject_emitters density max delta: " << md_d;
    EXPECT_LT(md_t, 1e-6f) << "inject_emitters temperature max delta: " << md_t;
}

TEST(GpuParity, ApplyBuoyancy) {
    constexpr int Nx = 16, Ny = 16;
    constexpr float buoyancy = 5.0f, dt = 0.1f;
    int dims[] = {Nx, Ny};

    CalculationArena cpu_arena(Backend::CPU, 2, dims, 0, false);
    fill_gaussian(cpu_arena.state.temperature, Nx, Ny);

    float* cvx = cpu_arena.state.velocity;
    cpu::compute_buoyancy(Nx, Ny, buoyancy, dt, cpu_arena.state.temperature, cvx);

    CalculationArena gpu_arena(Backend::GPU, 2, dims, 0, false);
    copy(cpu_arena, gpu_arena);
    // Reset temperature to original, reset vx to zero
    {
        CalculationArena tmp(Backend::CPU, 2, dims, 0, false);
        fill_gaussian(tmp.state.temperature, Nx, Ny);
        cudaMemcpy(gpu_arena.state.temperature, tmp.state.temperature,
                   (std::size_t)Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(gpu_arena.state.velocity, 0,
                   (std::size_t)((Nx + 1) * Ny + Nx * (Ny + 1)) * sizeof(float));
    }

    float* gvx = gpu_arena.state.velocity;
    gpu::compute_buoyancy(Nx, Ny, buoyancy, dt, gpu_arena.state.temperature, gvx);

    auto gpu_vx = d2h(gvx, (Nx + 1) * Ny);

    float md = max_diff(cvx, gpu_vx.data(), (Nx + 1) * Ny);
    EXPECT_LT(md, 1e-5f) << "apply_buoyancy max delta: " << md;
}

TEST(GpuParity, ApplyCooling) {
    constexpr int Nx = 16, Ny = 16;
    constexpr float cooling = 0.3f, dt = 0.05f;
    int dims[] = {Nx, Ny};

    CalculationArena cpu_arena(Backend::CPU, 2, dims, 0, false);
    fill_gaussian(cpu_arena.state.temperature, Nx, Ny);

    // Save original temperature
    std::vector<float> temp0(cpu_arena.state.temperature,
                              cpu_arena.state.temperature + Nx * Ny);

    cpu::compute_cooling(Nx * Ny, cooling, dt, cpu_arena.state.temperature);

    CalculationArena gpu_arena(Backend::GPU, 2, dims, 0, false);
    cudaMemcpy(gpu_arena.state.temperature, temp0.data(),
               (std::size_t)Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);

    gpu::compute_cooling(Nx * Ny, cooling, dt, gpu_arena.state.temperature);

    auto gpu_t = d2h(gpu_arena.state.temperature, Nx * Ny);

    float md = max_diff(cpu_arena.state.temperature, gpu_t.data(), Nx * Ny);
    EXPECT_LT(md, 1e-6f) << "apply_cooling max delta: " << md;
}

TEST(GpuParity, Project) {
    constexpr int Nx = 16, Ny = 16;
    int dims[] = {Nx, Ny};

    CalculationArena cpu_arena(Backend::CPU, 2, dims, 0, true);
    fill_velocity(cpu_arena.state);

    // Save original velocity
    const int vx_size = (Nx + 1) * Ny;
    const int vy_size = Nx * (Ny + 1);
    std::vector<float> vx0(cpu_arena.state.velocity, cpu_arena.state.velocity + vx_size);
    std::vector<float> vy0(cpu_arena.state.velocity + vx_size,
                           cpu_arena.state.velocity + vx_size + vy_size);

    float* cvx = cpu_arena.state.velocity;
    float* cvy = cpu_arena.state.velocity + vx_size;
    std::vector<float> rhs_buf(Nx * Ny, 0.0f);
    cpu::compute_projection(Nx, Ny, cpu_arena.state.obstacle,
                 cvx, cvy, cpu_arena.scratch->pressure, rhs_buf.data(),
                 100, 1e-3f);

    CalculationArena gpu_arena(Backend::GPU, 2, dims, 0, true);
    cudaMemcpy(gpu_arena.state.velocity, vx0.data(),
               (std::size_t)vx_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_arena.state.velocity + vx_size, vy0.data(),
               (std::size_t)vy_size * sizeof(float), cudaMemcpyHostToDevice);

    float* gvx = gpu_arena.state.velocity;
    float* gvy = gpu_arena.state.velocity + vx_size;
    gpu::compute_projection(Nx, Ny, gpu_arena.state.obstacle,
                 gvx, gvy, gpu_arena.scratch->pressure, gpu_arena.scratch->tmp,
                 100, 1e-3f);

    auto gpu_vx = d2h(gvx, vx_size);
    auto gpu_vy = d2h(gvy, vy_size);

    float md_vx = max_diff(cvx, gpu_vx.data(), vx_size);
    float md_vy = max_diff(cvy, gpu_vy.data(), vy_size);
    EXPECT_LT(md_vx, 1e-4f) << "project vx max delta: " << md_vx;
    EXPECT_LT(md_vy, 1e-4f) << "project vy max delta: " << md_vy;
}

TEST(GpuParity, StepRoundTrip) {
    constexpr int Nx = 32, Ny = 32;
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

    constexpr int steps = 10;
    for (int k = 0; k < steps; ++k) {
        cpu::step(cs, *cpu_arena.scratch, 0.04f);
        gpu::step(gs, *gpu_arena.scratch, 0.04f);
    }

    auto gpu_density = d2h(gs.density, Nx * Ny);

    float md = max_diff(cs.density, gpu_density.data(), Nx * Ny);
    EXPECT_LT(md, 1e-4f) << "step round-trip density max delta: " << md;
}

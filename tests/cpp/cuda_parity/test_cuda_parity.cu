#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

#include "memory.hpp"
#include "state.hpp"
#include "step.hpp"

using namespace slipstream;

namespace {

float max_diff(const float* a, const float* b, int n) {
    float d = 0.0f;
    for (int k = 0; k < n; ++k) d = std::max(d, std::abs(a[k] - b[k]));
    return d;
}

} // namespace

TEST(CudaParity, SmokePlume) {
    constexpr int   Nx    = 16;
    constexpr int   Ny    = 16;
    constexpr int   Nz    = 16;
    constexpr int   STEPS = 20;
    constexpr float DT    = 0.04f;
    constexpr float TOL   = 1e-3f;

    const std::size_t cpu_sz    = required_state_bytes(Nx, Ny, Nz, 1, true);
    const std::size_t host_sz   = required_state_bytes(Nx, Ny, Nz, 1, false);
    const std::size_t device_sz = required_state_bytes(Nx, Ny, Nz, 1, true);

    void* cpu_buf    = host_alloc(cpu_sz);
    void* host_buf   = host_alloc(host_sz);
    void* device_buf = device_alloc(device_sz);

    State cs{};
    State hs{};
    State gs{};
    init_state(cs, cpu_buf,    cpu_sz,    Nx, Ny, Nz, 1, true);
    init_state(hs, host_buf,   host_sz,   Nx, Ny, Nz, 1, false);
    init_state(gs, device_buf, device_sz, Nx, Ny, Nz, 1, true);

    const int cy = Ny / 2;
    const int cz = Nz / 2;
    for (int i = 0; i < 4; ++i)
        for (int j = cy - 2; j < cy + 2; ++j)
            for (int k = cz - 2; k < cz + 2; ++k)
                cs.emitter_masks[(i * Ny + j) * Nz + k] = 1.0f;
    cs.emitter_densities[0]    = 1.0f;
    cs.emitter_temperatures[0] = 200.0f;
    cs.buoyancy = 5.0f;
    cs.cooling  = 0.3f;

    const int total_cells = Nx * Ny * Nz;
    const int v_size      = (Nx + 1) * Ny * Nz
                          + Nx * (Ny + 1) * Nz
                          + Nx * Ny * (Nz + 1);

    std::copy(cs.density,              cs.density              + total_cells,  hs.density);
    std::copy(cs.velocity,             cs.velocity             + v_size,       hs.velocity);
    std::copy(cs.temperature,          cs.temperature          + total_cells,  hs.temperature);
    std::copy(cs.obstacle,             cs.obstacle             + total_cells,  hs.obstacle);
    std::copy(cs.emitter_masks,        cs.emitter_masks        + total_cells,  hs.emitter_masks);
    std::copy(cs.emitter_densities,    cs.emitter_densities    + 1,            hs.emitter_densities);
    std::copy(cs.emitter_temperatures, cs.emitter_temperatures + 1,            hs.emitter_temperatures);
    hs.viscosity = cs.viscosity;
    hs.buoyancy  = cs.buoyancy;
    hs.cooling   = cs.cooling;
    hs.vorticity = cs.vorticity;

    upload(hs, gs);

    for (int k = 0; k < STEPS; ++k) {
        step_cpu(cs, DT);
        step_cuda(gs, DT);
    }

    download(gs, hs);

    float md_d = max_diff(cs.density,     hs.density,     total_cells);
    float md_t = max_diff(cs.temperature, hs.temperature, total_cells);
    float md_v = max_diff(cs.velocity,    hs.velocity,    v_size);

    EXPECT_LT(md_d, TOL) << "density max delta: "     << md_d;
    EXPECT_LT(md_t, TOL) << "temperature max delta: " << md_t;
    EXPECT_LT(md_v, TOL) << "velocity max delta: "    << md_v;

    device_free(device_buf);
    host_free(host_buf);
    host_free(cpu_buf);
}

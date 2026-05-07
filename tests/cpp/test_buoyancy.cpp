#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include "arena.hpp"
#include "cpu/step.hpp"

using namespace slipstream;

TEST(Buoyancy, ImpulseFromRest) {
    const int Nx = 8, Ny = 8;
    int dims[] = {Nx, Ny};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    PersistentState& s = arena.state;

    std::fill(s.temperature, s.temperature + Nx * Ny, 1.0f);
    s.buoyancy = 1.0f;

    cpu::step(s, *arena.scratch, 1.0f);

    float* vx = s.velocity;
    for (int i = 1; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            EXPECT_GT(vx[i * Ny + j], 0.0f);

    float* vy = s.velocity + (Nx + 1) * Ny;
    float max_div = 0.0f;
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            float div = vx[(i+1)*Ny + j] - vx[i*Ny + j]
                      + vy[i*(Ny+1) + j+1] - vy[i*(Ny+1) + j];
            max_div = std::max(max_div, std::abs(div));
        }
    EXPECT_LT(max_div, 1e-3f);
}

TEST(Buoyancy, CoolingDecay) {
    const int Nx = 8, Ny = 8;
    int dims[] = {Nx, Ny};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    PersistentState& s = arena.state;

    std::fill(s.temperature, s.temperature + Nx * Ny, 1.0f);
    s.cooling = 0.5f;

    for (int k = 1; k <= 4; ++k) {
        cpu::step(s, *arena.scratch, 1.0f);
        float expected = std::pow(0.5f, (float)k);
        EXPECT_NEAR(s.temperature[0], expected, 1e-5f);
    }
}

TEST(Buoyancy, PlumeRises) {
    const int N = 32;
    int dims[] = {N, N};
    CalculationArena arena(Backend::CPU, 2, dims, 1, true);
    PersistentState& s = arena.state;

    const int cx = N / 2;
    for (int i = 0; i < 4; ++i)
        for (int j = cx - 2; j < cx + 2; ++j)
            s.emitter_masks[i * N + j] = 1.0f;

    s.emitter_densities[0]    = 1.0f;
    s.emitter_temperatures[0] = 200.0f;

    s.buoyancy = 15.0f;
    s.cooling  = 0.5f;

    auto centre_of_mass = [&]() {
        float sum_d = 0.0f, sum_di = 0.0f;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float d = s.density[i * N + j];
                sum_d  += d;
                sum_di += d * (float)i;
            }
        return sum_d > 0.0f ? sum_di / sum_d : 0.0f;
    };

    for (int k = 0; k < 10; ++k) cpu::step(s, *arena.scratch, 0.04f);
    float com_10 = centre_of_mass();

    for (int k = 10; k < 40; ++k) cpu::step(s, *arena.scratch, 0.04f);
    float com_40 = centre_of_mass();

    EXPECT_GT(com_40, com_10);
}

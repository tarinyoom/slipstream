#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

TEST(Buoyancy, ImpulseFromRest) {
    const int Nx = 8, Ny = 8;
    int shape[] = {Nx, Ny};
    State state(shape, 2);

    std::vector<float> density_buf(Nx * Ny, 0.0f);
    std::vector<float> temp_buf(Nx * Ny, 1.0f);
    std::vector<float> vx_buf((Nx + 1) * Ny, 0.0f);
    std::vector<float> vy_buf(Nx * (Ny + 1), 0.0f);

    state.density     = density_buf;
    state.temperature = temp_buf;
    state.velocity    = {std::span<float>(vx_buf), std::span<float>(vy_buf)};
    state.buoyancy    = 1.0f;

    Solver solver(state, Backend::CPU);
    solver.step(1.0f);

    for (int i = 1; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            EXPECT_GT(vx_buf[i * Ny + j], 0.0f);

    float max_div = 0.0f;
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            float div = vx_buf[(i+1)*Ny + j] - vx_buf[i*Ny + j]
                      + vy_buf[i*(Ny+1) + j+1] - vy_buf[i*(Ny+1) + j];
            max_div = std::max(max_div, std::abs(div));
        }
    EXPECT_LT(max_div, 1e-3f);
}

TEST(Buoyancy, CoolingDecay) {
    const int Nx = 8, Ny = 8;
    int shape[] = {Nx, Ny};
    State state(shape, 2);

    std::vector<float> density_buf(Nx * Ny, 0.0f);
    std::vector<float> temp_buf(Nx * Ny, 1.0f);
    std::vector<float> vx_buf((Nx + 1) * Ny, 0.0f);
    std::vector<float> vy_buf(Nx * (Ny + 1), 0.0f);

    state.density     = density_buf;
    state.temperature = temp_buf;
    state.velocity    = {std::span<float>(vx_buf), std::span<float>(vy_buf)};
    state.cooling     = 0.5f;

    Solver solver(state, Backend::CPU);
    for (int k = 1; k <= 4; ++k) {
        solver.step(1.0f);
        float expected = std::pow(0.5f, (float)k);
        EXPECT_NEAR(temp_buf[0], expected, 1e-5f);
    }
}

TEST(Buoyancy, PlumeRises) {
    const int N = 32;
    int shape[] = {N, N};
    State state(shape, 2);

    std::vector<float> density_buf(N * N, 0.0f);
    std::vector<float> temp_buf(N * N, 0.0f);
    std::vector<float> vx_buf((N + 1) * N, 0.0f);
    std::vector<float> vy_buf(N * (N + 1), 0.0f);

    std::vector<char> em_mask_buf(N * N, 0);
    const int cx = N / 2;
    for (int i = 0; i < 4; ++i)
        for (int j = cx - 2; j < cx + 2; ++j)
            em_mask_buf[i * N + j] = 1;
    auto em_mask = reinterpret_cast<const bool*>(em_mask_buf.data());

    std::vector<float> em_dens  = {1.0f};
    std::vector<float> em_temps = {200.0f};

    state.density              = density_buf;
    state.temperature          = temp_buf;
    state.velocity             = {std::span<float>(vx_buf), std::span<float>(vy_buf)};
    state.emitter_masks        = std::span<const bool>(em_mask, N * N);
    state.emitter_densities    = em_dens;
    state.emitter_temperatures = em_temps;
    state.buoyancy             = 15.0f;
    state.cooling              = 0.5f;

    Solver solver(state, Backend::CPU);

    auto centre_of_mass = [&]() {
        float sum_d = 0.0f, sum_di = 0.0f;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float d = density_buf[i * N + j];
                sum_d  += d;
                sum_di += d * (float)i;
            }
        return sum_d > 0.0f ? sum_di / sum_d : 0.0f;
    };

    for (int k = 0; k < 10; ++k) solver.step(0.04f);
    float com_10 = centre_of_mass();

    for (int k = 10; k < 40; ++k) solver.step(0.04f);
    float com_40 = centre_of_mass();

    EXPECT_GT(com_40, com_10);
}

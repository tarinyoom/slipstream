#include <gtest/gtest.h>
#include <vector>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

TEST(Scaffolding, EmitterSetsDensityAtSourceCells) {
    const int N = 16;
    int shape[] = {N, N};
    State state(shape, 2);

    std::vector<float> density    (N * N, 0.0f);
    std::vector<float> vx_data   ((N + 1) * N, 0.0f);
    std::vector<float> vy_data   (N * (N + 1), 0.0f);

    bool em_mask_data[N * N] = {};
    em_mask_data[8 * N + 8] = true;
    std::vector<float> em_dens = {1.0f};

    state.density           = density;
    state.velocity          = {std::span<float>(vx_data), std::span<float>(vy_data)};
    state.emitter_masks     = std::span<const bool>(em_mask_data, N * N);
    state.emitter_densities = em_dens;

    Solver solver(state, Backend::CPU);
    solver.step(1.0f / 24.0f);

    EXPECT_GT(state.density[8 * N + 8], 0.0f);
}

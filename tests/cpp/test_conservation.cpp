#include <gtest/gtest.h>
#include <vector>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

TEST(Scaffolding, StepIncreasesDensity) {
    int shape[] = {16, 16};
    State state(shape, 2);

    std::vector<float> density    (16 * 16,       0.0f);
    std::vector<float> vx_data   (17 * 16,       0.0f);
    std::vector<float> vy_data   (16 * 17,       0.0f);
    std::vector<float> temperature(16 * 16,      0.0f);

    state.density     = density;
    state.velocity    = {std::span<float>(vx_data), std::span<float>(vy_data)};
    state.temperature = temperature;

    Solver solver(state, Backend::CPU);
    solver.step(1.0f / 24.0f);

    for (float d : state.density)
        EXPECT_GT(d, 0.0f);
}

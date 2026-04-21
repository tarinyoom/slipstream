#include <gtest/gtest.h>
#include <vector>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

TEST(Scaffolding, StepIncreasesDensity) {
    int shape[] = {16, 16};
    State state(shape, 2);

    std::vector<float> density    (16 * 16,     0.0f);
    std::vector<float> velocity   (16 * 16 * 2, 0.0f);
    std::vector<float> temperature(16 * 16,     0.0f);

    state.set_density    (density);
    state.set_velocity   (velocity);
    state.set_temperature(temperature);

    Solver solver(Backend::CPU);
    solver.step(state, 1.0f / 24.0f);

    for (float d : state.density())
        EXPECT_GT(d, 0.0f);
}

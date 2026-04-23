#include <gtest/gtest.h>
#include <vector>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

/* Divergence-free projection tests — added in physics implementation phase. */

TEST(Scaffolding, DivergencePlaceholder) {
    /* Verify that Solver constructs and destructs cleanly. */
    int shape[] = {8, 8};
    State state(shape, 2);

    std::vector<float> density    (8 * 8,  0.0f);
    std::vector<float> vx_data   (9 * 8,  0.0f);
    std::vector<float> vy_data   (8 * 9,  0.0f);
    std::vector<float> temperature(8 * 8, 0.0f);

    state.density     = density;
    state.velocity    = {std::span<float>(vx_data), std::span<float>(vy_data)};
    state.temperature = temperature;

    Solver solver(state, Backend::CPU);
}

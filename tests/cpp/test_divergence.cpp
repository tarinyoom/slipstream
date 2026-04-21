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

    std::vector<float> density    (8 * 8,     0.0f);
    std::vector<float> velocity   (8 * 8 * 2, 0.0f);
    std::vector<float> temperature(8 * 8,     0.0f);

    state.set_density    (density);
    state.set_velocity   (velocity);
    state.set_temperature(temperature);

    Solver solver(Backend::CPU);
}

#include <gtest/gtest.h>
#include <stdexcept>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

/* Pressure solver convergence tests — added in physics implementation phase. */

TEST(Scaffolding, ConvergencePlaceholder) {
    /* Verify that GPU backend throws. */
    int shape[] = {8, 8};
    State state(shape, 2);

    EXPECT_THROW(Solver(state, Backend::GPU), std::runtime_error);
}

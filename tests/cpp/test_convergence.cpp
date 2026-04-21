#include <gtest/gtest.h>
#include <stdexcept>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

/* Pressure solver convergence tests — added in physics implementation phase. */

TEST(Scaffolding, ConvergencePlaceholder) {
    /* Verify that GPU backend throws. */
    EXPECT_THROW(Solver(Backend::GPU), std::runtime_error);
}

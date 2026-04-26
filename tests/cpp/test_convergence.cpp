#include <gtest/gtest.h>
#include <stdexcept>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

TEST(Scaffolding, ConvergencePlaceholder) {
    int shape[] = {8, 8};
    State state(shape, 2);

    EXPECT_THROW(Solver(state, Backend::GPU), std::runtime_error);
}

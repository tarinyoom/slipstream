#include <gtest/gtest.h>
#include <stdexcept>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

TEST(Scaffolding, CpuSolverStepsWithoutError) {
    int dims[] = {8, 8};
    CpuState cs(2, dims);
    Solver solver(cs.s);
    solver.step(0.1f);
}

TEST(Scaffolding, GpuBackendThrows) {
    int dims[] = {8, 8};
    CpuState cs(2, dims);
    EXPECT_THROW(Solver(cs.s, Backend::GPU), std::runtime_error);
}

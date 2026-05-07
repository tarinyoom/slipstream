#include <gtest/gtest.h>
#include "arena.hpp"
#include "cpu/step.hpp"

using namespace slipstream;

TEST(Scaffolding, CpuSolverStepsWithoutError) {
    int dims[] = {8, 8};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    cpu::step(arena.state, *arena.scratch, 0.1f);
}

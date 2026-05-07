#include <gtest/gtest.h>
#include <vector>
#include "arena.hpp"
#include "cpu/step.hpp"

using namespace slipstream;

TEST(Scaffolding, EmitterSetsDensityAtSourceCells) {
    const int N = 16;
    int dims[] = {N, N};
    CalculationArena arena(Backend::CPU, 2, dims, 1, true);
    PersistentState& s = arena.state;

    s.emitter_masks[8 * N + 8]  = 1.0f;
    s.emitter_densities[0]       = 1.0f;

    cpu::step(s, *arena.scratch, 1.0f / 24.0f);

    EXPECT_GT(s.density[8 * N + 8], 0.0f);
}

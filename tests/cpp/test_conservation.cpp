#include <gtest/gtest.h>
#include <vector>
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

TEST(Scaffolding, EmitterSetsDensityAtSourceCells) {
    const int N = 16;
    int dims[] = {N, N};
    CpuState cs(2, dims, 1);
    State& s = cs.s;

    const_cast<bool*>(s.emitter_masks)[8 * N + 8]  = true;
    const_cast<float*>(s.emitter_densities)[0]      = 1.0f;

    Solver solver(s);
    solver.step(1.0f / 24.0f);

    EXPECT_GT(s.density[8 * N + 8], 0.0f);
}

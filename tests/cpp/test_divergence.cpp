#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "arena.hpp"
#include "cpu/step.hpp"

using namespace slipstream;

namespace {

float max_div_2d(int Nx, int Ny, const float* vx, const float* vy)
{
    float max_div = 0.0f;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            float div = vx[(i + 1) * Ny + j    ] - vx[i * Ny + j    ]
                      + vy[ i      * (Ny + 1) + j + 1] - vy[i * (Ny + 1) + j];
            max_div = std::max(max_div, std::abs(div));
        }
    }
    return max_div;
}

} // anonymous namespace

TEST(Projection, ZeroesDiv_UniformField) {
    int dims[] = {8, 8};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    PersistentState& s = arena.state;

    float* vx = s.velocity;
    float* vy = s.velocity + 9 * 8;
    std::fill(vx, vx + 9 * 8, 1.0f);
    std::fill(vy, vy + 8 * 9, 1.0f);

    cpu::step(s, *arena.scratch, 1.0f / 24.0f);

    EXPECT_LT(max_div_2d(8, 8, vx, vy), 1e-3f);
}

TEST(Projection, ZeroesDiv_RandomField) {
    int dims[] = {8, 8};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    PersistentState& s = arena.state;

    float* vx = s.velocity;
    float* vy = s.velocity + 9 * 8;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 1; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j)
            vx[i * dims[1] + j] = dist(rng);
    for (int i = 0; i < dims[0]; ++i)
        for (int j = 1; j < dims[1]; ++j)
            vy[i * (dims[1] + 1) + j] = dist(rng);

    cpu::step(s, *arena.scratch, 1.0f / 24.0f);

    EXPECT_LT(max_div_2d(8, 8, vx, vy), 1e-3f);
}

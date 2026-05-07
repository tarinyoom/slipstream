#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "state.hpp"
#include "solver.hpp"
#include "grid.hpp"

using namespace slipstream;

namespace {

float max_div_2d(const int* shape, const float* vx, const float* vy)
{
    float max_div = 0.0f;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            int lo[2]  = {i,     j    };
            int hix[2] = {i + 1, j    };
            int hiy[2] = {i,     j + 1};
            float div = vx[face_idx(shape, 2, 0, hix)]
                      - vx[face_idx(shape, 2, 0, lo )]
                      + vy[face_idx(shape, 2, 1, hiy)]
                      - vy[face_idx(shape, 2, 1, lo )];
            max_div = std::max(max_div, std::abs(div));
        }
    }
    return max_div;
}

} // anonymous namespace

TEST(Projection, ZeroesDiv_UniformField) {
    int dims[] = {8, 8};
    CpuState cs(2, dims);
    State& s = cs.s;

    float* vx = s.v;
    float* vy = s.v + 9 * 8;
    std::fill(vx, vx + 9 * 8, 1.0f);
    std::fill(vy, vy + 8 * 9, 1.0f);

    Solver solver(s);
    solver.step(1.0f / 24.0f);

    EXPECT_LT(max_div_2d(dims, vx, vy), 1e-3f);
}

TEST(Projection, ZeroesDiv_RandomField) {
    int dims[] = {8, 8};
    CpuState cs(2, dims);
    State& s = cs.s;

    float* vx = s.v;
    float* vy = s.v + 9 * 8;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 1; i < dims[0]; ++i)
        for (int j = 0; j < dims[1]; ++j) {
            int f[2] = {i, j};
            vx[face_idx(dims, 2, 0, f)] = dist(rng);
        }
    for (int i = 0; i < dims[0]; ++i)
        for (int j = 1; j < dims[1]; ++j) {
            int f[2] = {i, j};
            vy[face_idx(dims, 2, 1, f)] = dist(rng);
        }

    Solver solver(s);
    solver.step(1.0f / 24.0f);

    EXPECT_LT(max_div_2d(dims, vx, vy), 1e-3f);
}

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

float max_div_2d(const int shape[2],
                 const std::vector<float>& vx,
                 const std::vector<float>& vy)
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
    int shape[] = {8, 8};
    State state(shape, 2);

    std::vector<float> density    (8 * 8, 0.0f);
    std::vector<float> vx_data   (9 * 8, 1.0f);
    std::vector<float> vy_data   (8 * 9, 1.0f);
    std::vector<float> temperature(8 * 8, 0.0f);

    state.density     = density;
    state.velocity    = {std::span<float>(vx_data), std::span<float>(vy_data)};
    state.temperature = temperature;

    Solver solver(state, Backend::CPU);
    solver.step(1.0f / 24.0f);

    EXPECT_LT(max_div_2d(shape, vx_data, vy_data), 1e-3f);
}

TEST(Projection, ZeroesDiv_RandomField) {
    int shape[] = {8, 8};
    State state(shape, 2);

    std::vector<float> density    (8 * 8, 0.0f);
    std::vector<float> vx_data   (9 * 8, 0.0f);
    std::vector<float> vy_data   (8 * 9, 0.0f);
    std::vector<float> temperature(8 * 8, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 1; i < shape[0]; ++i)
        for (int j = 0; j < shape[1]; ++j) {
            int f[2] = {i, j};
            vx_data[face_idx(shape, 2, 0, f)] = dist(rng);
        }
    for (int i = 0; i < shape[0]; ++i)
        for (int j = 1; j < shape[1]; ++j) {
            int f[2] = {i, j};
            vy_data[face_idx(shape, 2, 1, f)] = dist(rng);
        }

    state.density     = density;
    state.velocity    = {std::span<float>(vx_data), std::span<float>(vy_data)};
    state.temperature = temperature;

    Solver solver(state, Backend::CPU);
    solver.step(1.0f / 24.0f);

    EXPECT_LT(max_div_2d(shape, vx_data, vy_data), 1e-3f);
}

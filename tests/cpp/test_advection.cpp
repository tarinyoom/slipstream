#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "arena.hpp"
#include "cpu/advect.hpp"

using namespace slipstream;

TEST(Advection, GaussianBumpTranslates) {
    const int Nx = 64, Ny = 64;
    int dims[] = {Nx, Ny};
    CalculationArena arena(Backend::CPU, 2, dims, 0, false);
    PersistentState& s = arena.state;

    const float cx0 = 24.0f, cy0 = 24.0f;
    const float sigma = 8.0f;
    const float A = 1.5f;

    float* vx = s.velocity;
    float* vy = s.velocity + (Nx + 1) * Ny;

    for (int i = 0; i <= Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            float r2 = (i - cx0)*(i - cx0) + (j - cy0)*(j - cy0);
            vx[i * Ny + j] = A * std::exp(-r2 / (2.0f * sigma * sigma));
        }
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j <= Ny; ++j) {
            float r2 = (i - cx0)*(i - cx0) + (j - cy0)*(j - cy0);
            vy[i * (Ny + 1) + j] = A * std::exp(-r2 / (2.0f * sigma * sigma));
        }

    const int steps = 6;
    const float dt = 0.5f;
    int max_faces = std::max((Nx + 1) * Ny, Nx * (Ny + 1));
    std::vector<float> scratch(max_faces, 0.0f);

    for (int step = 0; step < steps; ++step)
        cpu::advect_velocity(s, scratch.data(), dt);

    double wx = 0.0, wy = 0.0, wsum = 0.0;
    for (int i = 0; i <= Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            double w = vx[i * Ny + j];
            wx += w * i;
            wy += w * j;
            wsum += w;
        }
    float centroid_x = (float)(wx / wsum);
    float centroid_y = (float)(wy / wsum);

    float expected_x = cx0 + 0.5f * A * steps * dt;
    float expected_y = cy0 + 0.5f * A * steps * dt;
    float dist = std::hypot(centroid_x - expected_x, centroid_y - expected_y);
    EXPECT_LT(dist, 2.0f) << "centroid=(" << centroid_x << "," << centroid_y
                           << ") expected=(" << expected_x << "," << expected_y << ")";
}

TEST(Advection, XReflectionSymmetryPreserved) {
    const int Nx = 32, Ny = 32;
    int dims[] = {Nx, Ny};
    CalculationArena arena(Backend::CPU, 2, dims, 0, false);
    PersistentState& s = arena.state;

    float* vx = s.velocity;
    float* vy = s.velocity + (Nx + 1) * Ny;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i <= Nx / 2; ++i) {
            float val = (i == Nx / 2) ? 0.0f : dist(rng);
            vx[ i        * Ny + j] =  val;
            vx[(Nx - i)  * Ny + j] = -val;
        }
    }

    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i < Nx / 2; ++i) {
            float val = dist(rng);
            vy[ i           * (Ny + 1) + j] = val;
            vy[(Nx - 1 - i) * (Ny + 1) + j] = val;
        }
    }

    int max_faces = std::max((Nx + 1) * Ny, Nx * (Ny + 1));
    std::vector<float> scratch(max_faces, 0.0f);

    for (int step = 0; step < 4; ++step)
        cpu::advect_velocity(s, scratch.data(), 0.5f);

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            float lhs =  vx[ i       * Ny + j];
            float rhs = -vx[(Nx - i) * Ny + j];
            EXPECT_NEAR(lhs, rhs, 1e-4f) << "vx antisymmetry at (" << i << "," << j << ")";
        }
    }

    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            float lhs = vy[ i           * (Ny + 1) + j];
            float rhs = vy[(Nx - 1 - i) * (Ny + 1) + j];
            EXPECT_NEAR(lhs, rhs, 1e-4f) << "vy symmetry at (" << i << "," << j << ")";
        }
    }
}

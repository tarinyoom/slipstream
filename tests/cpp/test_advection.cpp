#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "cpu/advect.hpp"

using namespace slipstream;

TEST(Advection, GaussianBumpTranslates) {
    const int Nx = 64, Ny = 64;
    int shape[] = {Nx, Ny};

    const float cx0 = 24.0f, cy0 = 24.0f;
    const float sigma = 8.0f;
    const float A = 1.5f;

    std::vector<float> vx_data((Nx + 1) * Ny, 0.0f);
    std::vector<float> vy_data(Nx * (Ny + 1), 0.0f);

    for (int i = 0; i <= Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            float r2 = (i - cx0)*(i - cx0) + (j - cy0)*(j - cy0);
            vx_data[i * Ny + j] = A * std::exp(-r2 / (2.0f * sigma * sigma));
        }
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j <= Ny; ++j) {
            float r2 = (i - cx0)*(i - cx0) + (j - cy0)*(j - cy0);
            vy_data[i * (Ny + 1) + j] = A * std::exp(-r2 / (2.0f * sigma * sigma));
        }

    std::vector<std::span<float>> velocity = {
        std::span<float>(vx_data),
        std::span<float>(vy_data)
    };

    const int steps = 6;
    const float dt = 0.5f;
    int max_faces = std::max((Nx + 1) * Ny, Nx * (Ny + 1));
    std::vector<float> scratch(max_faces, 0.0f);
    std::span<const int> shape_span(shape, 2);

    for (int s = 0; s < steps; ++s)
        cpu::advect_velocity(shape_span, velocity, scratch, dt);

    // Weighted centroid of vx (all values positive, no threshold needed)
    double wx = 0.0, wy = 0.0, wsum = 0.0;
    for (int i = 0; i <= Nx; ++i)
        for (int j = 0; j < Ny; ++j) {
            double w = vx_data[i * Ny + j];
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
    int shape[] = {Nx, Ny};

    std::vector<float> vx_data((Nx + 1) * Ny,  0.0f);
    std::vector<float> vy_data( Nx * (Ny + 1), 0.0f);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i <= Nx / 2; ++i) {
            float val = (i == Nx / 2) ? 0.0f : dist(rng);
            vx_data[ i        * Ny + j] =  val;
            vx_data[(Nx - i)  * Ny + j] = -val;
        }
    }

    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i < Nx / 2; ++i) {
            float val = dist(rng);
            vy_data[ i           * (Ny + 1) + j] = val;
            vy_data[(Nx - 1 - i) * (Ny + 1) + j] = val;
        }
    }

    std::vector<std::span<float>> velocity = {
        std::span<float>(vx_data),
        std::span<float>(vy_data)
    };

    int max_faces = std::max((Nx + 1) * Ny, Nx * (Ny + 1));
    std::vector<float> scratch(max_faces, 0.0f);
    std::span<const int> shape_span(shape, 2);

    for (int step = 0; step < 4; ++step)
        cpu::advect_velocity(shape_span, velocity, scratch, 0.5f);

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            float lhs =  vx_data[ i       * Ny + j];
            float rhs = -vx_data[(Nx - i) * Ny + j];
            EXPECT_NEAR(lhs, rhs, 1e-4f) << "vx antisymmetry at (" << i << "," << j << ")";
        }
    }

    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            float lhs = vy_data[ i           * (Ny + 1) + j];
            float rhs = vy_data[(Nx - 1 - i) * (Ny + 1) + j];
            EXPECT_NEAR(lhs, rhs, 1e-4f) << "vy symmetry at (" << i << "," << j << ")";
        }
    }
}

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "ppm.hpp"
#include "state.hpp"
#include "solver.hpp"

using namespace slipstream;

namespace {

std::vector<uint8_t> read_ppm_pixels(const std::string& path, int& w, int& h) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    char magic[8];
    int maxval;
    std::fscanf(f, "%7s %d %d %d", magic, &w, &h, &maxval);
    std::fgetc(f);
    std::vector<uint8_t> pixels((size_t)w * h * 3);
    std::fread(pixels.data(), 1, pixels.size(), f);
    std::fclose(f);
    return pixels;
}

} // anonymous namespace

TEST(PPM, ZeroFieldProducesLUT0) {
    const int nx = 4, ny = 4;
    std::vector<float> density(nx * ny, 0.0f);
    const std::string path = "/tmp/test_ppm_t1.ppm";
    write_ppm(path.c_str(), density, nx, ny);

    int w, h;
    auto pixels = read_ppm_pixels(path, w, h);
    ASSERT_EQ(w, nx);
    ASSERT_EQ(h, ny);
    ASSERT_EQ(pixels.size(), (size_t)w * h * 3);

    for (int i = 0; i < w * h; ++i) {
        EXPECT_EQ(pixels[i*3+0], INFERNO_LUT[0][0]);
        EXPECT_EQ(pixels[i*3+1], INFERNO_LUT[0][1]);
        EXPECT_EQ(pixels[i*3+2], INFERNO_LUT[0][2]);
    }
}

TEST(PPM, PixelDimensionsMatchScale) {
    const int nx = 8, ny = 6, scale = 3;
    std::vector<float> density(nx * ny, 0.0f);
    const std::string path = "/tmp/test_ppm_t2.ppm";
    write_ppm(path.c_str(), density, nx, ny, 1.0f, scale);

    int w, h;
    auto pixels = read_ppm_pixels(path, w, h);
    EXPECT_EQ(w, nx * scale);
    EXPECT_EQ(h, ny * scale);
    EXPECT_EQ(pixels.size(), (size_t)w * h * 3);
}

TEST(PPM, KnownValueMapsToLUTEntry) {
    std::vector<float> density = {0.5f};
    const std::string path = "/tmp/test_ppm_t3.ppm";
    write_ppm(path.c_str(), density, 1, 1, 1.0f);

    int w, h;
    auto pixels = read_ppm_pixels(path, w, h);
    ASSERT_EQ((int)pixels.size(), 3);
    EXPECT_EQ(pixels[0], INFERNO_LUT[127][0]);
    EXPECT_EQ(pixels[1], INFERNO_LUT[127][1]);
    EXPECT_EQ(pixels[2], INFERNO_LUT[127][2]);
}

TEST(PPM, DensityAdvectsCrossFrames) {
    const int Nx = 32, Ny = 32;
    int shape[] = {Nx, Ny};
    State state(shape, 2);

    std::vector<float> density_buf(Nx * Ny, 0.0f);
    std::vector<float> vx_buf((Nx + 1) * Ny, 1.0f);
    std::vector<float> vy_buf(Nx * (Ny + 1), 0.0f);

    bool em_mask_data[32 * 32] = {};
    for (int j = 0; j < Ny; ++j)
        em_mask_data[0 * Ny + j] = true;
    std::vector<float> em_dens = {1.0f};

    state.density           = std::span<float>(density_buf);
    state.velocity          = {std::span<float>(vx_buf), std::span<float>(vy_buf)};
    state.emitter_masks     = std::span<const bool>(em_mask_data, (size_t)Nx * Ny);
    state.emitter_densities = std::span<const float>(em_dens);

    Solver solver(state);

    solver.step(1.0f);

    auto com_x = [&]() {
        double sum_x = 0.0, sum_w = 0.0;
        for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j) {
                float d = density_buf[i * Ny + j];
                sum_x += d * i;
                sum_w += d;
            }
        return sum_w > 0.0 ? sum_x / sum_w : 0.0;
    };

    double com0 = com_x();

    for (int s = 1; s < 8; ++s)
        solver.step(1.0f);

    double com8 = com_x();
    EXPECT_GT(com8, com0);
}

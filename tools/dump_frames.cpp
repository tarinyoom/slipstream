#include "state.hpp"
#include "solver.hpp"
#include "ppm.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

using namespace slipstream;

int main(int argc, char** argv) {
    int nx = 64, ny = 64, steps = 120, scale = 4;
    float dt = 0.04f;
    std::string output_dir = "frames";

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--nx")     == 0) nx         = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ny")     == 0) ny         = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--steps")  == 0) steps      = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dt")     == 0) dt         = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--scale")  == 0) scale      = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output") == 0) output_dir = argv[++i];
    }

    while (output_dir.size() > 1 && output_dir.back() == '/')
        output_dir.pop_back();

    int shape[] = {nx, ny};
    State state(shape, 2);

    std::vector<float> density_buf(nx * ny, 0.0f);
    std::vector<float> vx_buf((nx + 1) * ny, 0.0f);
    std::vector<float> vy_buf(nx * (ny + 1), 0.0f);

    for (float& v : vx_buf) v = 32.0f;

    state.density  = std::span<float>(density_buf);
    state.velocity = {std::span<float>(vx_buf), std::span<float>(vy_buf)};

    auto em_mask_arr = std::make_unique<bool[]>((size_t)nx * ny);
    std::fill(em_mask_arr.get(), em_mask_arr.get() + nx * ny, false);
    const int cx = nx / 2, cy = ny / 2;
    for (int i = cx - 2; i < cx + 2; ++i)
        for (int j = cy - 2; j < cy + 2; ++j)
            em_mask_arr[i * ny + j] = true;
    std::vector<float> em_dens = {1.0f};

    state.emitter_masks     = std::span<const bool>(em_mask_arr.get(), (size_t)nx * ny);
    state.emitter_densities = std::span<const float>(em_dens);

    std::filesystem::create_directories(output_dir);

    Solver solver(state);

    for (int s = 0; s < steps; ++s) {
        solver.step(dt);

        float max_d = *std::max_element(density_buf.begin(), density_buf.end());

        char path[512];
        std::snprintf(path, sizeof(path), "%s/frame_%04d.ppm", output_dir.c_str(), s);
        write_ppm(path, density_buf, nx, ny, 1.0f, scale);

        std::printf("frame %04d  max=%.4f\n", s, max_d);
    }

    return 0;
}

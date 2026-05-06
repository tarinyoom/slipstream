#include "state.hpp"
#include "solver.hpp"
#include "ppm.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace slipstream;

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <preset> [options]\n"
        "\n"
        "Presets:\n"
        "  single_emitter    Rising smoke plume from a central bottom emitter\n"
        "\n"
        "Options (all presets):\n"
        "  --output DIR      Output directory for PPM frames  (default: frames/)\n"
        "  --scale N         Pixel scale factor               (default: 4)\n"
        "  --vmax F          Density clamp for colour mapping (default: 1.0)\n"
        "\n"
        "Options (single_emitter):\n"
        "  --nx N            Grid width                       (default: 64)\n"
        "  --ny N            Grid height                      (default: 64)\n"
        "  --steps N         Number of frames to simulate     (default: 120)\n"
        "  --dt F            Timestep                         (default: 0.04)\n"
        "  --buoyancy F      Buoyancy coefficient             (default: 15.0)\n"
        "  --cooling F       Cooling decay rate               (default: 0.5)\n"
        "  --emitter-temp F  Temperature injected per step    (default: 200.0)\n"
        "  --emitter-dens F  Density injected per step        (default: 1.0)\n",
        prog);
}

static int run_single_emitter(int argc, char** argv) {
    int   nx           = 64;
    int   ny           = 64;
    int   steps        = 120;
    int   scale        = 4;
    float dt           = 0.04f;
    float buoyancy     = 15.0f;
    float cooling      = 0.5f;
    float emitter_temp = 200.0f;
    float emitter_dens = 1.0f;
    float vmax         = 1.0f;
    std::string output_dir = "frames";

    for (int i = 0; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--nx")           == 0) nx           = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ny")           == 0) ny           = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--steps")        == 0) steps        = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dt")           == 0) dt           = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--scale")        == 0) scale        = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--vmax")         == 0) vmax         = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--buoyancy")     == 0) buoyancy     = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--cooling")      == 0) cooling      = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--emitter-temp") == 0) emitter_temp = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--emitter-dens") == 0) emitter_dens = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--output")       == 0) output_dir   = argv[++i];
    }

    while (output_dir.size() > 1 && output_dir.back() == '/')
        output_dir.pop_back();

    int shape[] = {nx, ny};
    State state(shape, 2);

    std::vector<float> density_buf(nx * ny, 0.0f);
    std::vector<float> temperature_buf(nx * ny, 0.0f);
    std::vector<float> vx_buf((nx + 1) * ny, 0.0f);
    std::vector<float> vy_buf(nx * (ny + 1), 0.0f);

    state.density     = std::span<float>(density_buf);
    state.temperature = std::span<float>(temperature_buf);
    state.velocity    = {std::span<float>(vx_buf), std::span<float>(vy_buf)};
    state.buoyancy    = buoyancy;
    state.cooling     = cooling;

    auto em_mask_arr = std::make_unique<bool[]>((size_t)nx * ny);
    std::fill(em_mask_arr.get(), em_mask_arr.get() + nx * ny, false);
    const int cx = nx / 2, cy = ny / 2;
    for (int i = cx - 2; i < cx + 2; ++i)
        for (int j = cy - 2; j < cy + 2; ++j)
            em_mask_arr[i * ny + j] = true;
    std::vector<float> em_dens  = {emitter_dens};
    std::vector<float> em_temps = {emitter_temp};

    state.emitter_masks        = std::span<const bool>(em_mask_arr.get(), (size_t)nx * ny);
    state.emitter_densities    = std::span<const float>(em_dens);
    state.emitter_temperatures = std::span<const float>(em_temps);

    std::filesystem::create_directories(output_dir);

    Solver solver(state);

    for (int s = 0; s < steps; ++s) {
        solver.step(dt);

        float max_d = *std::max_element(density_buf.begin(), density_buf.end());

        char path[512];
        std::snprintf(path, sizeof(path), "%s/frame_%04d.ppm", output_dir.c_str(), s);
        write_ppm(path, density_buf, nx, ny, vmax, scale);

        std::printf("frame %04d  max=%.4f\n", s, max_d);
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char* preset = argv[1];

    if (std::strcmp(preset, "single_emitter") == 0)
        return run_single_emitter(argc - 2, argv + 2);

    std::fprintf(stderr, "error: unknown preset '%s'\n\n", preset);
    usage(argv[0]);
    return 1;
}

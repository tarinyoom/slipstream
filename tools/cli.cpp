#include "memory.hpp"
#include "ppm.hpp"
#include "state.hpp"
#include "step.hpp"
#ifdef SLIPSTREAM_HAS_OPENVDB
#include "vdb.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>

#ifdef SLIPSTREAM_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace slipstream;

enum class Backend { CPU, CUDA };
enum class OutputFormat { PPM, VDB };

static Backend select_backend(bool force_cpu) {
#ifdef SLIPSTREAM_HAS_CUDA
    if (!force_cpu) {
        int n = 0;
        cudaError_t err = cudaGetDeviceCount(&n);
        if (err == cudaSuccess && n > 0) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, 0);
            std::printf("backend: cuda  (device 0: %s)\n", prop.name);
            return Backend::CUDA;
        }
        std::fprintf(stderr,
            "warning: SLIPSTREAM_BUILD_CUDA is on but no CUDA device "
            "is visible; falling back to CPU\n");
    }
#else
    (void)force_cpu;
#endif
    std::printf("backend: cpu\n");
    return Backend::CPU;
}

struct PerfStats {
    int    steps;
    int    nx, ny;
    double solver_ms;
    double dtoh_ms;
    double render_ms;
};

static void print_perf_stats(const char* preset, const PerfStats& s) {
    std::printf("%s: %d steps, %dx%d -> solver %.2f ms (%.2f ms/step)",
                preset, s.steps, s.nx, s.ny,
                s.solver_ms, s.solver_ms / s.steps);

    double total = s.solver_ms;
    if (s.dtoh_ms   >= 0.0) { std::printf(", dtoh %.2f ms",   s.dtoh_ms);   total += s.dtoh_ms;   }
    if (s.render_ms >= 0.0) { std::printf(", render %.2f ms", s.render_ms); total += s.render_ms; }
    std::printf(", total %.2f ms\n", total);
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <preset> [--cpu] [--vdb]\n"
        "\n"
        "Presets:\n"
        "  single_emitter    Rising smoke plume from a bottom emitter (512x512, 120 steps)\n"
        "  perf_snapshot     Fixed 512x512 / 10-step run for profiling\n"
        "\n"
        "Options:\n"
        "  --cpu             Force the CPU backend, even on CUDA builds\n"
        "  --vdb             Write OpenVDB volumes instead of PPM (frame_NNNN.vdb)\n",
        prog);
}

static void write_frame(const char* dir, int step,
                        const State& s, float vmax, int scale,
                        OutputFormat fmt) {
    const int nx = s.nx, ny = s.ny;
    float max_d = *std::max_element(s.density, s.density + nx * ny);

    char path[512];
    if (fmt == OutputFormat::PPM) {
        std::snprintf(path, sizeof(path), "%s/frame_%04d.ppm", dir, step);
        write_ppm(path, std::span<const float>(s.density, nx * ny), nx, ny, vmax, scale);
    } else {
#ifdef SLIPSTREAM_HAS_OPENVDB
        std::snprintf(path, sizeof(path), "%s/frame_%04d.vdb", dir, step);
        write_vdb(path, std::span<const float>(s.density, nx * ny), nx, ny);
#endif
    }

    std::printf("frame %04d  max=%.4f\n", step, max_d);
}

static int run_single_emitter(Backend backend, OutputFormat fmt) {
    const int   nx           = 512;
    const int   ny           = 512;
    const int   steps        = 120;
    const int   scale        = 4;
    const float dt           = 0.04f;
    const float buoyancy     = 10.0f;
    const float cooling      = 0.5f;
    const float emitter_temp = 200.0f;
    const float emitter_dens = 1.0f;
    const float vmax         = 1.0f;
    const char* output_dir   = "frames";

    int dims[] = {nx, ny};
    std::span<const int> dims_span(dims, 2);
    const std::size_t sz = required_state_bytes(dims_span, 1, true);

    void* host_buf = host_alloc(sz);
    State host{};
    init_state(host, host_buf, sz, dims_span, 1, true);

    host.buoyancy = buoyancy;
    host.cooling  = cooling;

    const int i0 = 32, i1 = 48;
    const int j0 = ny / 2 - 8, j1 = ny / 2 + 8;
    for (int i = i0; i < i1; ++i)
        for (int j = j0; j < j1; ++j)
            host.emitter_masks[i * ny + j] = 1.0f;
    host.emitter_densities[0]    = emitter_dens;
    host.emitter_temperatures[0] = emitter_temp;

    std::filesystem::create_directories(output_dir);

    using clock = std::chrono::steady_clock;
    auto ms = [](clock::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    if (backend == Backend::CPU) {
        double solver_ms = 0.0, render_ms = 0.0;
        for (int step = 0; step < steps; ++step) {
            auto t0 = clock::now();
            step_cpu(host, dt);
            auto t1 = clock::now();
            write_frame(output_dir, step, host, vmax, scale, fmt);
            auto t2 = clock::now();
            solver_ms += ms(t1 - t0);
            render_ms += ms(t2 - t1);
        }
        print_perf_stats("single_emitter",
            {steps, nx, ny, solver_ms, -1.0, render_ms});
        host_free(host_buf);
        return 0;
    }

#ifdef SLIPSTREAM_HAS_CUDA
    void* dev_buf = device_alloc(sz);
    State dev{};
    init_state(dev, dev_buf, sz, dims_span, 1, true);
    upload(host, dev);

    double solver_ms = 0.0, dtoh_ms = 0.0, render_ms = 0.0;
    for (int step = 0; step < steps; ++step) {
        auto t0 = clock::now();
        step_cuda(dev, dt);
        cudaDeviceSynchronize();
        auto t1 = clock::now();
        download_field(dev, host, Field::Density);
        auto t2 = clock::now();
        write_frame(output_dir, step, host, vmax, scale, fmt);
        auto t3 = clock::now();
        solver_ms += ms(t1 - t0);
        dtoh_ms   += ms(t2 - t1);
        render_ms += ms(t3 - t2);
    }
    print_perf_stats("single_emitter",
        {steps, nx, ny, solver_ms, dtoh_ms, render_ms});

    device_free(dev_buf);
    host_free(host_buf);
    return 0;
#else
    host_free(host_buf);
    return 1;
#endif
}

static int run_perf_snapshot(Backend backend) {
    const int   nx           = 512;
    const int   ny           = 512;
    const int   steps        = 10;
    const float dt           = 0.04f;
    const float buoyancy     = 15.0f;
    const float cooling      = 0.5f;
    const float emitter_temp = 200.0f;
    const float emitter_dens = 1.0f;

    int dims[] = {nx, ny};
    std::span<const int> dims_span(dims, 2);
    const std::size_t sz = required_state_bytes(dims_span, 1, true);

    void* host_buf = host_alloc(sz);
    State host{};
    init_state(host, host_buf, sz, dims_span, 1, true);

    host.buoyancy = buoyancy;
    host.cooling  = cooling;

    const int cx = nx / 2, cy = ny / 2;
    for (int i = cx - 8; i < cx + 8; ++i)
        for (int j = cy - 8; j < cy + 8; ++j)
            host.emitter_masks[i * ny + j] = 1.0f;
    host.emitter_densities[0]    = emitter_dens;
    host.emitter_temperatures[0] = emitter_temp;

    using clock = std::chrono::steady_clock;
    auto ms = [](clock::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    if (backend == Backend::CPU) {
        step_cpu(host, dt);

        auto t0 = clock::now();
        for (int step = 0; step < steps; ++step)
            step_cpu(host, dt);
        auto t1 = clock::now();

        const double total = ms(t1 - t0);
        print_perf_stats("perf_snapshot",
            {steps, nx, ny, total, -1.0, -1.0});
        host_free(host_buf);
        return 0;
    }

#ifdef SLIPSTREAM_HAS_CUDA
    void* dev_buf = device_alloc(sz);
    State dev{};
    init_state(dev, dev_buf, sz, dims_span, 1, true);
    upload(host, dev);

    step_cuda(dev, dt);
    cudaDeviceSynchronize();

    auto t0 = clock::now();
    for (int step = 0; step < steps; ++step)
        step_cuda(dev, dt);
    cudaDeviceSynchronize();
    auto t1 = clock::now();

    download_field(dev, host, Field::Density);
    auto t2 = clock::now();

    const double total   = ms(t1 - t0);
    const double dtoh_ms = ms(t2 - t1);
    print_perf_stats("perf_snapshot",
        {steps, nx, ny, total, dtoh_ms, -1.0});

    device_free(dev_buf);
    host_free(host_buf);
    return 0;
#else
    host_free(host_buf);
    return 1;
#endif
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    bool force_cpu = false;
    bool want_vdb  = false;
    int w = 1;
    for (int r = 1; r < argc; ++r) {
        if      (std::strcmp(argv[r], "--cpu") == 0) force_cpu = true;
        else if (std::strcmp(argv[r], "--vdb") == 0) want_vdb  = true;
        else                                         argv[w++] = argv[r];
    }
    argc = w;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    OutputFormat fmt = OutputFormat::PPM;
    if (want_vdb) {
#ifdef SLIPSTREAM_HAS_OPENVDB
        fmt = OutputFormat::VDB;
#else
        std::fprintf(stderr,
            "error: --vdb requires building with -DSLIPSTREAM_BUILD_OPENVDB=ON\n");
        return 1;
#endif
    }

    Backend backend = select_backend(force_cpu);

    const char* preset = argv[1];

    if (std::strcmp(preset, "single_emitter") == 0)
        return run_single_emitter(backend, fmt);

    if (std::strcmp(preset, "perf_snapshot") == 0)
        return run_perf_snapshot(backend);

    std::fprintf(stderr, "error: unknown preset '%s'\n\n", preset);
    usage(argv[0]);
    return 1;
}

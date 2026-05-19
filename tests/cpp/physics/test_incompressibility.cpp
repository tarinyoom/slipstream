#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>

#include "memory.hpp"
#include "state.hpp"
#include "k_cpu/projection.hpp"

using namespace slipstream;

namespace {

constexpr int   N         = 16;
constexpr int   MAX_ITER  = 2000;
constexpr float TOL_SOLVE = 1e-4f;
constexpr float TOL_DIV   = 1e-3f;

float max_divergence_3d(int nx, int ny, int nz,
                        const float* vx, const float* vy, const float* vz,
                        const float* obstacle = nullptr)
{
    float max_div = 0.0f;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k) {
                int c = (i * ny + j) * nz + k;
                if (obstacle && obstacle[c] != 0.0f) continue;
                float div = vx[((i + 1) * ny + j) * nz + k] - vx[(i * ny + j) * nz + k]
                          + vy[(i * (ny + 1) + j + 1) * nz + k] - vy[(i * (ny + 1) + j) * nz + k]
                          + vz[(i * ny + j) * (nz + 1) + k + 1] - vz[(i * ny + j) * (nz + 1) + k];
                max_div = std::max(max_div, std::abs(div));
            }
    return max_div;
}

struct OwnedState {
    void* buf;
    State s;

    OwnedState(int nx, int ny, int nz, int n_emitters, bool ws) {
        const std::size_t sz = required_state_bytes(nx, ny, nz, n_emitters, ws);
        buf = host_alloc(sz);
        s = {};
        init_state(s, buf, sz, nx, ny, nz, n_emitters, ws);
    }
    ~OwnedState() { host_free(buf); }
    OwnedState(const OwnedState&) = delete;
    OwnedState& operator=(const OwnedState&) = delete;
};

void zero_domain_boundary_3d(int nx, int ny, int nz,
                             float* vx, float* vy, float* vz)
{
    for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k) {
            vx[(0  * ny + j) * nz + k] = 0.0f;
            vx[(nx * ny + j) * nz + k] = 0.0f;
        }
    for (int i = 0; i < nx; ++i)
        for (int k = 0; k < nz; ++k) {
            vy[(i * (ny + 1) + 0 ) * nz + k] = 0.0f;
            vy[(i * (ny + 1) + ny) * nz + k] = 0.0f;
        }
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            vz[(i * ny + j) * (nz + 1) + 0 ] = 0.0f;
            vz[(i * ny + j) * (nz + 1) + nz] = 0.0f;
        }
}

void project_3d(State& s) {
    float* vx = s.velocity;
    float* vy = vx + (s.nx + 1) * s.ny * s.nz;
    float* vz = vy + s.nx * (s.ny + 1) * s.nz;
    k_cpu::compute_projection(s.nx, s.ny, s.nz, s.obstacle, vx, vy, vz,
                              s.pressure, s.scratch, MAX_ITER, TOL_SOLVE);
}

} // namespace

TEST(Incompressible3D, ZeroField) {
    OwnedState st(N, N, N, 0, true);
    State& s = st.s;
    project_3d(s);

    const int vx_size = (N + 1) * N * N;
    const int vy_size = N * (N + 1) * N;
    float* vx = s.velocity;
    float* vy = vx + vx_size;
    float* vz = vy + vy_size;
    EXPECT_LT(max_divergence_3d(N, N, N, vx, vy, vz), TOL_DIV);
}

TEST(Incompressible3D, UniformFlow) {
    OwnedState st(N, N, N, 0, true);
    State& s = st.s;

    const int vx_size = (N + 1) * N * N;
    const int vy_size = N * (N + 1) * N;
    const int vz_size = N * N * (N + 1);
    float* vx = s.velocity;
    float* vy = vx + vx_size;
    float* vz = vy + vy_size;
    std::fill(vx, vx + vx_size, 1.0f);
    std::fill(vy, vy + vy_size, 1.0f);
    std::fill(vz, vz + vz_size, 1.0f);

    project_3d(s);
    EXPECT_LT(max_divergence_3d(N, N, N, vx, vy, vz), TOL_DIV);
}

TEST(Incompressible3D, RandomField) {
    OwnedState st(N, N, N, 0, true);
    State& s = st.s;

    const int vx_size = (N + 1) * N * N;
    const int vy_size = N * (N + 1) * N;
    const int vz_size = N * N * (N + 1);
    float* vx = s.velocity;
    float* vy = vx + vx_size;
    float* vz = vy + vy_size;

    std::mt19937 rng(2032);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int k = 0; k < vx_size; ++k) vx[k] = dist(rng);
    for (int k = 0; k < vy_size; ++k) vy[k] = dist(rng);
    for (int k = 0; k < vz_size; ++k) vz[k] = dist(rng);
    zero_domain_boundary_3d(N, N, N, vx, vy, vz);

    project_3d(s);
    EXPECT_LT(max_divergence_3d(N, N, N, vx, vy, vz), TOL_DIV);
}

TEST(Incompressible3D, WithObstacles) {
    OwnedState st(N, N, N, 0, true);
    State& s = st.s;

    const int vx_size = (N + 1) * N * N;
    const int vy_size = N * (N + 1) * N;
    const int vz_size = N * N * (N + 1);
    float* vx = s.velocity;
    float* vy = vx + vx_size;
    float* vz = vy + vy_size;

    std::mt19937 rng(2033);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int k = 0; k < vx_size; ++k) vx[k] = dist(rng);
    for (int k = 0; k < vy_size; ++k) vy[k] = dist(rng);
    for (int k = 0; k < vz_size; ++k) vz[k] = dist(rng);
    zero_domain_boundary_3d(N, N, N, vx, vy, vz);

    for (int i = 5; i < 11; ++i)
        for (int j = 5; j < 11; ++j)
            for (int k = 5; k < 11; ++k)
                s.obstacle[(i * N + j) * N + k] = 1.0f;

    project_3d(s);
    EXPECT_LT(max_divergence_3d(N, N, N, vx, vy, vz, s.obstacle), TOL_DIV);
}

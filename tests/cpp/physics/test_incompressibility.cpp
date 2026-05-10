#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <span>

#include "memory.hpp"
#include "state.hpp"
#include "k_cpu/projection.hpp"

using namespace slipstream;

namespace {

constexpr int   N        = 32;
constexpr int   MAX_ITER = 2000;
constexpr float TOL_SOLVE = 1e-4f;
constexpr float TOL_DIV   = 1e-3f;

float max_divergence(int nx, int ny, const float* vx, const float* vy,
                     const float* obstacle = nullptr)
{
    float max_div = 0.0f;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            if (obstacle && obstacle[i * ny + j] != 0.0f) continue;
            float div = vx[(i + 1) * ny + j    ] - vx[i * ny + j]
                      + vy[ i      * (ny + 1) + j + 1] - vy[i * (ny + 1) + j];
            max_div = std::max(max_div, std::abs(div));
        }
    return max_div;
}

void project(State& s) {
    float* vx = s.velocity;
    float* vy = s.velocity + (s.nx + 1) * s.ny;
    k_cpu::compute_projection(s.nx, s.ny, s.obstacle, vx, vy,
                              s.pressure, s.tmp, MAX_ITER, TOL_SOLVE);
}

struct OwnedState {
    void* buf;
    State s;

    OwnedState(int nx, int ny, int n_emitters, bool scratch) {
        int dims[] = {nx, ny};
        std::span<const int> dims_span(dims, 2);
        const std::size_t sz = required_state_bytes(dims_span, n_emitters, scratch);
        buf = host_alloc(sz);
        s = {};
        init_state(s, buf, sz, dims_span, n_emitters, scratch);
    }
    ~OwnedState() { host_free(buf); }
    OwnedState(const OwnedState&) = delete;
    OwnedState& operator=(const OwnedState&) = delete;
};

} // namespace

TEST(Incompressible, ZeroField) {
    OwnedState st(N, N, 0, true);
    State& s = st.s;

    project(s);

    float* vx = s.velocity;
    float* vy = s.velocity + (N + 1) * N;
    EXPECT_LT(max_divergence(N, N, vx, vy), TOL_DIV);
}

TEST(Incompressible, UniformFlow) {
    OwnedState st(N, N, 0, true);
    State& s = st.s;

    float* vx = s.velocity;
    float* vy = s.velocity + (N + 1) * N;
    std::fill(vx, vx + (N + 1) * N, 1.0f);
    std::fill(vy, vy + N * (N + 1), 1.0f);

    project(s);

    EXPECT_LT(max_divergence(N, N, vx, vy), TOL_DIV);
}

TEST(Incompressible, PureRotation) {
    OwnedState st(N, N, 0, true);
    State& s = st.s;

    float* vx = s.velocity;
    float* vy = s.velocity + (N + 1) * N;
    const float cx    = N * 0.5f;
    const float cy    = N * 0.5f;
    const float omega = 0.1f;

    for (int i = 0; i <= N; ++i)
        for (int j = 0; j < N; ++j)
            vx[i * N + j] = -omega * ((float)j + 0.5f - cy);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j <= N; ++j)
            vy[i * (N + 1) + j] = omega * ((float)i + 0.5f - cx);

    project(s);

    EXPECT_LT(max_divergence(N, N, vx, vy), TOL_DIV);
}

TEST(Incompressible, PointSource) {
    OwnedState st(N, N, 0, true);
    State& s = st.s;

    float* vx = s.velocity;
    float* vy = s.velocity + (N + 1) * N;

    const int   ci = N / 2;
    const int   cj = N / 2;
    const float V  = 5.0f;
    vx[ ci      * N + cj]       = -V;
    vx[(ci + 1) * N + cj]       = +V;
    vy[ ci * (N + 1) + cj      ] = -V;
    vy[ ci * (N + 1) + cj + 1  ] = +V;

    project(s);

    EXPECT_LT(max_divergence(N, N, vx, vy), TOL_DIV);
}

namespace {

void zero_domain_boundary(int n, float* vx, float* vy) {
    for (int j = 0; j < n; ++j) {
        vx[0 * n + j] = 0.0f;
        vx[n * n + j] = 0.0f;
    }
    for (int i = 0; i < n; ++i) {
        vy[i * (n + 1) + 0] = 0.0f;
        vy[i * (n + 1) + n] = 0.0f;
    }
}

} // namespace

TEST(Incompressible, RandomField) {
    OwnedState st(N, N, 0, true);
    State& s = st.s;

    float* vx = s.velocity;
    float* vy = s.velocity + (N + 1) * N;

    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int k = 0; k < (N + 1) * N; ++k) vx[k] = dist(rng);
    for (int k = 0; k < N * (N + 1); ++k) vy[k] = dist(rng);
    zero_domain_boundary(N, vx, vy);

    project(s);

    EXPECT_LT(max_divergence(N, N, vx, vy), TOL_DIV);
}

TEST(Incompressible, WithObstacles) {
    OwnedState st(N, N, 0, true);
    State& s = st.s;

    float* vx = s.velocity;
    float* vy = s.velocity + (N + 1) * N;

    std::mt19937 rng(2025);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int k = 0; k < (N + 1) * N; ++k) vx[k] = dist(rng);
    for (int k = 0; k < N * (N + 1); ++k) vy[k] = dist(rng);
    zero_domain_boundary(N, vx, vy);

    const int oi0 = 11, oj0 = 11, oh = 10, ow = 10;
    for (int i = oi0; i < oi0 + oh; ++i)
        for (int j = oj0; j < oj0 + ow; ++j)
            s.obstacle[i * N + j] = 1.0f;

    project(s);

    EXPECT_LT(max_divergence(N, N, vx, vy, s.obstacle), TOL_DIV);
}

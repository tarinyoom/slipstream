#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

#include "memory.hpp"
#include "state.hpp"
#include "step.hpp"

using namespace slipstream;

namespace {

constexpr int   N     = 24;
constexpr int   STEPS = 20;
constexpr float DT    = 0.5f;
constexpr float U     = 0.01f;
constexpr float V     = 0.01f;
constexpr float W     = 0.01f;

double total_mass(const float* density, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += density[i];
    return s;
}

bool all_finite(const float* density, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(density[i])) return false;
    return true;
}

void set_uniform_velocity(State& s, float u, float v, float w) {
    const int vx_size = (s.nx + 1) * s.ny * s.nz;
    const int vy_size = s.nx * (s.ny + 1) * s.nz;
    const int vz_size = s.nx * s.ny * (s.nz + 1);
    float* vx = s.velocity;
    float* vy = vx + vx_size;
    float* vz = vy + vy_size;
    std::fill(vx, vx + vx_size, u);
    std::fill(vy, vy + vy_size, v);
    std::fill(vz, vz + vz_size, w);
}

void run_steps(State& s) {
    for (int k = 0; k < STEPS; ++k) step_cpu(s, DT);
}

void check_conserved(double m0, const State& s) {
    const int total = s.nx * s.ny * s.nz;
    EXPECT_TRUE(all_finite(s.density, total));
    double m = total_mass(s.density, total);
    if (m0 > 0.0)
        EXPECT_LT(std::abs(m - m0) / m0, 0.01)
            << "mass " << m << " vs initial " << m0;
    else
        EXPECT_NEAR(m, 0.0, 1e-6);
}

struct OwnedState {
    void* buf;
    State s;

    OwnedState(int nx, int ny, int nz, int n_emitters, bool scratch) {
        const std::size_t sz = required_state_bytes(nx, ny, nz, n_emitters, scratch);
        buf = host_alloc(sz);
        s = {};
        init_state(s, buf, sz, nx, ny, nz, n_emitters, scratch);
    }
    ~OwnedState() { host_free(buf); }
    OwnedState(const OwnedState&) = delete;
    OwnedState& operator=(const OwnedState&) = delete;
};

} // namespace

TEST(ScalarTransport, ZeroField) {
    OwnedState st(N, N, N, 0, true);
    set_uniform_velocity(st.s, U, V, W);
    const int total = N * N * N;
    double m0 = total_mass(st.s.density, total);
    run_steps(st.s);
    check_conserved(m0, st.s);
}

TEST(ScalarTransport, UniformField) {
    OwnedState st(N, N, N, 0, true);
    const int total = N * N * N;
    std::fill(st.s.density, st.s.density + total, 0.5f);
    set_uniform_velocity(st.s, U, V, W);
    double m0 = total_mass(st.s.density, total);
    run_steps(st.s);
    check_conserved(m0, st.s);
}

TEST(ScalarTransport, SingleCellConcentration) {
    OwnedState st(N, N, N, 0, true);
    const int total = N * N * N;
    st.s.density[((N / 2) * N + (N / 2)) * N + (N / 2)] = 1000.0f;
    set_uniform_velocity(st.s, U, V, W);
    double m0 = total_mass(st.s.density, total);
    run_steps(st.s);
    check_conserved(m0, st.s);
}

TEST(ScalarTransport, Checkerboard) {
    OwnedState st(N, N, N, 0, true);
    const int total = N * N * N;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                st.s.density[(i * N + j) * N + k] = ((i + j + k) & 1) == 0 ? 1.0f : 0.0f;
    set_uniform_velocity(st.s, U, V, W);
    double m0 = total_mass(st.s.density, total);
    run_steps(st.s);
    check_conserved(m0, st.s);
}

TEST(ScalarTransport, StepFunction) {
    OwnedState st(N, N, N, 0, true);
    const int total = N * N * N;
    for (int i = 0; i < N / 2; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                st.s.density[(i * N + j) * N + k] = 1.0f;
    set_uniform_velocity(st.s, U, V, W);
    double m0 = total_mass(st.s.density, total);
    run_steps(st.s);
    check_conserved(m0, st.s);
}

TEST(ScalarTransport, NearFloatMax) {
    OwnedState st(N, N, N, 0, true);
    const int total = N * N * N;
    std::fill(st.s.density, st.s.density + total, 1e30f);
    set_uniform_velocity(st.s, U, V, W);
    double m0 = total_mass(st.s.density, total);
    run_steps(st.s);
    check_conserved(m0, st.s);
}

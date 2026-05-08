#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

#include "arena.hpp"
#include "cpu/step.hpp"

using namespace slipstream;

namespace {

constexpr int   N     = 32;
constexpr int   STEPS = 20;
constexpr float DT    = 0.5f;
constexpr float U     = 0.01f;
constexpr float V     = 0.01f;

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

void set_uniform_velocity(PersistentState& s, float u, float v) {
    float* vx = s.velocity;
    float* vy = s.velocity + (s.nx + 1) * s.ny;
    std::fill(vx, vx + (s.nx + 1) * s.ny, u);
    std::fill(vy, vy + s.nx * (s.ny + 1), v);
}

void run_steps(PersistentState& s, ScratchState& sc) {
    for (int k = 0; k < STEPS; ++k) cpu::step(s, sc, DT);
}

void check_conserved(double m0, const PersistentState& s) {
    const int total = s.nx * s.ny;
    EXPECT_TRUE(all_finite(s.density, total));
    double m = total_mass(s.density, total);
    if (m0 > 0.0)
        EXPECT_LT(std::abs(m - m0) / m0, 0.01)
            << "mass " << m << " vs initial " << m0;
    else
        EXPECT_NEAR(m, 0.0, 1e-6);
}

} // namespace

TEST(ScalarTransport, ZeroField) {
    int dims[] = {N, N};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    set_uniform_velocity(arena.state, U, V);
    double m0 = total_mass(arena.state.density, N * N);
    run_steps(arena.state, *arena.scratch);
    check_conserved(m0, arena.state);
}

TEST(ScalarTransport, UniformField) {
    int dims[] = {N, N};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    std::fill(arena.state.density, arena.state.density + N * N, 0.5f);
    set_uniform_velocity(arena.state, U, V);
    double m0 = total_mass(arena.state.density, N * N);
    run_steps(arena.state, *arena.scratch);
    check_conserved(m0, arena.state);
}

TEST(ScalarTransport, SingleCellConcentration) {
    int dims[] = {N, N};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    arena.state.density[(N / 2) * N + (N / 2)] = 1000.0f;
    set_uniform_velocity(arena.state, U, V);
    double m0 = total_mass(arena.state.density, N * N);
    run_steps(arena.state, *arena.scratch);
    check_conserved(m0, arena.state);
}

TEST(ScalarTransport, Checkerboard) {
    int dims[] = {N, N};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            arena.state.density[i * N + j] = ((i + j) & 1) == 0 ? 1.0f : 0.0f;
    set_uniform_velocity(arena.state, U, V);
    double m0 = total_mass(arena.state.density, N * N);
    run_steps(arena.state, *arena.scratch);
    check_conserved(m0, arena.state);
}

TEST(ScalarTransport, StepFunction) {
    int dims[] = {N, N};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    for (int i = 0; i < N / 2; ++i)
        for (int j = 0; j < N; ++j)
            arena.state.density[i * N + j] = 1.0f;
    set_uniform_velocity(arena.state, U, V);
    double m0 = total_mass(arena.state.density, N * N);
    run_steps(arena.state, *arena.scratch);
    check_conserved(m0, arena.state);
}

TEST(ScalarTransport, NearFloatMax) {
    int dims[] = {N, N};
    CalculationArena arena(Backend::CPU, 2, dims, 0, true);
    std::fill(arena.state.density, arena.state.density + N * N, 1e30f);
    set_uniform_velocity(arena.state, U, V);
    double m0 = total_mass(arena.state.density, N * N);
    run_steps(arena.state, *arena.scratch);
    check_conserved(m0, arena.state);
}

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <span>

#include "memory.hpp"
#include "state.hpp"
#include "cpu/step.hpp"

using namespace slipstream;

namespace {

constexpr int   N         = 32;
constexpr float DT        = 0.04f;
constexpr float EMITTER_T = 200.0f;
constexpr float EMITTER_D = 1.0f;
constexpr int   EMIT_J0   = N / 2 - 2;
constexpr int   EMIT_J1   = N / 2 + 2;

struct PlumeStatistics {
    double vertical_centroid;
    double vertical_spread;
    double vertical_skewness;
    double lateral_centroid;
    double lateral_spread;
    double lateral_skewness;
};

PlumeStatistics plume_stats(const float* density, int nx, int ny) {
    double sum_d = 0.0, sum_di = 0.0, sum_dj = 0.0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            double d = density[i * ny + j];
            sum_d  += d;
            sum_di += d * (double)i;
            sum_dj += d * (double)j;
        }
    if (sum_d <= 0.0) return {};
    double mi = sum_di / sum_d;
    double mj = sum_dj / sum_d;
    double m2i = 0.0, m2j = 0.0, m3i = 0.0, m3j = 0.0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            double d  = density[i * ny + j];
            double di = (double)i - mi;
            double dj = (double)j - mj;
            m2i += d * di * di;
            m2j += d * dj * dj;
            m3i += d * di * di * di;
            m3j += d * dj * dj * dj;
        }
    return {
        mi, m2i / sum_d, m3i / sum_d,
        mj, m2j / sum_d, m3j / sum_d,
    };
}

void set_emitter_block(State& s, int i_lo, int i_hi, int j_lo, int j_hi) {
    for (int i = i_lo; i < i_hi; ++i)
        for (int j = j_lo; j < j_hi; ++j)
            s.emitter_masks[i * s.ny + j] = 1.0f;
    s.emitter_densities[0]    = EMITTER_D;
    s.emitter_temperatures[0] = EMITTER_T;
}

void run(State& s, int steps) {
    for (int k = 0; k < steps; ++k) cpu::step(s, DT);
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

TEST(Plume, ZeroBuoyancy_CentroidStable) {
    OwnedState st(N, N, 1, true);
    State& s = st.s;
    set_emitter_block(s, 0, 4, EMIT_J0, EMIT_J1);
    s.buoyancy = 0.0f;
    s.cooling  = 0.0f;

    run(s, 1);
    auto p1 = plume_stats(s.density, N, N);
    run(s, 29);
    auto p30 = plume_stats(s.density, N, N);

    EXPECT_NEAR(p30.vertical_centroid, p1.vertical_centroid, 1e-3);
}

TEST(Plume, Rises_CentroidIncreases) {
    OwnedState st(N, N, 1, true);
    State& s = st.s;
    set_emitter_block(s, 0, 4, EMIT_J0, EMIT_J1);
    s.buoyancy = 15.0f;
    s.cooling  = 0.5f;

    run(s, 10);
    auto p10 = plume_stats(s.density, N, N);
    run(s, 30);
    auto p40 = plume_stats(s.density, N, N);

    EXPECT_GT(p40.vertical_centroid, p10.vertical_centroid);
}

TEST(Plume, Rises_SkewnessShifts) {
    OwnedState st(N, N, 1, true);
    State& s = st.s;
    set_emitter_block(s, 0, 4, EMIT_J0, EMIT_J1);
    s.buoyancy = 15.0f;
    s.cooling  = 2.0f;

    run(s, 5);
    auto p5 = plume_stats(s.density, N, N);
    run(s, 35);
    auto p40 = plume_stats(s.density, N, N);

    EXPECT_GT(std::abs(p40.vertical_skewness), std::abs(p5.vertical_skewness));
}

TEST(Plume, StrongerBuoyancy_HigherCentroid) {
    auto run_with = [&](float buoyancy) {
        OwnedState st(N, N, 1, true);
        State& s = st.s;
        set_emitter_block(s, 0, 4, EMIT_J0, EMIT_J1);
        s.buoyancy = buoyancy;
        s.cooling  = 0.5f;
        run(s, 40);
        return plume_stats(s.density, N, N);
    };

    auto weak   = run_with(2.0f);
    auto strong = run_with(8.0f);

    EXPECT_GT(strong.vertical_centroid, weak.vertical_centroid);
}

TEST(Plume, CoolingLimitsRise) {
    auto run_with = [&](float cooling) {
        OwnedState st(N, N, 1, true);
        State& s = st.s;
        set_emitter_block(s, 0, 4, EMIT_J0, EMIT_J1);
        s.buoyancy = 15.0f;
        s.cooling  = cooling;
        run(s, 40);
        return plume_stats(s.density, N, N);
    };

    auto without_cool = run_with(0.0f);
    auto with_cool    = run_with(2.0f);

    EXPECT_LT(with_cool.vertical_centroid, without_cool.vertical_centroid);
    EXPECT_LT(with_cool.vertical_spread,   without_cool.vertical_spread);
}

TEST(Plume, SymmetricEmitter_LateralSkewnessNearZero) {
    OwnedState st(N, N, 1, true);
    State& s = st.s;
    set_emitter_block(s, 0, 4, EMIT_J0, EMIT_J1);
    s.buoyancy = 0.0f;
    s.cooling  = 0.0f;

    float* vx = s.velocity;
    std::fill(vx, vx + (N + 1) * N, 0.5f);

    for (int milestone = 5; milestone <= 40; milestone += 5) {
        run(s, 5);
        auto p = plume_stats(s.density, N, N);
        EXPECT_LT(std::abs(p.lateral_skewness), 1e-6)
            << "step " << milestone << " lateral_skewness=" << p.lateral_skewness;
    }
}

TEST(Plume, WiderEmitter_GreaterInitialSpread) {
    auto run_with_height = [&](int height) {
        OwnedState st(N, N, 1, true);
        State& s = st.s;
        set_emitter_block(s, 0, height, EMIT_J0, EMIT_J1);
        s.buoyancy = 0.0f;
        s.cooling  = 0.0f;
        run(s, 20);
        return plume_stats(s.density, N, N);
    };

    auto narrow = run_with_height(2);
    auto wide   = run_with_height(8);

    EXPECT_GT(wide.vertical_spread, narrow.vertical_spread);
}

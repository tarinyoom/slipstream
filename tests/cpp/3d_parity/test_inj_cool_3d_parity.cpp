#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include "memory.hpp"
#include "state.hpp"
#include "k_cpu/injection.hpp"
#include "k_cpu/cooling.hpp"
#include "k_cpu/injection_3d.hpp"
#include "k_cpu/cooling_3d.hpp"

using namespace slipstream;

namespace {

constexpr int NX = 16;
constexpr int NY = 16;

struct OwnedState {
    void* buf;
    State s;

    OwnedState(std::span<const int> dims, int n_emitters, bool ws) {
        const std::size_t sz = required_state_bytes(dims, n_emitters, ws);
        buf = host_alloc(sz);
        s = {};
        init_state(s, buf, sz, dims, n_emitters, ws);
    }
    ~OwnedState() { host_free(buf); }
    OwnedState(const OwnedState&) = delete;
    OwnedState& operator=(const OwnedState&) = delete;
};

} // namespace

TEST(InjectionCooling3DParity, Injection) {
    int dims2[] = {NX, NY};
    int dims3[] = {NX, NY, 1};
    OwnedState st2(std::span<const int>(dims2, 2), 2, false);
    OwnedState st3(std::span<const int>(dims3, 3), 2, false);

    const int total = NX * NY;

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> mask_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> field_dist(-1.0f, 1.0f);

    for (int k = 0; k < 2 * total; ++k) {
        float v = mask_dist(rng) < 0.3f ? 1.0f : 0.0f;
        st2.s.emitter_masks[k] = v;
        st3.s.emitter_masks[k] = v;
    }
    for (int k = 0; k < total; ++k) {
        float d = field_dist(rng);
        float t = field_dist(rng);
        st2.s.density[k]     = d; st3.s.density[k]     = d;
        st2.s.temperature[k] = t; st3.s.temperature[k] = t;
    }
    st2.s.emitter_densities[0] = 1.0f;
    st2.s.emitter_densities[1] = 2.0f;
    st2.s.emitter_temperatures[0] = 100.0f;
    st2.s.emitter_temperatures[1] = 200.0f;
    st3.s.emitter_densities[0] = 1.0f;
    st3.s.emitter_densities[1] = 2.0f;
    st3.s.emitter_temperatures[0] = 100.0f;
    st3.s.emitter_temperatures[1] = 200.0f;

    k_cpu::compute_injection(2, total,
                             st2.s.emitter_masks,
                             st2.s.emitter_densities,
                             st2.s.emitter_temperatures,
                             st2.s.density, st2.s.temperature);
    k_cpu_3d::compute_injection(2, total,
                                st3.s.emitter_masks,
                                st3.s.emitter_densities,
                                st3.s.emitter_temperatures,
                                st3.s.density, st3.s.temperature);

    for (int k = 0; k < total; ++k) {
        EXPECT_EQ(st2.s.density[k],     st3.s.density[k])     << "k=" << k;
        EXPECT_EQ(st2.s.temperature[k], st3.s.temperature[k]) << "k=" << k;
    }
}

TEST(InjectionCooling3DParity, Cooling) {
    int dims2[] = {NX, NY};
    int dims3[] = {NX, NY, 1};
    OwnedState st2(std::span<const int>(dims2, 2), 0, false);
    OwnedState st3(std::span<const int>(dims3, 3), 0, false);

    const int total = NX * NY;

    std::mt19937 rng(2027);
    std::uniform_real_distribution<float> field_dist(0.0f, 300.0f);
    for (int k = 0; k < total; ++k) {
        float t = field_dist(rng);
        st2.s.temperature[k] = t;
        st3.s.temperature[k] = t;
    }

    k_cpu::compute_cooling   (total, 0.5f, 0.04f, st2.s.temperature);
    k_cpu_3d::compute_cooling(total, 0.5f, 0.04f, st3.s.temperature);

    for (int k = 0; k < total; ++k)
        EXPECT_EQ(st2.s.temperature[k], st3.s.temperature[k]) << "k=" << k;
}

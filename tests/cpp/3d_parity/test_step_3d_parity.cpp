#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <span>

#include "memory.hpp"
#include "state.hpp"
#include "step.hpp"

using namespace slipstream;

namespace {

constexpr int   NX        = 16;
constexpr int   NY        = 16;
constexpr int   STEPS     = 20;
constexpr float DT        = 0.04f;
constexpr float EMITTER_T = 200.0f;
constexpr float EMITTER_D = 1.0f;
constexpr float TOL       = 1e-4f;

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

TEST(Step3DParity, SmallPlume_Nz1) {
    int dims2[] = {NX, NY};
    int dims3[] = {NX, NY, 1};
    OwnedState st2(std::span<const int>(dims2, 2), 1, true);
    OwnedState st3(std::span<const int>(dims3, 3), 1, true);

    for (int i = 0; i < 3; ++i)
        for (int j = NY / 2 - 2; j < NY / 2 + 2; ++j) {
            st2.s.emitter_masks[i * NY + j] = 1.0f;
            st3.s.emitter_masks[(i * NY + j) * 1 + 0] = 1.0f;
        }
    st2.s.emitter_densities[0]    = EMITTER_D;
    st2.s.emitter_temperatures[0] = EMITTER_T;
    st3.s.emitter_densities[0]    = EMITTER_D;
    st3.s.emitter_temperatures[0] = EMITTER_T;

    st2.s.buoyancy = 10.0f;
    st2.s.cooling  = 0.5f;
    st3.s.buoyancy = 10.0f;
    st3.s.cooling  = 0.5f;

    for (int k = 0; k < STEPS; ++k) {
        step_cpu   (st2.s, DT);
        step_3d_cpu(st3.s, DT);
    }

    const int total = NX * NY;
    for (int c = 0; c < total; ++c) {
        EXPECT_NEAR(st2.s.density[c],     st3.s.density[c],     TOL) << "density c=" << c;
        EXPECT_NEAR(st2.s.temperature[c], st3.s.temperature[c], TOL) << "temperature c=" << c;
    }

    const int vx_size = (NX + 1) * NY;
    const int vy_size = NX * (NY + 1);
    float* vx2 = st2.s.velocity;
    float* vy2 = st2.s.velocity + vx_size;
    float* vx3 = st3.s.velocity;
    float* vy3 = st3.s.velocity + vx_size;
    for (int k = 0; k < vx_size; ++k)
        EXPECT_NEAR(vx2[k], vx3[k], TOL) << "vx k=" << k;
    for (int k = 0; k < vy_size; ++k)
        EXPECT_NEAR(vy2[k], vy3[k], TOL) << "vy k=" << k;
}

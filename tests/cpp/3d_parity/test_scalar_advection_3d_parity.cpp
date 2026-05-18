#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <span>

#include "memory.hpp"
#include "state.hpp"
#include "k_cpu/advection.hpp"
#include "k_cpu/advection_3d.hpp"

using namespace slipstream;

namespace {

constexpr int   NX = 16;
constexpr int   NY = 16;
constexpr float DT = 0.7f;

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

TEST(ScalarAdvection3DParity, RandomField_Nz1) {
    int dims2[] = {NX, NY};
    int dims3[] = {NX, NY, 1};
    OwnedState st2(std::span<const int>(dims2, 2), 0, true);
    OwnedState st3(std::span<const int>(dims3, 3), 0, true);

    const int total = NX * NY;
    const int vx_size = (NX + 1) * NY;
    const int vy_size = NX * (NY + 1);

    std::mt19937 rng(2028);
    std::uniform_real_distribution<float> field_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> vel_dist  (-0.5f, 0.5f);

    for (int k = 0; k < total; ++k) {
        float d = field_dist(rng);
        st2.s.density[k] = d;
        st3.s.density[k] = d;
    }
    for (int k = 0; k < vx_size; ++k) {
        float v = vel_dist(rng);
        st2.s.velocity[k] = v;
        st3.s.velocity[k] = v;
    }
    for (int k = 0; k < vy_size; ++k) {
        float v = vel_dist(rng);
        st2.s.velocity[vx_size + k] = v;
        st3.s.velocity[vx_size + k] = v;
    }
    // vz lives at offset (vx_size + vy_size) in the 3D buffer; left zero.

    float* vx2 = st2.s.velocity;
    float* vy2 = st2.s.velocity + vx_size;
    k_cpu::compute_scalar_advection(NX, NY, vx2, vy2,
                                    st2.s.density, st2.s.scratch, DT);

    float* vx3 = st3.s.velocity;
    float* vy3 = st3.s.velocity + vx_size;
    float* vz3 = st3.s.velocity + vx_size + vy_size;
    k_cpu_3d::compute_scalar_advection(NX, NY, 1, vx3, vy3, vz3,
                                       st3.s.density, st3.s.scratch, DT);

    for (int k = 0; k < total; ++k)
        EXPECT_EQ(st2.s.scratch[k], st3.s.scratch[k]) << "k=" << k;
}

TEST(ScalarAdvection3DParity, PointSource_Nz1) {
    int dims2[] = {NX, NY};
    int dims3[] = {NX, NY, 1};
    OwnedState st2(std::span<const int>(dims2, 2), 0, true);
    OwnedState st3(std::span<const int>(dims3, 3), 0, true);

    st2.s.density[(NX/2) * NY + (NY/2)] = 100.0f;
    st3.s.density[(NX/2) * NY + (NY/2)] = 100.0f;

    const int vx_size = (NX + 1) * NY;
    float* vx2 = st2.s.velocity;
    float* vy2 = st2.s.velocity + vx_size;
    std::fill(vx2, vx2 + vx_size, 0.3f);
    std::fill(vy2, vy2 + NX * (NY + 1), -0.2f);

    float* vx3 = st3.s.velocity;
    float* vy3 = st3.s.velocity + vx_size;
    float* vz3 = st3.s.velocity + vx_size + NX * (NY + 1);
    std::fill(vx3, vx3 + vx_size, 0.3f);
    std::fill(vy3, vy3 + NX * (NY + 1), -0.2f);

    k_cpu::compute_scalar_advection   (NX, NY, vx2, vy2,
                                       st2.s.density, st2.s.scratch, DT);
    k_cpu_3d::compute_scalar_advection(NX, NY, 1, vx3, vy3, vz3,
                                       st3.s.density, st3.s.scratch, DT);

    for (int k = 0; k < NX * NY; ++k)
        EXPECT_EQ(st2.s.scratch[k], st3.s.scratch[k]) << "k=" << k;
}

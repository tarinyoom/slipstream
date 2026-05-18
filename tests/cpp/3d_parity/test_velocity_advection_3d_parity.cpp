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
constexpr float DT = 0.5f;

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

TEST(VelocityAdvection3DParity, RandomField_Nz1) {
    int dims2[] = {NX, NY};
    int dims3[] = {NX, NY, 1};
    OwnedState st2(std::span<const int>(dims2, 2), 0, true);
    OwnedState st3(std::span<const int>(dims3, 3), 0, true);

    const int vx_size = (NX + 1) * NY;
    const int vy_size = NX * (NY + 1);

    std::mt19937 rng(2029);
    std::uniform_real_distribution<float> vel_dist(-0.5f, 0.5f);

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
    k_cpu::compute_velocity_advection(NX, NY, vx2, vy2, st2.s.scratch, DT);

    float* vx3 = st3.s.velocity;
    float* vy3 = st3.s.velocity + vx_size;
    float* vz3 = st3.s.velocity + vx_size + vy_size;
    k_cpu_3d::compute_velocity_advection(NX, NY, 1, vx3, vy3, vz3, st3.s.scratch, DT);

    for (int k = 0; k < vx_size; ++k)
        EXPECT_EQ(vx2[k], vx3[k]) << "vx k=" << k;
    for (int k = 0; k < vy_size; ++k)
        EXPECT_EQ(vy2[k], vy3[k]) << "vy k=" << k;
}

TEST(VelocityAdvection3DParity, ShearFlow_Nz1) {
    int dims2[] = {NX, NY};
    int dims3[] = {NX, NY, 1};
    OwnedState st2(std::span<const int>(dims2, 2), 0, true);
    OwnedState st3(std::span<const int>(dims3, 3), 0, true);

    const int vx_size = (NX + 1) * NY;
    const int vy_size = NX * (NY + 1);

    for (int i = 0; i < NX + 1; ++i) {
        for (int j = 0; j < NY; ++j) {
            float v = 0.2f * (float)j;
            st2.s.velocity[i * NY + j] = v;
            st3.s.velocity[i * NY + j] = v;
        }
    }
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY + 1; ++j) {
            float v = -0.1f * (float)i;
            st2.s.velocity[vx_size + i * (NY + 1) + j] = v;
            st3.s.velocity[vx_size + i * (NY + 1) + j] = v;
        }
    }

    float* vx2 = st2.s.velocity;
    float* vy2 = st2.s.velocity + vx_size;
    k_cpu::compute_velocity_advection(NX, NY, vx2, vy2, st2.s.scratch, DT);

    float* vx3 = st3.s.velocity;
    float* vy3 = st3.s.velocity + vx_size;
    float* vz3 = st3.s.velocity + vx_size + vy_size;
    k_cpu_3d::compute_velocity_advection(NX, NY, 1, vx3, vy3, vz3, st3.s.scratch, DT);

    for (int k = 0; k < vx_size; ++k)
        EXPECT_EQ(vx2[k], vx3[k]) << "vx k=" << k;
    for (int k = 0; k < vy_size; ++k)
        EXPECT_EQ(vy2[k], vy3[k]) << "vy k=" << k;
}

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <span>

#include "memory.hpp"
#include "state.hpp"
#include "k_cpu/projection.hpp"
#include "k_cpu/projection_3d.hpp"

using namespace slipstream;

namespace {

constexpr int   NX        = 16;
constexpr int   NY        = 16;
constexpr int   MAX_ITER  = 200;
constexpr float TOL_SOLVE = 1e-5f;
constexpr float TOL_FIELD = 1e-5f;

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

void zero_domain_boundary(int nx, int ny, float* vx, float* vy) {
    for (int j = 0; j < ny; ++j) {
        vx[0  * ny + j] = 0.0f;
        vx[nx * ny + j] = 0.0f;
    }
    for (int i = 0; i < nx; ++i) {
        vy[i * (ny + 1) + 0 ] = 0.0f;
        vy[i * (ny + 1) + ny] = 0.0f;
    }
}

} // namespace

TEST(Projection3DParity, RandomField_Nz1) {
    int dims2[] = {NX, NY};
    int dims3[] = {NX, NY, 1};
    OwnedState st2(std::span<const int>(dims2, 2), 0, true);
    OwnedState st3(std::span<const int>(dims3, 3), 0, true);

    const int vx_size = (NX + 1) * NY;
    const int vy_size = NX * (NY + 1);

    std::mt19937 rng(2030);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int k = 0; k < vx_size; ++k) {
        float v = dist(rng);
        st2.s.velocity[k] = v;
        st3.s.velocity[k] = v;
    }
    for (int k = 0; k < vy_size; ++k) {
        float v = dist(rng);
        st2.s.velocity[vx_size + k] = v;
        st3.s.velocity[vx_size + k] = v;
    }
    zero_domain_boundary(NX, NY, st2.s.velocity, st2.s.velocity + vx_size);
    zero_domain_boundary(NX, NY, st3.s.velocity, st3.s.velocity + vx_size);

    float* vx2 = st2.s.velocity;
    float* vy2 = st2.s.velocity + vx_size;
    k_cpu::compute_projection(NX, NY, nullptr, vx2, vy2,
                              st2.s.pressure, st2.s.scratch,
                              MAX_ITER, TOL_SOLVE);

    float* vx3 = st3.s.velocity;
    float* vy3 = st3.s.velocity + vx_size;
    float* vz3 = st3.s.velocity + vx_size + vy_size;
    k_cpu_3d::compute_projection(NX, NY, 1, nullptr, vx3, vy3, vz3,
                                 st3.s.pressure, st3.s.scratch,
                                 MAX_ITER, TOL_SOLVE);

    for (int k = 0; k < vx_size; ++k)
        EXPECT_NEAR(vx2[k], vx3[k], TOL_FIELD) << "vx k=" << k;
    for (int k = 0; k < vy_size; ++k)
        EXPECT_NEAR(vy2[k], vy3[k], TOL_FIELD) << "vy k=" << k;
    for (int k = 0; k < NX * NY; ++k)
        EXPECT_NEAR(st2.s.pressure[k], st3.s.pressure[k], TOL_FIELD) << "p k=" << k;
}

TEST(Projection3DParity, WithObstacles_Nz1) {
    int dims2[] = {NX, NY};
    int dims3[] = {NX, NY, 1};
    OwnedState st2(std::span<const int>(dims2, 2), 0, true);
    OwnedState st3(std::span<const int>(dims3, 3), 0, true);

    const int vx_size = (NX + 1) * NY;
    const int vy_size = NX * (NY + 1);

    std::mt19937 rng(2031);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int k = 0; k < vx_size; ++k) {
        float v = dist(rng);
        st2.s.velocity[k] = v;
        st3.s.velocity[k] = v;
    }
    for (int k = 0; k < vy_size; ++k) {
        float v = dist(rng);
        st2.s.velocity[vx_size + k] = v;
        st3.s.velocity[vx_size + k] = v;
    }
    zero_domain_boundary(NX, NY, st2.s.velocity, st2.s.velocity + vx_size);
    zero_domain_boundary(NX, NY, st3.s.velocity, st3.s.velocity + vx_size);

    const int oi0 = 5, oj0 = 5, oh = 4, ow = 4;
    for (int i = oi0; i < oi0 + oh; ++i)
        for (int j = oj0; j < oj0 + ow; ++j) {
            st2.s.obstacle[i * NY + j] = 1.0f;
            st3.s.obstacle[(i * NY + j) * 1 + 0] = 1.0f;
        }

    float* vx2 = st2.s.velocity;
    float* vy2 = st2.s.velocity + vx_size;
    k_cpu::compute_projection(NX, NY, st2.s.obstacle, vx2, vy2,
                              st2.s.pressure, st2.s.scratch,
                              MAX_ITER, TOL_SOLVE);

    float* vx3 = st3.s.velocity;
    float* vy3 = st3.s.velocity + vx_size;
    float* vz3 = st3.s.velocity + vx_size + vy_size;
    k_cpu_3d::compute_projection(NX, NY, 1, st3.s.obstacle, vx3, vy3, vz3,
                                 st3.s.pressure, st3.s.scratch,
                                 MAX_ITER, TOL_SOLVE);

    for (int k = 0; k < vx_size; ++k)
        EXPECT_NEAR(vx2[k], vx3[k], TOL_FIELD) << "vx k=" << k;
    for (int k = 0; k < vy_size; ++k)
        EXPECT_NEAR(vy2[k], vy3[k], TOL_FIELD) << "vy k=" << k;
    for (int k = 0; k < NX * NY; ++k)
        EXPECT_NEAR(st2.s.pressure[k], st3.s.pressure[k], TOL_FIELD) << "p k=" << k;
}

#include <gtest/gtest.h>
#include <span>

#include "memory.hpp"
#include "state.hpp"

using namespace slipstream;

namespace {

std::size_t expected_bytes_3d(int nx, int ny, int nz, int n_emitters, bool ws) {
    const std::size_t total  = (std::size_t)nx * ny * nz;
    const std::size_t v_size = (std::size_t)(nx + 1) * ny * nz
                             + (std::size_t)nx * (ny + 1) * nz
                             + (std::size_t)nx * ny * (nz + 1);
    const std::size_t f_x = (std::size_t)(nx + 1) * ny * nz;
    const std::size_t f_y = (std::size_t)nx * (ny + 1) * nz;
    const std::size_t f_z = (std::size_t)nx * ny * (nz + 1);
    std::size_t max_faces = f_x;
    if (f_y > max_faces) max_faces = f_y;
    if (f_z > max_faces) max_faces = f_z;
#ifdef SLIPSTREAM_HAS_CUDA
    const std::size_t scratch = std::max(max_faces, 2 * total);
#else
    const std::size_t scratch = max_faces;
#endif

    std::size_t floats = 0;
    floats += total;
    floats += v_size;
    floats += total;
    floats += total;
    if (n_emitters > 0) {
        floats += (std::size_t)n_emitters * total;
        floats += (std::size_t)n_emitters;
        floats += (std::size_t)n_emitters;
    }
    if (ws) {
        floats += total;
        floats += scratch;
    }
    return floats * sizeof(float);
}

} // namespace

TEST(Allocate3D, ByteCountMatches) {
    int dims[] = {32, 32, 32};
    std::span<const int> dims_span(dims, 3);

    const std::size_t got      = required_state_bytes(dims_span, 1, true);
    const std::size_t expected = expected_bytes_3d(32, 32, 32, 1, true);
    EXPECT_EQ(got, expected);
}

TEST(Allocate3D, InitAndFree) {
    int dims[] = {16, 24, 32};
    std::span<const int> dims_span(dims, 3);

    const std::size_t sz = required_state_bytes(dims_span, 2, true);
    void* buf = host_alloc(sz);
    State s{};
    init_state(s, buf, sz, dims_span, 2, true);

    EXPECT_EQ(s.nx, 16);
    EXPECT_EQ(s.ny, 24);
    EXPECT_EQ(s.nz, 32);
    EXPECT_EQ(s.n_emitters, 2);
    EXPECT_NE(s.density, nullptr);
    EXPECT_NE(s.velocity, nullptr);
    EXPECT_NE(s.pressure, nullptr);
    EXPECT_NE(s.scratch, nullptr);
    EXPECT_NE(s.emitter_masks, nullptr);

    host_free(buf);
}

TEST(Allocate3D, NoWorkspace) {
    int dims[] = {8, 8, 8};
    std::span<const int> dims_span(dims, 3);

    const std::size_t sz = required_state_bytes(dims_span, 0, false);
    void* buf = host_alloc(sz);
    State s{};
    init_state(s, buf, sz, dims_span, 0, false);

    EXPECT_EQ(s.pressure, nullptr);
    EXPECT_EQ(s.scratch, nullptr);
    EXPECT_EQ(s.emitter_masks, nullptr);

    host_free(buf);
}

TEST(Allocate2D_Unchanged, ByteCountAndNzZero) {
    int dims[] = {32, 32};
    std::span<const int> dims_span(dims, 2);

    const std::size_t sz = required_state_bytes(dims_span, 1, true);
    void* buf = host_alloc(sz);
    State s{};
    init_state(s, buf, sz, dims_span, 1, true);

    EXPECT_EQ(s.nx, 32);
    EXPECT_EQ(s.ny, 32);
    EXPECT_EQ(s.nz, 0);

    host_free(buf);
}

#include "advection.hpp"

#include <algorithm>
#include <cmath>

namespace slipstream::k_cpu {

namespace {

float trilinear_sample(const float* v, const int fs[3],
                       float px, float py, float pz)
{
    int i0 = (int)std::floor(px);
    int j0 = (int)std::floor(py);
    int k0 = (int)std::floor(pz);
    int i1 = i0 + 1;
    int j1 = j0 + 1;
    int k1 = k0 + 1;
    float tx = px - (float)i0;
    float ty = py - (float)j0;
    float tz = pz - (float)k0;

    i0 = std::clamp(i0, 0, fs[0] - 1);
    i1 = std::clamp(i1, 0, fs[0] - 1);
    j0 = std::clamp(j0, 0, fs[1] - 1);
    j1 = std::clamp(j1, 0, fs[1] - 1);
    k0 = std::clamp(k0, 0, fs[2] - 1);
    k1 = std::clamp(k1, 0, fs[2] - 1);

    auto at = [&](int a, int b, int c) {
        return v[(a * fs[1] + b) * fs[2] + c];
    };

    float v000 = at(i0, j0, k0);
    float v100 = at(i1, j0, k0);
    float v010 = at(i0, j1, k0);
    float v110 = at(i1, j1, k0);
    float v001 = at(i0, j0, k1);
    float v101 = at(i1, j0, k1);
    float v011 = at(i0, j1, k1);
    float v111 = at(i1, j1, k1);

    float w000 = (1.0f - tx) * (1.0f - ty) * (1.0f - tz);
    float w100 =          tx * (1.0f - ty) * (1.0f - tz);
    float w010 = (1.0f - tx) *          ty * (1.0f - tz);
    float w110 =          tx *          ty * (1.0f - tz);
    float w001 = (1.0f - tx) * (1.0f - ty) *          tz;
    float w101 =          tx * (1.0f - ty) *          tz;
    float w011 = (1.0f - tx) *          ty *          tz;
    float w111 =          tx *          ty *          tz;

    return w000 * v000 + w100 * v100 + w010 * v010 + w110 * v110
         + w001 * v001 + w101 * v101 + w011 * v011 + w111 * v111;
}

float safe_get(const float* v, const int fs[3], int a, int b, int c) {
    if (a < 0 || a >= fs[0] || b < 0 || b >= fs[1] || c < 0 || c >= fs[2])
        return 0.0f;
    return v[(a * fs[1] + b) * fs[2] + c];
}

void advect_vx(int nx, int ny, int nz,
               const float* vx, const float* vy, const float* vz,
               float* out, float dt)
{
    const int fs_x[3] = {nx + 1, ny,     nz    };
    const int fs_y[3] = {nx,     ny + 1, nz    };
    const int fs_z[3] = {nx,     ny,     nz + 1};

    for (int i = 0; i < fs_x[0]; ++i) {
        for (int j = 0; j < fs_x[1]; ++j) {
            for (int k = 0; k < fs_x[2]; ++k) {
                float vd = vx[(i * ny + j) * nz + k];
                float vo_y = 0.25f * (safe_get(vy, fs_y, i - 1, j,     k)
                                    + safe_get(vy, fs_y, i,     j,     k)
                                    + safe_get(vy, fs_y, i - 1, j + 1, k)
                                    + safe_get(vy, fs_y, i,     j + 1, k));
                float vo_z = 0.25f * (safe_get(vz, fs_z, i - 1, j, k    )
                                    + safe_get(vz, fs_z, i,     j, k    )
                                    + safe_get(vz, fs_z, i - 1, j, k + 1)
                                    + safe_get(vz, fs_z, i,     j, k + 1));

                float bx = std::clamp((float)i - dt * vd,   0.0f, (float)(fs_x[0] - 1));
                float by = std::clamp((float)j - dt * vo_y, 0.0f, (float)(fs_x[1] - 1));
                float bz = std::clamp((float)k - dt * vo_z, 0.0f, (float)(fs_x[2] - 1));

                out[(i * ny + j) * nz + k] = trilinear_sample(vx, fs_x, bx, by, bz);
            }
        }
    }
}

void advect_vy(int nx, int ny, int nz,
               const float* vx, const float* vy, const float* vz,
               float* out, float dt)
{
    const int fs_x[3] = {nx + 1, ny,     nz    };
    const int fs_y[3] = {nx,     ny + 1, nz    };
    const int fs_z[3] = {nx,     ny,     nz + 1};

    for (int i = 0; i < fs_y[0]; ++i) {
        for (int j = 0; j < fs_y[1]; ++j) {
            for (int k = 0; k < fs_y[2]; ++k) {
                float vd = vy[(i * (ny + 1) + j) * nz + k];
                float vo_x = 0.25f * (safe_get(vx, fs_x, i,     j - 1, k)
                                    + safe_get(vx, fs_x, i + 1, j - 1, k)
                                    + safe_get(vx, fs_x, i,     j,     k)
                                    + safe_get(vx, fs_x, i + 1, j,     k));
                float vo_z = 0.25f * (safe_get(vz, fs_z, i, j - 1, k    )
                                    + safe_get(vz, fs_z, i, j,     k    )
                                    + safe_get(vz, fs_z, i, j - 1, k + 1)
                                    + safe_get(vz, fs_z, i, j,     k + 1));

                float bx = std::clamp((float)i - dt * vo_x, 0.0f, (float)(fs_y[0] - 1));
                float by = std::clamp((float)j - dt * vd,   0.0f, (float)(fs_y[1] - 1));
                float bz = std::clamp((float)k - dt * vo_z, 0.0f, (float)(fs_y[2] - 1));

                out[(i * (ny + 1) + j) * nz + k] = trilinear_sample(vy, fs_y, bx, by, bz);
            }
        }
    }
}

void advect_vz(int nx, int ny, int nz,
               const float* vx, const float* vy, const float* vz,
               float* out, float dt)
{
    const int fs_x[3] = {nx + 1, ny,     nz    };
    const int fs_y[3] = {nx,     ny + 1, nz    };
    const int fs_z[3] = {nx,     ny,     nz + 1};

    for (int i = 0; i < fs_z[0]; ++i) {
        for (int j = 0; j < fs_z[1]; ++j) {
            for (int k = 0; k < fs_z[2]; ++k) {
                float vd = vz[(i * ny + j) * (nz + 1) + k];
                float vo_x = 0.25f * (safe_get(vx, fs_x, i,     j, k - 1)
                                    + safe_get(vx, fs_x, i + 1, j, k - 1)
                                    + safe_get(vx, fs_x, i,     j, k    )
                                    + safe_get(vx, fs_x, i + 1, j, k    ));
                float vo_y = 0.25f * (safe_get(vy, fs_y, i, j,     k - 1)
                                    + safe_get(vy, fs_y, i, j + 1, k - 1)
                                    + safe_get(vy, fs_y, i, j,     k    )
                                    + safe_get(vy, fs_y, i, j + 1, k    ));

                float bx = std::clamp((float)i - dt * vo_x, 0.0f, (float)(fs_z[0] - 1));
                float by = std::clamp((float)j - dt * vo_y, 0.0f, (float)(fs_z[1] - 1));
                float bz = std::clamp((float)k - dt * vd,   0.0f, (float)(fs_z[2] - 1));

                out[(i * ny + j) * (nz + 1) + k] = trilinear_sample(vz, fs_z, bx, by, bz);
            }
        }
    }
}

} // anonymous namespace

void compute_scalar_advection(int nx, int ny, int nz,
                              const float* vx, const float* vy, const float* vz,
                              const float* field_in, float* field_out, float dt)
{
    int cs[3] = {nx, ny, nz};

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                float vx_c = 0.5f * (vx[( i      * ny + j) * nz + k]
                                   + vx[((i + 1) * ny + j) * nz + k]);
                float vy_c = 0.5f * (vy[(i * (ny + 1) +  j     ) * nz + k]
                                   + vy[(i * (ny + 1) + (j + 1)) * nz + k]);
                float vz_c = 0.5f * (vz[(i * ny + j) * (nz + 1) +  k     ]
                                   + vz[(i * ny + j) * (nz + 1) + (k + 1)]);

                float bx = std::clamp((float)i - dt * vx_c, 0.0f, (float)(nx - 1));
                float by = std::clamp((float)j - dt * vy_c, 0.0f, (float)(ny - 1));
                float bz = std::clamp((float)k - dt * vz_c, 0.0f, (float)(nz - 1));

                field_out[(i * ny + j) * nz + k] = trilinear_sample(field_in, cs, bx, by, bz);
            }
        }
    }
}

void compute_velocity_advection(int nx, int ny, int nz,
                                float* vx, float* vy, float* vz,
                                float* scratch, float dt)
{
    const int total_vx = (nx + 1) * ny * nz;
    const int total_vy = nx * (ny + 1) * nz;
    const int total_vz = nx * ny * (nz + 1);

    advect_vx(nx, ny, nz, vx, vy, vz, scratch, dt);
    std::copy(scratch, scratch + total_vx, vx);

    advect_vy(nx, ny, nz, vx, vy, vz, scratch, dt);
    std::copy(scratch, scratch + total_vy, vy);

    advect_vz(nx, ny, nz, vx, vy, vz, scratch, dt);
    std::copy(scratch, scratch + total_vz, vz);
}

} // namespace slipstream::k_cpu

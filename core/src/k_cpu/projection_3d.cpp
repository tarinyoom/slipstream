#include "projection_3d.hpp"

#include <algorithm>
#include <cmath>

namespace slipstream::k_cpu_3d {

namespace {

int cell_idx(int ny, int nz, int i, int j, int k) {
    return (i * ny + j) * nz + k;
}

int face_idx_x(int ny, int nz, int i, int j, int k) {
    return (i * ny + j) * nz + k;
}

int face_idx_y(int ny, int nz, int i, int j, int k) {
    return (i * (ny + 1) + j) * nz + k;
}

int face_idx_z(int ny, int nz, int i, int j, int k) {
    return (i * ny + j) * (nz + 1) + k;
}

void flat_to_ijk(int ny, int nz, int flat, int& i, int& j, int& k) {
    k = flat % nz;
    int rest = flat / nz;
    j = rest % ny;
    i = rest / ny;
}

} // anonymous namespace

void compute_red_black_gs(int nx, int ny, int nz, const float* obstacle,
                          const float* rhs, float* pressure,
                          int max_iterations, float tolerance)
{
    const int total = nx * ny * nz;
    const int di[6] = {-1, +1,  0,  0,  0,  0};
    const int dj[6] = { 0,  0, -1, +1,  0,  0};
    const int dk[6] = { 0,  0,  0,  0, -1, +1};

    for (int iter = 0; iter < max_iterations; ++iter) {
        for (int color = 0; color < 2; ++color) {
            for (int c = 0; c < total; ++c) {
                if (obstacle && obstacle[c] != 0.0f) continue;
                int ci, cj, ck;
                flat_to_ijk(ny, nz, c, ci, cj, ck);
                if ((ci + cj + ck) % 2 != color) continue;

                float sum_nbr = 0.0f;
                int   num_nbr = 0;

                for (int n = 0; n < 6; ++n) {
                    int ni = ci + di[n];
                    int nj = cj + dj[n];
                    int nk = ck + dk[n];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue;
                    int nc = cell_idx(ny, nz, ni, nj, nk);
                    if (obstacle && obstacle[nc] != 0.0f) continue;
                    sum_nbr += pressure[nc];
                    ++num_nbr;
                }
                if (num_nbr > 0)
                    pressure[c] = (sum_nbr - rhs[c]) / static_cast<float>(num_nbr);
            }
        }

        if ((iter + 1) % 10 == 0) {
            float residual = 0.0f;
            for (int c = 0; c < total; ++c) {
                if (obstacle && obstacle[c] != 0.0f) continue;
                int ci, cj, ck;
                flat_to_ijk(ny, nz, c, ci, cj, ck);

                float sum_nbr = 0.0f;
                int   num_nbr = 0;
                for (int n = 0; n < 6; ++n) {
                    int ni = ci + di[n];
                    int nj = cj + dj[n];
                    int nk = ck + dk[n];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue;
                    int nc = cell_idx(ny, nz, ni, nj, nk);
                    if (obstacle && obstacle[nc] != 0.0f) continue;
                    sum_nbr += pressure[nc];
                    ++num_nbr;
                }
                float r = std::abs(static_cast<float>(num_nbr) * pressure[c]
                                   - sum_nbr + rhs[c]);
                residual = std::max(residual, r);
            }
            if (residual < tolerance) break;
        }
    }
}

void compute_projection(int nx, int ny, int nz, const float* obstacle,
                        float* vx, float* vy, float* vz,
                        float* pressure, float* rhs_scratch,
                        int max_iterations, float tolerance)
{
    const int total = nx * ny * nz;

    if (obstacle) {
        for (int c = 0; c < total; ++c) {
            if (obstacle[c] == 0.0f) continue;
            int ci, cj, ck;
            flat_to_ijk(ny, nz, c, ci, cj, ck);
            vx[face_idx_x(ny, nz, ci,     cj,     ck    )] = 0.0f;
            vx[face_idx_x(ny, nz, ci + 1, cj,     ck    )] = 0.0f;
            vy[face_idx_y(ny, nz, ci,     cj,     ck    )] = 0.0f;
            vy[face_idx_y(ny, nz, ci,     cj + 1, ck    )] = 0.0f;
            vz[face_idx_z(ny, nz, ci,     cj,     ck    )] = 0.0f;
            vz[face_idx_z(ny, nz, ci,     cj,     ck + 1)] = 0.0f;
        }
    }

    for (int c = 0; c < total; ++c) {
        int ci, cj, ck;
        flat_to_ijk(ny, nz, c, ci, cj, ck);
        if (obstacle && obstacle[c] != 0.0f) {
            rhs_scratch[c] = 0.0f;
            continue;
        }
        rhs_scratch[c] = vx[face_idx_x(ny, nz, ci + 1, cj,     ck    )] - vx[face_idx_x(ny, nz, ci, cj, ck)]
                       + vy[face_idx_y(ny, nz, ci,     cj + 1, ck    )] - vy[face_idx_y(ny, nz, ci, cj, ck)]
                       + vz[face_idx_z(ny, nz, ci,     cj,     ck + 1)] - vz[face_idx_z(ny, nz, ci, cj, ck)];
    }

    compute_red_black_gs(nx, ny, nz, obstacle, rhs_scratch, pressure, max_iterations, tolerance);

    for (int c = 0; c < total; ++c) {
        if (obstacle && obstacle[c] != 0.0f) continue;
        int ci, cj, ck;
        flat_to_ijk(ny, nz, c, ci, cj, ck);
        if (ci > 0) {
            int lc = cell_idx(ny, nz, ci - 1, cj, ck);
            if (!obstacle || obstacle[lc] == 0.0f)
                vx[face_idx_x(ny, nz, ci, cj, ck)] -= pressure[c] - pressure[lc];
        }
        if (cj > 0) {
            int lc = cell_idx(ny, nz, ci, cj - 1, ck);
            if (!obstacle || obstacle[lc] == 0.0f)
                vy[face_idx_y(ny, nz, ci, cj, ck)] -= pressure[c] - pressure[lc];
        }
        if (ck > 0) {
            int lc = cell_idx(ny, nz, ci, cj, ck - 1);
            if (!obstacle || obstacle[lc] == 0.0f)
                vz[face_idx_z(ny, nz, ci, cj, ck)] -= pressure[c] - pressure[lc];
        }
    }
}

} // namespace slipstream::k_cpu_3d

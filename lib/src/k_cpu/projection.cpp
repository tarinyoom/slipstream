#include "projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace slipstream::k_cpu {

namespace {

int cell_idx(int ny, int i, int j) { return i * ny + j; }

int face_idx_x(int ny, int i, int j) { return i * ny + j; }
int face_idx_y(int ny, int i, int j) { return i * (ny + 1) + j; }

void flat_to_ij(int ny, int flat, int& i, int& j) {
    j = flat % ny;
    i = flat / ny;
}

} // anonymous namespace

void compute_red_black_gs(int nx, int ny, const float* obstacle,
                          const float* rhs, float* pressure,
                          int max_iterations, float tolerance)
{
    const int total = nx * ny;

    for (int iter = 0; iter < max_iterations; ++iter) {
        for (int color = 0; color < 2; ++color) {
            for (int c = 0; c < total; ++c) {
                if (obstacle && obstacle[c] != 0.0f) continue;
                int ci, cj;
                flat_to_ij(ny, c, ci, cj);
                if ((ci + cj) % 2 != color) continue;

                float sum_nbr = 0.0f;
                int   num_nbr = 0;

                const int di[4] = {-1, +1,  0,  0};
                const int dj[4] = { 0,  0, -1, +1};
                for (int k = 0; k < 4; ++k) {
                    int ni = ci + di[k];
                    int nj = cj + dj[k];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                    int nc = cell_idx(ny, ni, nj);
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
                int ci, cj;
                flat_to_ij(ny, c, ci, cj);

                float sum_nbr = 0.0f;
                int   num_nbr = 0;
                const int di[4] = {-1, +1,  0,  0};
                const int dj[4] = { 0,  0, -1, +1};
                for (int k = 0; k < 4; ++k) {
                    int ni = ci + di[k];
                    int nj = cj + dj[k];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                    int nc = cell_idx(ny, ni, nj);
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

void compute_projection(int nx, int ny, const float* obstacle,
                        float* vx, float* vy,
                        float* pressure, float* rhs_scratch,
                        int max_iterations, float tolerance)
{
    const int total = nx * ny;

    if (obstacle) {
        for (int c = 0; c < total; ++c) {
            if (obstacle[c] == 0.0f) continue;
            int ci, cj;
            flat_to_ij(ny, c, ci, cj);
            vx[face_idx_x(ny, ci,     cj)] = 0.0f;
            vx[face_idx_x(ny, ci + 1, cj)] = 0.0f;
            vy[face_idx_y(ny, ci, cj    )] = 0.0f;
            vy[face_idx_y(ny, ci, cj + 1)] = 0.0f;
        }
    }

    std::memset(pressure,    0, static_cast<std::size_t>(total) * sizeof(float));
    std::memset(rhs_scratch, 0, static_cast<std::size_t>(total) * sizeof(float));
    for (int c = 0; c < total; ++c) {
        if (obstacle && obstacle[c] != 0.0f) continue;
        int ci, cj;
        flat_to_ij(ny, c, ci, cj);
        rhs_scratch[c] = vx[face_idx_x(ny, ci + 1, cj    )] - vx[face_idx_x(ny, ci, cj)]
                       + vy[face_idx_y(ny, ci,     cj + 1)] - vy[face_idx_y(ny, ci, cj)];
    }

    compute_red_black_gs(nx, ny, obstacle, rhs_scratch, pressure, max_iterations, tolerance);

    for (int c = 0; c < total; ++c) {
        if (obstacle && obstacle[c] != 0.0f) continue;
        int ci, cj;
        flat_to_ij(ny, c, ci, cj);
        if (ci > 0) {
            int lc = cell_idx(ny, ci - 1, cj);
            if (!obstacle || obstacle[lc] == 0.0f)
                vx[face_idx_x(ny, ci, cj)] -= pressure[c] - pressure[lc];
        }
        if (cj > 0) {
            int lc = cell_idx(ny, ci, cj - 1);
            if (!obstacle || obstacle[lc] == 0.0f)
                vy[face_idx_y(ny, ci, cj)] -= pressure[c] - pressure[lc];
        }
    }
}

} // namespace slipstream::k_cpu

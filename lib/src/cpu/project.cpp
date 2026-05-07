#include "project.hpp"
#include "grid.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace slipstream::cpu {

namespace {

void flat_to_ijk(const int* shape, int ndim, int flat, int* ijk) {
    for (int d = ndim - 1; d >= 0; --d) {
        ijk[d] = flat % shape[d];
        flat  /= shape[d];
    }
}

} // anonymous namespace

void red_black_gs(const State& s, const float* rhs, float* pressure,
                  int max_iterations, float tolerance)
{
    const int ndim  = s.n_dims;
    int       total = 1;
    for (int d = 0; d < ndim; ++d) total *= s.dims[d];

    int ijk[4], nbr[4];

    for (int iter = 0; iter < max_iterations; ++iter) {
        for (int color = 0; color < 2; ++color) {
            for (int c = 0; c < total; ++c) {
                if (s.obstacle && s.obstacle[c]) continue;
                flat_to_ijk(s.dims, ndim, c, ijk);

                int parity = 0;
                for (int d = 0; d < ndim; ++d) parity += ijk[d];
                if (parity % 2 != color) continue;

                float sum_nbr = 0.0f;
                int   num_nbr = 0;
                for (int d = 0; d < ndim; ++d) {
                    for (int delta : {-1, +1}) {
                        std::copy(ijk, ijk + ndim, nbr);
                        nbr[d] += delta;
                        if (nbr[d] < 0 || nbr[d] >= s.dims[d]) continue;
                        int nc = cell_idx(s.dims, ndim, nbr);
                        if (s.obstacle && s.obstacle[nc]) continue;
                        sum_nbr += pressure[nc];
                        ++num_nbr;
                    }
                }
                if (num_nbr > 0)
                    pressure[c] = (sum_nbr - rhs[c]) / static_cast<float>(num_nbr);
            }
        }

        if ((iter + 1) % 10 == 0) {
            float residual = 0.0f;
            for (int c = 0; c < total; ++c) {
                if (s.obstacle && s.obstacle[c]) continue;
                flat_to_ijk(s.dims, ndim, c, ijk);

                float sum_nbr = 0.0f;
                int   num_nbr = 0;
                for (int d = 0; d < ndim; ++d) {
                    for (int delta : {-1, +1}) {
                        std::copy(ijk, ijk + ndim, nbr);
                        nbr[d] += delta;
                        if (nbr[d] < 0 || nbr[d] >= s.dims[d]) continue;
                        int nc = cell_idx(s.dims, ndim, nbr);
                        if (s.obstacle && s.obstacle[nc]) continue;
                        sum_nbr += pressure[nc];
                        ++num_nbr;
                    }
                }
                float r = std::abs(static_cast<float>(num_nbr) * pressure[c]
                                   - sum_nbr + rhs[c]);
                residual = std::max(residual, r);
            }
            if (residual < tolerance) break;
        }
    }
}

void project(const State& s, float* pressure, int max_iterations, float tolerance)
{
    const int ndim  = s.n_dims;
    int       total = 1;
    for (int d = 0; d < ndim; ++d) total *= s.dims[d];

    float* vel[2] = { s.v, s.v + (s.dims[0] + 1) * s.dims[1] };

    int ijk[4], tmp[4];

    if (s.obstacle) {
        for (int c = 0; c < total; ++c) {
            if (!s.obstacle[c]) continue;
            flat_to_ijk(s.dims, ndim, c, ijk);
            for (int d = 0; d < ndim; ++d) {
                std::copy(ijk, ijk + ndim, tmp);
                vel[d][face_idx(s.dims, ndim, d, tmp)] = 0.0f;
                tmp[d] = ijk[d] + 1;
                vel[d][face_idx(s.dims, ndim, d, tmp)] = 0.0f;
            }
        }
    }

    std::fill(pressure, pressure + total, 0.0f);
    std::vector<float> rhs(total, 0.0f);
    {
        int face_hi[4];
        for (int c = 0; c < total; ++c) {
            if (s.obstacle && s.obstacle[c]) continue;
            flat_to_ijk(s.dims, ndim, c, ijk);
            float div = 0.0f;
            for (int d = 0; d < ndim; ++d) {
                std::copy(ijk, ijk + ndim, face_hi);
                face_hi[d] = ijk[d] + 1;
                div += vel[d][face_idx(s.dims, ndim, d, face_hi)]
                     - vel[d][face_idx(s.dims, ndim, d, ijk)];
            }
            rhs[c] = div;
        }
    }

    red_black_gs(s, rhs.data(), pressure, max_iterations, tolerance);

    {
        int lo_ijk[4];
        for (int c = 0; c < total; ++c) {
            if (s.obstacle && s.obstacle[c]) continue;
            flat_to_ijk(s.dims, ndim, c, ijk);
            for (int d = 0; d < ndim; ++d) {
                if (ijk[d] == 0) continue;
                std::copy(ijk, ijk + ndim, lo_ijk);
                lo_ijk[d] -= 1;
                int lo_c = cell_idx(s.dims, ndim, lo_ijk);
                if (s.obstacle && s.obstacle[lo_c]) continue;
                vel[d][face_idx(s.dims, ndim, d, ijk)] -=
                    pressure[c] - pressure[lo_c];
            }
        }
    }
}

} // namespace slipstream::cpu

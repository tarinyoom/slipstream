#include "advect.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace slipstream::cpu {

namespace {

float bilinear_sample(std::span<const float> v, const int fs[2], float px, float py) {
    int i0 = (int)std::floor(px);
    int j0 = (int)std::floor(py);
    int i1 = i0 + 1;
    int j1 = j0 + 1;
    float tx = px - (float)i0;
    float ty = py - (float)j0;

    i0 = std::clamp(i0, 0, fs[0] - 1);
    i1 = std::clamp(i1, 0, fs[0] - 1);
    j0 = std::clamp(j0, 0, fs[1] - 1);
    j1 = std::clamp(j1, 0, fs[1] - 1);

    float v00 = v[i0 * fs[1] + j0];
    float v10 = v[i1 * fs[1] + j0];
    float v01 = v[i0 * fs[1] + j1];
    float v11 = v[i1 * fs[1] + j1];

    return (1.0f - tx) * (1.0f - ty) * v00
         +          tx * (1.0f - ty) * v10
         + (1.0f - tx) *          ty * v01
         +          tx *          ty * v11;
}

float safe_get(std::span<const float> v, const int fs[2], int a, int b) {
    if (a < 0 || a >= fs[0] || b < 0 || b >= fs[1]) return 0.0f;
    return v[a * fs[1] + b];
}

} // anonymous namespace

void advect_velocity(std::span<const int>           shape,
                     std::vector<std::span<float>>& velocity,
                     std::span<float>               scratch,
                     float                          dt)
{
    const int Nx = shape[0];
    const int Ny = shape[1];

    for (int dim = 0; dim < 2; ++dim) {
        const int other = 1 - dim;

        int fs[2]       = { dim   == 0 ? Nx + 1 : Nx,
                            dim   == 1 ? Ny + 1 : Ny };
        int fs_other[2] = { other == 0 ? Nx + 1 : Nx,
                            other == 1 ? Ny + 1 : Ny };

        std::span<const float> vd    (velocity[dim  ].data(), velocity[dim  ].size());
        std::span<const float> vother(velocity[other].data(), velocity[other].size());

        const int total = fs[0] * fs[1];

        for (int i = 0; i < fs[0]; ++i) {
            for (int j = 0; j < fs[1]; ++j) {
                float vd_here;
                float vo_here;

                vd_here = vd[i * fs[1] + j];

                if (dim == 0) {
                    vo_here = 0.25f * (safe_get(vother, fs_other, i - 1, j    )
                                     + safe_get(vother, fs_other, i,     j    )
                                     + safe_get(vother, fs_other, i - 1, j + 1)
                                     + safe_get(vother, fs_other, i,     j + 1));
                } else {
                    vo_here = 0.25f * (safe_get(vother, fs_other, i,     j - 1)
                                     + safe_get(vother, fs_other, i + 1, j - 1)
                                     + safe_get(vother, fs_other, i,     j    )
                                     + safe_get(vother, fs_other, i + 1, j    ));
                }

                float bx = (float)i - dt * (dim == 0 ? vd_here : vo_here);
                float by = (float)j - dt * (dim == 1 ? vd_here : vo_here);

                bx = std::clamp(bx, 0.0f, (float)(fs[0] - 1));
                by = std::clamp(by, 0.0f, (float)(fs[1] - 1));

                scratch[i * fs[1] + j] = bilinear_sample(vd, fs, bx, by);
            }
        }

        std::copy(scratch.begin(), scratch.begin() + total, velocity[dim].begin());
    }
}

} // namespace slipstream::cpu

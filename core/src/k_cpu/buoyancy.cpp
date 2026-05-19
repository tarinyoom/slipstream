#include "buoyancy.hpp"

namespace slipstream::k_cpu {

void compute_buoyancy(int nx, int ny, int nz, float buoyancy, float dt,
                      const float* temperature, float* vx)
{
    for (int i = 1; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k) {
                float t_avg = 0.5f * (temperature[((i - 1) * ny + j) * nz + k]
                                    + temperature[( i      * ny + j) * nz + k]);
                vx[(i * ny + j) * nz + k] += buoyancy * t_avg * dt;
            }
}

} // namespace slipstream::k_cpu

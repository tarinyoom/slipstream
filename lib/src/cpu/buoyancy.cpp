#include "buoyancy.hpp"

namespace slipstream::cpu {

void apply_buoyancy(int nx, int ny, float buoyancy, float dt,
                    const float* temperature, float* vx)
{
    for (int i = 1; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            float t_avg = 0.5f * (temperature[(i - 1) * ny + j]
                                + temperature[ i      * ny + j]);
            vx[i * ny + j] += buoyancy * t_avg * dt;
        }
}

} // namespace slipstream::cpu

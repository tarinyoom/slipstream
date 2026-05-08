#include "compute_cooling.hpp"

namespace slipstream::cpu {

void compute_cooling(int total, float cooling, float dt, float* temperature) {
    float factor = 1.0f - cooling * dt;
    for (int c = 0; c < total; ++c)
        temperature[c] *= factor;
}

} // namespace slipstream::cpu

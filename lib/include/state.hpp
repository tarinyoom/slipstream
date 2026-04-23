#pragma once

#include <span>
#include <vector>

namespace slipstream {

struct State {
    std::vector<int> shape;
    int ndim  = 0;
    int total = 0;

    std::span<float>              density;
    std::vector<std::span<float>> velocity;
    std::span<float>              temperature;
    std::span<const bool>  obstacle;
    std::span<const bool>  emitter_masks;
    std::span<const float> emitter_densities;
    std::span<const float> emitter_temperatures;

    float viscosity = 0.0f;
    float buoyancy  = 0.0f;
    float vorticity = 0.0f;

    explicit State(const int* shape, int ndim);

    void validate() const;
};

} // namespace slipstream

#pragma once

#include <cstddef>
#include <span>

namespace slipstream {

struct State {
    int nx, ny;
    int n_emitters;

    float* density;
    float* velocity;
    float* temperature;
    float* obstacle;
    float* emitter_masks;
    float* emitter_densities;
    float* emitter_temperatures;

    float* pressure;
    float* scratch;

    float viscosity, buoyancy, cooling, vorticity;
};

std::size_t required_state_bytes(std::span<const int> dims,
                                 int n_emitters,
                                 bool allocate_solver_workspace);

void init_state(State& state,
                void* buf, std::size_t len,
                std::span<const int> dims,
                int n_emitters,
                bool allocate_solver_workspace);

} // namespace slipstream

#include "solver.hpp"
#include "cpu/advect.hpp"
#include "cpu/project.hpp"

#include <stdexcept>
#include <vector>

namespace slipstream {

struct Solver::Impl {
    State*  state   = nullptr;
    Backend backend = Backend::CPU;

    std::vector<float> pressure;
    std::vector<float> scratch;
    int   max_iterations = 100;
    float tolerance      = 1e-3f;
};

Solver::Solver(State& state, Backend backend) {
    if (backend == Backend::GPU)
        throw std::runtime_error("UNSUPPORTED_BACKEND");

    state.validate();

    impl_ = std::make_unique<Impl>();
    impl_->state   = &state;
    impl_->backend = backend;
    impl_->pressure.resize(state.total, 0.0f);

    int max_faces = 1;
    for (int d = 0; d < state.ndim; ++d) {
        int faces = 1;
        for (int e = 0; e < state.ndim; ++e)
            faces *= (d == e) ? state.shape[e] + 1 : state.shape[e];
        max_faces = std::max(max_faces, faces);
    }
    impl_->scratch.resize(max_faces, 0.0f);
}

Solver::~Solver() = default;

void Solver::step(float dt) {
    State& state = *impl_->state;
    auto density = state.density;
    if (density.empty())
        throw std::runtime_error("step: density field not set");

    if (!state.emitter_masks.empty()) {
        int n = (int)state.emitter_densities.size();
        for (int e = 0; e < n; ++e) {
            for (int c = 0; c < state.total; ++c) {
                if (state.emitter_masks[e * state.total + c]) {
                    density[c] = state.emitter_densities[e];
                    if (!state.temperature.empty() && !state.emitter_temperatures.empty())
                        state.temperature[c] = state.emitter_temperatures[e];
                }
            }
        }
    }

    if (!state.velocity.empty()) {
        cpu::advect_scalar(state.shape, density, impl_->scratch, state.velocity, dt);
        std::copy(impl_->scratch.begin(), impl_->scratch.begin() + state.total, density.begin());

        if (!state.temperature.empty()) {
            cpu::advect_scalar(state.shape, state.temperature, impl_->scratch, state.velocity, dt);
            std::copy(impl_->scratch.begin(), impl_->scratch.begin() + state.total,
                      state.temperature.begin());
        }

        cpu::advect_velocity(state.shape, state.velocity, impl_->scratch, dt);
        cpu::project(state.shape, state.velocity, state.obstacle,
                     impl_->pressure, impl_->max_iterations, impl_->tolerance);
    }
}

} // namespace slipstream

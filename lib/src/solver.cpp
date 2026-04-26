#include "solver.hpp"
#include "cpu/project.hpp"

#include <stdexcept>
#include <vector>

namespace slipstream {

struct Solver::Impl {
    State*  state   = nullptr;
    Backend backend = Backend::CPU;

    std::vector<float> pressure;
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
}

Solver::~Solver() = default;

void Solver::step(float dt) {
    auto density = impl_->state->density;
    if (density.empty())
        throw std::runtime_error("step: density field not set");

    for (float& d : density)
        d += dt;

    if (!impl_->state->velocity.empty())
        cpu::project(impl_->state->shape,
                     impl_->state->velocity,
                     impl_->state->obstacle,
                     impl_->pressure,
                     impl_->max_iterations,
                     impl_->tolerance);
}

} // namespace slipstream

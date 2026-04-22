#include "solver.hpp"

#include <stdexcept>

namespace slipstream {

struct Solver::Impl {
    State*  state   = nullptr;
    Backend backend = Backend::CPU;
};

Solver::Solver(State& state, Backend backend) {
    if (backend == Backend::GPU)
        throw std::runtime_error("UNSUPPORTED_BACKEND");

    impl_ = std::make_unique<Impl>();
    impl_->state   = &state;
    impl_->backend = backend;
}

Solver::~Solver() = default;

void Solver::step(float dt) {
    auto density = impl_->state->density;
    if (density.empty())
        throw std::runtime_error("step: density field not set");

    for (float& d : density)
        d += dt;
}

} // namespace slipstream

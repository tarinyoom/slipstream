#include "solver.hpp"

#include <stdexcept>

namespace slipstream {

struct Solver::Impl {
    Backend backend = Backend::CPU;
};

Solver::Solver(Backend backend) {
    if (backend == Backend::GPU)
        throw std::runtime_error("UNSUPPORTED_BACKEND");

    impl_ = std::make_unique<Impl>();
    impl_->backend = backend;
}

Solver::~Solver() = default;

void Solver::step(State& state, float dt) {
    auto density = state.density();
    if (density.empty())
        throw std::runtime_error("step: density field not set");

    for (float& d : density)
        d += dt;
}

} // namespace slipstream

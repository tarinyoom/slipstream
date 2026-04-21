#pragma once

#include "state.hpp"
#include <memory>

namespace slipstream {

enum class Backend { CPU, GPU };

class Solver {
public:
    explicit Solver(Backend backend = Backend::CPU);
    ~Solver();

    Solver(const Solver&)            = delete;
    Solver& operator=(const Solver&) = delete;

    void step(State& state, float dt);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace slipstream

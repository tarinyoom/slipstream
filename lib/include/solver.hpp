#pragma once

#include "state.hpp"
#include <memory>

namespace slipstream {

enum class Backend { CPU, GPU };

class Solver {
public:
    Solver(State& state, Backend backend = Backend::CPU);
    ~Solver();

    Solver(const Solver&)            = delete;
    Solver& operator=(const Solver&) = delete;

    void step(float dt);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace slipstream

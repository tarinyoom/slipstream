#include "state.hpp"

#include <stdexcept>
#include <string>

namespace slipstream {

State::State(const int* shape, int ndim) {
    if (!shape || ndim <= 0)
        throw std::invalid_argument("State: invalid shape");
    this->ndim = ndim;
    this->shape.assign(shape, shape + ndim);
    this->total = 1;
    for (int i = 0; i < ndim; i++)
        this->total *= shape[i];
}

void State::validate() const {
    auto check = [](auto sp, size_t expected, const char* name) {
        if (!sp.empty() && sp.size() != expected)
            throw std::invalid_argument(
                std::string("State: ") + name + " has wrong size");
    };

    check(density,     (size_t)total, "density");
    check(temperature, (size_t)total, "temperature");
    check(obstacle,    (size_t)total, "obstacle");

    if (!velocity.empty()) {
        if (velocity.size() != (size_t)ndim)
            throw std::invalid_argument("State: velocity must have ndim components");

        for (int i = 0; i < ndim; i++) {
            size_t expected = 1;
            for (int j = 0; j < ndim; j++)
                expected *= (j == i) ? (size_t)(shape[j] + 1) : (size_t)shape[j];
            check(velocity[i], expected, ("velocity[" + std::to_string(i) + "]").c_str());
        }
    }
}

} // namespace slipstream

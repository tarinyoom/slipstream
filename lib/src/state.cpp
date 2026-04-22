#include "state.hpp"

#include <stdexcept>

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

} // namespace slipstream

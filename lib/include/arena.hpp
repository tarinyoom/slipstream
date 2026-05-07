#pragma once

#include "state.hpp"
#include <optional>

namespace slipstream {

enum class Backend { CPU, GPU };

struct CalculationArena {
    char*                       buf;
    Backend                     backend;
    PersistentState             state;
    std::optional<ScratchState> scratch;

    CalculationArena(Backend b, int n_dims, const int* dims,
                     int n_emitters, bool allocate_scratch);
    ~CalculationArena();
    CalculationArena(const CalculationArena&)            = delete;
    CalculationArena& operator=(const CalculationArena&) = delete;
};

void copy(const CalculationArena& src, CalculationArena& dst);

} // namespace slipstream

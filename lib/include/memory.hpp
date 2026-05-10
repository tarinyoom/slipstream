#pragma once

#include "state.hpp"
#include <cstddef>

namespace slipstream {

void* host_alloc(std::size_t sz);
void  host_free (void* p);

#ifdef SLIPSTREAM_HAS_CUDA

void* device_alloc(std::size_t sz);
void  device_free (void* p);

enum class Field {
    Density,
    Velocity,
};

void upload  (const State& host,   State& device);
void download(const State& device, State& host);
void download_field(const State& device, State& host, Field field);

#endif

} // namespace slipstream

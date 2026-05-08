#pragma once
#include <cstddef>

namespace slipstream::gpu {

void memcpy_h2d(void* dst, const void* src, std::size_t sz);
void memcpy_d2h(void* dst, const void* src, std::size_t sz);
void memcpy_d2d(void* dst, const void* src, std::size_t sz);

} // namespace slipstream::gpu

#pragma once
#include <cstddef>

namespace slipstream::gpu {

char* gpu_alloc(std::size_t sz);
void  gpu_free(char* p);

} // namespace slipstream::gpu

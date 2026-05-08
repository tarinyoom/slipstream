#include "transfer.hpp"

#include <cuda_runtime_api.h>
#include <stdexcept>

namespace slipstream::gpu {

void memcpy_h2d(void* dst, const void* src, std::size_t sz) {
    if (cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy H2D failed");
}

void memcpy_d2h(void* dst, const void* src, std::size_t sz) {
    if (cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy D2H failed");
}

void memcpy_d2d(void* dst, const void* src, std::size_t sz) {
    if (cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy D2D failed");
}

} // namespace slipstream::gpu

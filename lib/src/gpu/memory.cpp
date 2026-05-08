#include "memory.hpp"

#include <cuda_runtime_api.h>
#include <stdexcept>

namespace slipstream::gpu {

char* gpu_alloc(std::size_t sz) {
    void* p = nullptr;
    if (cudaMalloc(&p, sz) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed");
    cudaMemset(p, 0, sz);
    return static_cast<char*>(p);
}

void gpu_free(char* p) {
    cudaFree(p);
}

} // namespace slipstream::gpu

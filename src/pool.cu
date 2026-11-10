// CudaBackend, out-of-line: the only translation unit in the repository that requires nvcc. There is
// no pool logic here, only driver glue.

#include <poolkit/pool.cuh>

#include <cuda_runtime.h>

#include <string>

namespace poolkit {

void* CudaBackend::allocate_raw(std::size_t bytes) {
    void* ptr = nullptr;
    const cudaError_t status = cudaMalloc(&ptr, bytes);
    if (status == cudaSuccess) {
        return ptr;
    }
    if (status == cudaErrorMemoryAllocation) {
        // OOM is reported as nullptr, not by throwing, so the pool's message keeps the request, class
        // and resident bytes.
        return nullptr;
    }
    throw PoolError("poolkit: cudaMalloc failed: " + std::string(cudaGetErrorString(status)),
                    status);
}

void CudaBackend::free_raw(void* ptr) noexcept {
    if (ptr != nullptr) {
        (void)cudaFree(ptr);
    }
}

void* CudaBackend::record_event(StreamHandle stream) {
    cudaEvent_t event = nullptr;
    cudaError_t status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (status != cudaSuccess) {
        return nullptr;
    }
    status = cudaEventRecord(event, stream);
    if (status != cudaSuccess) {
        (void)cudaEventDestroy(event);
        return nullptr;
    }
    return static_cast<void*>(event);
}

bool CudaBackend::query(void* event) {
    if (event == nullptr) {
        return true;
    }
    const cudaError_t status = cudaEventQuery(static_cast<cudaEvent_t>(event));
    if (status == cudaSuccess) {
        return true;
    }
    if (status == cudaErrorNotReady) {
        return false;
    }
    throw PoolError("poolkit: cudaEventQuery failed: " +
                        std::string(cudaGetErrorString(status)),
                    status);
}

void CudaBackend::destroy_event(void* event) noexcept {
    if (event != nullptr) {
        (void)cudaEventDestroy(static_cast<cudaEvent_t>(event));
    }
}

}  // namespace poolkit

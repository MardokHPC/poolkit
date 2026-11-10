#pragma once

// The CUDA glue: CudaBackend, the one place in the library that names cudaMalloc / cudaFree /
// cudaEvent*. Included only by src/pool.cu and the device binaries, never by the core headers, which
// stay CUDA-free. No pool logic lives here.

#include <cuda_runtime.h>

#include <poolkit/backend.hpp>
#include <poolkit/config.hpp>
#include <poolkit/pool_error.hpp>

#include <cstddef>

namespace poolkit {

class CudaBackend final : public Backend {
public:
    explicit CudaBackend(std::size_t resident_step = 0) noexcept : resident_step_(resident_step) {}

    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    // OOM -> nullptr (the pool turns it into PoolError). Any other error is thrown: a driver failure
    // that is not OOM must surface, not be mistaken for "no memory" and retried.
    void* allocate_raw(std::size_t bytes) override;

    void free_raw(void* ptr) noexcept override;

    std::size_t resident_step() const noexcept override { return resident_step_; }
    void set_resident_step(std::size_t step) noexcept { resident_step_ = step; }

    // cudaEventDisableTiming: the pool asks whether an event is done, never how long it took. nullptr
    // on failure, so the quarantine path takes over.
    void* record_event(StreamHandle stream) override;

    // cudaErrorNotReady -> false; any other error -> exception. Mapping every error onto "ready" would
    // release memory a stream may still be writing to.
    bool query(void* event) override;

    void destroy_event(void* event) noexcept override;

private:
    std::size_t resident_step_ = 0;
};

}  // namespace poolkit

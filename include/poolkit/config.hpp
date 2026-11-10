#pragma once

// Compatibility layer between plain C++17 (host-only CI, no toolkit) and CUDA host/device code.
// Nothing here includes a CUDA header: that is what keeps the allocator logic testable in CI on
// GPU-less runners.

#define POOLKIT_VERSION_MAJOR 1
#define POOLKIT_VERSION_MINOR 0
#define POOLKIT_VERSION_PATCH 0
#define POOLKIT_VERSION_STRING "1.0.0"

#if defined(__CUDACC__)
#define POOLKIT_HD __host__ __device__
#else
#define POOLKIT_HD
#endif

#if defined(__cplusplus) && __cplusplus >= 201703L
#define POOLKIT_NODISCARD [[nodiscard]]
#else
#define POOLKIT_NODISCARD
#endif

#if defined(__CUDACC__)
#define POOLKIT_FORCE_INLINE __forceinline__
#elif defined(_MSC_VER)
#define POOLKIT_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define POOLKIT_FORCE_INLINE inline __attribute__((always_inline))
#else
#define POOLKIT_FORCE_INLINE inline
#endif

// Gates every mention of cudaError_t/cudaStream_t in the public headers. 0/1 rather than
// defined/undefined so it can be used with #if.
#if defined(__CUDACC__) || defined(POOLKIT_USE_CUDA_RUNTIME)
#define POOLKIT_HAS_CUDA_RUNTIME 1
#else
#define POOLKIT_HAS_CUDA_RUNTIME 0
#endif

namespace poolkit {

// The one stream type the core headers may name: backend.hpp, device_ptr.hpp, pool.hpp and
// metrics.hpp must stay CUDA-free. With the runtime StreamHandle *is* cudaStream_t; without it, an
// opaque void* whose null value means "the default stream".
#if POOLKIT_HAS_CUDA_RUNTIME
#if !defined(__CUDACC__)
// Opt-in *and* a promise: the runtime headers are on the include path. nvcc injects them itself.
#include <cuda_runtime.h>
#endif
using StreamHandle = cudaStream_t;
#else
using StreamHandle = void*;
#endif

}  // namespace poolkit

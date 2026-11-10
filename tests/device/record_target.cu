// The workload poolkit_record is tested against: three cudaMalloc of known sizes around a kernel,
// their frees, and one allocation below the default --min-size so the filtering path has something
// to filter. Built twice from this file: a standalone executable and a shared library.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;

constexpr std::size_t kSizes[] = {4 * kMiB, 64 * 1024, 1 * kMiB};

constexpr std::size_t kTinySize = 1024;

__global__ void touch(float* data, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        data[index] = static_cast<float>(index) * 0.5f;
    }
}

bool check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "record_target: CUDA error in %s: %s\n", what,
                     cudaGetErrorString(status));
        return false;
    }
    return true;
}

}  // namespace

// argv is whatever followed `--`: an optional repeat count (default 1), and "async"/"fail" to drive
// cudaMallocAsync or to return 7 after one cycle.
#if defined(_WIN32) && defined(POOLKIT_RECORD_TARGET_SHARED)
#define POOLKIT_WORKLOAD_API extern "C" __declspec(dllexport)
#else
#define POOLKIT_WORKLOAD_API extern "C"
#endif

POOLKIT_WORKLOAD_API int poolkit_record_workload(int argc, char** argv) {
    // Exit 77 = "cannot run here": the standalone build is launched by ctest through the injection
    // shim, and a GPU-less runner must skip, not fail.
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::fprintf(stderr, "record_target: no CUDA device (exit 77)\n");
            return 77;
        }
    }

    int repeat = 1;
    if (argc > 0 && argv != nullptr && argv[0] != nullptr) {
        repeat = std::atoi(argv[0]);
        if (repeat < 1) {
            repeat = 1;
        }
    }
    // "async" drives the same sizes through cudaMallocAsync: CUPTI must report the driver's pool as
    // ordinary alloc/free.
    const bool use_async = argc > 1 && argv != nullptr && argv[1] != nullptr &&
                           std::strcmp(argv[1], "async") == 0;
    // "fail" returns 7 after one cycle: the recorder still writes the trace and returns it.
    const bool fail_after_cycle = argc > 1 && argv != nullptr && argv[1] != nullptr &&
                                  std::strcmp(argv[1], "fail") == 0;

    cudaStream_t stream = nullptr;
    if (use_async && !check(cudaStreamCreate(&stream), "cudaStreamCreate")) {
        return 1;
    }

    for (int cycle = 0; cycle < repeat; ++cycle) {
        void* blocks[3] = {nullptr, nullptr, nullptr};
        for (int i = 0; i < 3; ++i) {
            const cudaError_t status = use_async ? cudaMallocAsync(&blocks[i], kSizes[i], stream)
                                                 : cudaMalloc(&blocks[i], kSizes[i]);
            if (!check(status, use_async ? "cudaMallocAsync" : "cudaMalloc")) {
                return 1;
            }
        }

        touch<<<64, 256, 0, stream>>>(static_cast<float*>(blocks[0]), 64 * 256);
        if (!check(cudaGetLastError(), "kernel launch")) {
            return 1;
        }
        if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
            return 1;
        }

        void* tiny = nullptr;
        const cudaError_t tiny_status =
            use_async ? cudaMallocAsync(&tiny, kTinySize, stream) : cudaMalloc(&tiny, kTinySize);
        if (!check(tiny_status, "cudaMalloc(tiny)")) {
            return 1;
        }
        const cudaError_t tiny_free =
            use_async ? cudaFreeAsync(tiny, stream) : cudaFree(tiny);
        if (!check(tiny_free, "cudaFree(tiny)")) {
            return 1;
        }

        for (int i = 0; i < 3; ++i) {
            const cudaError_t status =
                use_async ? cudaFreeAsync(blocks[i], stream) : cudaFree(blocks[i]);
            if (!check(status, use_async ? "cudaFreeAsync" : "cudaFree")) {
                return 1;
            }
        }
        if (use_async && !check(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
            return 1;
        }
    }
    if (stream != nullptr) {
        (void)cudaStreamDestroy(stream);
    }
    if (fail_after_cycle) {
        std::printf("record_target: returning 7 on purpose (fail-after-cycle)\n");
        return 7;
    }
    std::printf("record_target: %d cycles%s, 4 allocations x %zu + %zu + %zu B tracked\n", repeat,
                use_async ? " (cudaMallocAsync)" : "", kSizes[0], kSizes[1], kSizes[2]);
    return 0;
}

#if defined(POOLKIT_RECORD_TARGET_STANDALONE)
int main(int argc, char** argv) {
    return poolkit_record_workload(argc > 1 ? argc - 1 : 0, argc > 1 ? argv + 1 : nullptr);
}
#endif

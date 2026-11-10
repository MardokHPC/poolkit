// M5 spike: can a trace recorder built on CUPTI see the pool's driver traffic?
//
// This is *not* part of the library. It is a throwaway experiment (timeboxed to
// two hours) that answers one question with evidence instead of confidence:
// does CUPTI's activity API, on Windows + CUDA 12.6 + sm_75, deliver
// CUPTI_ACTIVITY_KIND_MEMORY2 records with size, address and timestamp for every
// cudaMalloc / cudaFree?
//
// It subscribes before any CUDA call, enables MEMORY2 only, then runs a tiny
// program (cudaMalloc -> kernel -> cudaFree, twice with different sizes) and
// prints every memory record it receives. Exit code 0 means "the recorder is
// feasible": records arrived, both allocation and release were seen, and the
// byte counts match what the program asked for.
//
// Build (off by default):
//   cmake -S . -B build-spike -G Ninja -DPOOLKIT_WITH_CUDA=ON ^
//         -DCMAKE_CUDA_ARCHITECTURES=75 -DPOOLKIT_BUILD_SPIKE=ON
//   cmake --build build-spike --target poolkit_spike_cupti_memory
// Run (the CUPTI DLL is not copied next to the exe):
//   set PATH=C:\...\CUDA\v12.6\extras\CUPTI\lib64;%PATH%

// CUPTI's activity records are declared with an explicit alignment and MSVC
// reports C4324 ("structure was padded due to alignment specifier") inside the
// header itself. /W4 /WX is right for our code, not for a vendor header, so the
// warning is silenced around the include and nowhere else.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
#include <cupti.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace {

constexpr std::size_t kBufferSize = 8 * 1024 * 1024;

std::size_t g_records = 0;
std::size_t g_allocations = 0;
std::size_t g_releases = 0;
std::size_t g_allocated_bytes = 0;
std::size_t g_released_bytes = 0;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

void check_cupti(CUptiResult status, const char* what) {
    if (status != CUPTI_SUCCESS) {
        const char* text = nullptr;
        cuptiGetResultString(status, &text);
        std::fprintf(stderr, "CUPTI error in %s: %s\n", what, text != nullptr ? text : "unknown");
        std::exit(3);
    }
}

void CUPTIAPI on_api_callback(void* /*userdata*/, CUpti_CallbackDomain /*domain*/,
                              CUpti_CallbackId /*callback_id*/, const void* /*data*/) {}

// A record sink. CUPTI hands whole buffers to the callback; every buffer is
// walked with cuptiActivityGetNextRecord and returned to the pool of buffers by
// freeing it (the simplest legal lifetime for a spike).
void CUPTIAPI on_buffer_completed(CUcontext /*context*/, uint32_t /*stream_id*/, uint8_t* buffer,
                                  size_t /*size*/, size_t valid_size) {
    CUpti_Activity* record = nullptr;
    for (;;) {
        const CUptiResult status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
        if (status != CUPTI_SUCCESS) {
            break;
        }
        if (record->kind != CUPTI_ACTIVITY_KIND_MEMORY2) {
            continue;
        }
        const auto* memory = reinterpret_cast<const CUpti_ActivityMemory3*>(record);
        ++g_records;
        const char* operation = "other";
        if (memory->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION) {
            operation = "alloc";
            ++g_allocations;
            g_allocated_bytes += static_cast<std::size_t>(memory->bytes);
        } else if (memory->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE) {
            operation = "free";
            ++g_releases;
            g_released_bytes += static_cast<std::size_t>(memory->bytes);
        }
        std::printf("MEMORY2 op=%-5s address=0x%llx bytes=%-8llu timestamp_ns=%-12llu "
                    "device=%u context=%u\n",
                    operation, static_cast<unsigned long long>(memory->address),
                    static_cast<unsigned long long>(memory->bytes),
                    static_cast<unsigned long long>(memory->timestamp),
                    memory->deviceId, memory->contextId);
    }
    std::free(buffer);
}

void CUPTIAPI on_buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records) {
    *buffer = static_cast<uint8_t*>(std::malloc(kBufferSize));
    if (*buffer == nullptr) {
        std::fprintf(stderr, "out of host memory for a CUPTI buffer\n");
        std::exit(4);
    }
    *size = kBufferSize;
    *max_records = 0;
}

__global__ void touch(float* data, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        data[index] = static_cast<float>(index);
    }
}

}  // namespace

int main() {
    std::printf("=== CUPTI MEMORY2 spike ===\n");
    std::printf("CUPTI header version: %d\n", CUPTI_API_VERSION);

    CUpti_SubscriberHandle subscriber{};
    check_cupti(cuptiSubscribe(&subscriber, on_api_callback, nullptr), "cuptiSubscribe");
    check_cupti(cuptiActivityRegisterCallbacks(on_buffer_requested, on_buffer_completed),
                "cuptiActivityRegisterCallbacks");
    check_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2),
                "cuptiActivityEnable(MEMORY2)");

    // The workload a recorder has to see: a 1 MiB block used by a kernel, then a
    // 4 KiB block, both freed. Sizes and addresses are checked afterwards.
    void* large = nullptr;
    check(cudaMalloc(&large, std::size_t{1} << 20), "cudaMalloc(1 MiB)");
    touch<<<1, 256>>>(static_cast<float*>(large), 256);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(cudaFree(large), "cudaFree(1 MiB)");

    void* small = nullptr;
    check(cudaMalloc(&small, 4096), "cudaMalloc(4 KiB)");
    check(cudaFree(small), "cudaFree(4 KiB)");

    // Everything has to be delivered before the process exits.
    check_cupti(cuptiActivityFlushAll(0), "cuptiActivityFlushAll");
    check_cupti(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMORY2),
                "cuptiActivityDisable(MEMORY2)");
    check_cupti(cuptiUnsubscribe(subscriber), "cuptiUnsubscribe");

    std::printf("records=%zu allocations=%zu releases=%zu allocated=%zu released=%zu\n", g_records,
                g_allocations, g_releases, g_allocated_bytes, g_released_bytes);

    const bool saw_both_sizes = g_allocations >= 2 && g_releases >= 2;
    const bool saw_our_bytes = g_allocated_bytes >= (std::size_t{1} << 20) + 4096 &&
                               g_released_bytes >= (std::size_t{1} << 20) + 4096;
    if (!saw_both_sizes || !saw_our_bytes) {
        std::fprintf(stderr, "SPIKE FAILED: no size/address/timestamp records for the workload\n");
        return 1;
    }
    std::printf("SPIKE OK: MEMORY2 delivers size, address and timestamp per operation\n");
    return 0;
}

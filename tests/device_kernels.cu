// Device kernels and launch helpers for the device suite. nvcc compiles this file; the gtest body
// calls the extern "C" wrappers, so GoogleTest never touches the CUDA compiler.

#include <cuda_runtime.h>

namespace {

// __nanosleep was measured to be a ~no-op on this driver (610.x, sm_75): __nanosleep(1000 ms)
// returned in ~1 ms. clock64() ticks at the SM clock (~1.46 GHz), so the spin is reliable.
__device__ void busy_spin(unsigned long long cycles) {
    const unsigned long long start = static_cast<unsigned long long>(clock64());
    while (static_cast<unsigned long long>(clock64()) - start < cycles) {
    }
}

__global__ void write_value_kernel(int* out, int value, unsigned long long cycles) {
    *out = value;
    busy_spin(cycles);
}

// Spin only, a separate kernel on purpose: writing through a nullptr here would poison the CUDA
// context for every test that follows.
__global__ void spin_only_kernel(unsigned long long cycles) {
    busy_spin(cycles);
}

}  // namespace

// The kernels are in an anonymous namespace, so these wrappers are the stable ABI the C++ test links.
extern "C" void poolkit_test_write_value(int* out, int value, unsigned long long cycles,
                                         cudaStream_t stream) {
    write_value_kernel<<<1, 1, 0, stream>>>(out, value, cycles);
}

extern "C" void poolkit_test_spin(unsigned long long cycles, cudaStream_t stream) {
    spin_only_kernel<<<1, 1, 0, stream>>>(cycles);
}

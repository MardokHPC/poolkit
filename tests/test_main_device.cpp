// Main for the device suite: plain C++, so the gtest headers stay out of nvcc. Exit 77 when no device,
// which ctest reports as SKIPPED thanks to SKIP_RETURN_CODE 77.

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <cstdio>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::printf("poolkit: no CUDA device available, skipping the device suite\n");
        return 77;
    }
    return RUN_ALL_TESTS();
}

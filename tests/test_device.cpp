// Device suite: real cudaMalloc, real events, real kernels. Without a device every test exits 77
// (see test_main_device.cpp). The no-synchronisation invariant is asserted by behaviour, not by
// grepping: DeferredReuseWaitsForTheRealEvent would reuse a pending block if allocate() waited.

#include <poolkit/pool.cuh>
#include <poolkit/pool.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" void poolkit_test_write_value(int* out, int value, unsigned long long cycles,
                                         cudaStream_t stream);
extern "C" void poolkit_test_spin(unsigned long long cycles, cudaStream_t stream);

namespace {

std::uintptr_t address(const void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr);
}

// ~100 ms of busy spin on the SM clock: the event is reliably still pending when the test queries
// it, and the suite stays fast.
constexpr unsigned long long kBusyCycles = 150000000ull;

}  // namespace

TEST(DevicePool, RealAllocationReturnsAlignedValidPointers) {
    poolkit::CudaBackend backend;
    poolkit::Pool pool(backend);

    std::vector<poolkit::DevicePtr> live;
    live.reserve(64);
    for (std::size_t i = 1; i <= 64; ++i) {
        const std::size_t size = 1 + (i * 7919) % 100000;
        poolkit::DevicePtr ptr = pool.allocate(size);
        ASSERT_TRUE(ptr.valid()) << "iteration " << i;
        EXPECT_EQ(address(ptr.get()) % 256, std::uintptr_t{0})
            << "size " << size;
        EXPECT_GE(ptr.capacity(), ptr.size());
        EXPECT_EQ(ptr.capacity() % 256, std::size_t{0});
        live.push_back(std::move(ptr));
    }
    for (auto& ptr : live) {
        pool.free(std::move(ptr));
    }
    EXPECT_EQ(pool.snapshot().bytes_requested_live, std::size_t{0});
}

TEST(DeviceBackend, ResidentStepDefaultsToUnknown) {
    poolkit::CudaBackend backend;
    EXPECT_EQ(backend.resident_step(), std::size_t{0});
    backend.set_resident_step(std::size_t{2} << 20);
    EXPECT_EQ(backend.resident_step(), std::size_t{2} << 20);
}

TEST(DevicePool, DeferredReuseWaitsForTheRealEvent) {
    poolkit::CudaBackend backend;
    poolkit::Pool pool(backend);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    constexpr int kValue = 0x5a5a5a5a;
    constexpr int kOther = 0x11111111;
    constexpr std::size_t kBytes = 1024;

    poolkit::DevicePtr first = pool.allocate(kBytes, stream);
    ASSERT_TRUE(first.valid());
    int* first_addr = static_cast<int*>(first.get());

    // Keep the stream busy, so the event recorded by free() below is still pending.
    poolkit_test_write_value(first_addr, kValue, kBusyCycles, stream);

    const std::size_t growth_before = pool.snapshot().growth_count;
    pool.free(std::move(first), stream);  // records an event on `stream`

    // Same class while the stream is busy: the pool must grow, not reuse.
    poolkit::DevicePtr second = pool.allocate(kBytes, stream);
    ASSERT_TRUE(second.valid());
    EXPECT_NE(second.get(), first_addr) << "reused a block whose event is still pending";
    EXPECT_EQ(pool.snapshot().growth_count, growth_before + 1);

    // If the pool had handed out the in-flight block, this write would corrupt it.
    poolkit_test_write_value(static_cast<int*>(second.get()), kOther, 0ull, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(pool.poll(), std::size_t{1});  // the pending block is now ready

    poolkit::DevicePtr third = pool.allocate(kBytes, stream);
    ASSERT_TRUE(third.valid());
    EXPECT_EQ(third.get(), first_addr) << "did not recycle once the event completed";
    EXPECT_EQ(pool.snapshot().reuse_count, std::size_t{1});

    int host = 0;
    ASSERT_EQ(cudaMemcpy(&host, first_addr, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(host, kValue);

    pool.free(std::move(third), stream);
    pool.free(std::move(second), stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(DevicePool, AllocateAndFreeDoNotBlockOnTheDevice) {
    poolkit::CudaBackend backend;
    poolkit::Pool pool(backend);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    poolkit_test_spin(300000000ull, stream);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000; ++i) {
        poolkit::DevicePtr ptr = pool.allocate(512, stream);
        pool.free(std::move(ptr), stream);
    }
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();

    // 10 000 deferred frees cost milliseconds; a cudaDeviceSynchronize per operation would
    // serialise with the 500 ms kernel.
    EXPECT_LT(elapsed_ms, 5000.0) << "allocate/free appear to synchronise with the device";

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(DevicePool, ReleaseAllRefusesWhileAHandleIsAlive) {
    poolkit::CudaBackend backend;
    poolkit::Pool pool(backend);

    poolkit::DevicePtr live_ptr = pool.allocate(1000);
    const std::size_t resident = pool.snapshot().bytes_resident;
    ASSERT_GT(resident, std::size_t{0});

    EXPECT_EQ(pool.release_all(), std::size_t{0});
    poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.release_all_refused, std::size_t{1});
    EXPECT_EQ(snap.bytes_resident, resident);
    EXPECT_TRUE(live_ptr.valid());

    pool.free(std::move(live_ptr));

    // Test-only synchronisation: the chunk goes back to the driver only once the event recorded
    // by free() has completed, and the pool never waits for that.
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(pool.poll(), std::size_t{1});  // the block is reusable now
    EXPECT_EQ(pool.release_all(), resident);
    EXPECT_EQ(pool.snapshot().bytes_resident, std::size_t{0});
}

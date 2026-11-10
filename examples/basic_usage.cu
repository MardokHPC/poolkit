// The smallest honest poolkit program: create a pool, allocate, look at what the pool did, free, hand
// the memory back. It is a ctest entry too, so the code in the README cannot rot away from the headers.

#include <poolkit/pool.cuh>
#include <poolkit/pool.hpp>

#include <cstddef>
#include <cstdio>
#include <utility>

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;

void print_snapshot(const char* when, const poolkit::PoolSnapshot& snap) {
    std::printf("%s\n", when);
    std::printf("  resident           %zu B in %zu chunks\n", snap.bytes_resident,
                snap.chunk_count);
    std::printf("  live               %zu B requested, %zu B capacity\n", snap.bytes_requested_live,
                snap.bytes_capacity_live);
    std::printf("  large path         %zu B in %zu blocks (never pooled, never reused)\n",
                snap.bytes_large_live, snap.large_path_count);
    std::printf("  fragmentation      internal %.2f %%, tail %.2f %%, resident slack %.2f %%\n",
                100.0 * snap.internal_fragmentation, 100.0 * snap.tail_fragmentation,
                100.0 * snap.resident_slack);
    std::printf("  reuse/growth       %zu / %zu (hit rate %.1f %%)\n", snap.reuse_count,
                snap.growth_count, 100.0 * snap.hit_rate);
    std::printf("  usable free for 4 KiB: %zu B\n", snap.usable_free(4096));
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "basic_usage: no CUDA device (exit 77)\n");
        return 77;
    }

    poolkit::CudaBackend backend;  // the only CUDA-specific object in the program
    poolkit::Pool pool(backend);   // default config: 2 MiB chunks, g = 1.5, 256 B..2 MiB

    try {
        // Two sizes inside the table, one above: the third takes the large path, tracked but never
        // carved or reused.
        poolkit::DevicePtr small = pool.allocate(4096);
        poolkit::DevicePtr medium = pool.allocate(256 * 1024);
        poolkit::DevicePtr large = pool.allocate(8 * kMiB);

        print_snapshot("after three allocations:", pool.snapshot());

        pool.free(std::move(small));
        pool.free(std::move(medium));
        pool.free(std::move(large));

        (void)cudaDeviceSynchronize();
        const std::size_t promoted = pool.poll();
        std::printf("poll promoted %zu blocks to reusable\n", promoted);

        const std::size_t returned = pool.release_all();
        std::printf("release_all returned %zu B, resident is now %zu B\n", returned,
                    pool.capacity());
    } catch (const poolkit::PoolError& error) {
        // allocate() throws instead of returning nullptr: an OOM is not a value.
        std::fprintf(stderr, "basic_usage: %s\n", error.what());
        return 1;
    }

    return 0;
}

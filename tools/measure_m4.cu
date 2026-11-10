// M4 measurement tool: cudaMalloc granularity, cudaEventQuery cost and pool allocate/free cost,
// printed in the form docs/measurements-m4.md transcribes. A device tool: nvcc, poolkit_cuda, not
// part of the default build.

#include <poolkit/pool.cuh>
#include <poolkit/pool.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// Busy-wait on the SM clock: __nanosleep is a ~no-op on this driver, clock64() is not.
__global__ void spin_kernel(unsigned long long cycles) {
    const unsigned long long start = static_cast<unsigned long long>(clock64());
    while (static_cast<unsigned long long>(clock64()) - start < cycles) {
    }
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

double ns_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::nano>(b - a).count();
}

struct RunStats {
    double median = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

RunStats compute_stats(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    const auto at = [&](double p) {
        const std::size_t idx =
            static_cast<std::size_t>(p * static_cast<double>(samples.size() - 1));
        return samples[idx];
    };
    RunStats s;
    s.median = at(0.50);
    s.p99 = at(0.99);
    s.min = samples.front();
    s.max = samples.back();
    return s;
}

struct Aggregated {
    double median = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

// 1 warmup + N runs: median of the per-run medians, median of the p99s, smallest min, largest max.
template <typename Body>
Aggregated measure(int runs, int per_run, Body&& body) {
    std::vector<double> samples(static_cast<std::size_t>(per_run));

    for (int i = 0; i < per_run; ++i) {
        samples[static_cast<std::size_t>(i)] = body();
    }

    std::vector<double> medians;
    std::vector<double> p99s;
    std::vector<double> mins;
    std::vector<double> maxs;
    medians.reserve(static_cast<std::size_t>(runs));
    p99s.reserve(static_cast<std::size_t>(runs));
    mins.reserve(static_cast<std::size_t>(runs));
    maxs.reserve(static_cast<std::size_t>(runs));

    for (int r = 0; r < runs; ++r) {
        for (int i = 0; i < per_run; ++i) {
            samples[static_cast<std::size_t>(i)] = body();
        }
        const RunStats stats = compute_stats(samples);
        medians.push_back(stats.median);
        p99s.push_back(stats.p99);
        mins.push_back(stats.min);
        maxs.push_back(stats.max);
    }

    std::sort(medians.begin(), medians.end());
    std::sort(p99s.begin(), p99s.end());
    Aggregated agg;
    agg.median = medians[medians.size() / 2];
    agg.p99 = p99s[p99s.size() / 2];
    agg.min = *std::min_element(mins.begin(), mins.end());
    agg.max = *std::max_element(maxs.begin(), maxs.end());
    return agg;
}

void print_stats(const char* label, const Aggregated& agg) {
    std::printf("  %-28s median %10.1f ns   p99 %10.1f ns   min %10.1f ns   max %10.1f ns\n",
                label, agg.median, agg.p99, agg.min, agg.max);
}

void print_header() {
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
    int driver_version = 0;
    check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    int runtime_version = 0;
    check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    std::printf("=== poolkit M4 measurements ===\n");
    std::printf("GPU            : %s\n", props.name);
    std::printf("compute        : sm_%d%d\n", props.major, props.minor);
    std::printf("driver version : %d\n", driver_version);
    std::printf("runtime version: %d\n", runtime_version);
    std::printf("total mem      : %zu MiB\n", total_bytes >> 20);
    std::printf("free mem       : %zu MiB\n", free_bytes >> 20);
    std::printf("\n");
}

void measure_granularity() {
    std::printf("--- experiment 1: cudaMalloc granularity ---\n");

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    // A single 1 MiB allocation shows a 0-byte delta (the driver had already reserved the memory), so
    // the granularity only appears across a cumulative, growing sequence.
    std::printf("1a. increasing blocks (single run, cumulative):\n");
    std::vector<void*> blocks;
    std::size_t free_prev = free_bytes;
    for (const std::size_t mb : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u}) {
        const std::size_t bytes = mb * 1024 * 1024;
        void* ptr = nullptr;
        check(cudaMalloc(&ptr, bytes), "cudaMalloc (growing)");
        blocks.push_back(ptr);
        std::size_t free_now = 0;
        check(cudaMemGetInfo(&free_now, &total_bytes), "cudaMemGetInfo");
        const std::size_t delta = free_prev - free_now;
        std::printf("  alloc %4zu MiB  ->  free delta %8zu B  (%zu MiB)\n", mb, delta,
                    delta >> 20);
        free_prev = free_now;
    }
    for (void* ptr : blocks) {
        check(cudaFree(ptr), "cudaFree (growing)");
    }

    // Guarded so a busy card fails loudly instead of corrupting the result; the margin keeps ~1 GiB free.
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const std::size_t requested = 1000 * std::size_t{4} * 1024 * 1024;
    const std::size_t margin = std::size_t{1} << 30;  // leave 1 GiB free
    std::printf("1b. 1000 x 4 MiB (requested %zu MiB, free before %zu MiB):\n",
                requested >> 20, free_bytes >> 20);
    if (requested + margin > free_bytes) {
        std::printf("  SKIPPED: not enough free memory to leave margin\n");
    } else {
        std::vector<void*> many;
        many.reserve(1000);
        std::size_t free_before = free_bytes;
        std::size_t first_delta = 0;
        bool first = true;
        std::size_t last_delta = 0;
        for (int i = 0; i < 1000; ++i) {
            void* ptr = nullptr;
            check(cudaMalloc(&ptr, std::size_t{4} * 1024 * 1024), "cudaMalloc (4 MiB)");
            many.push_back(ptr);
            std::size_t free_now = 0;
            check(cudaMemGetInfo(&free_now, &total_bytes), "cudaMemGetInfo");
            last_delta = free_before - free_now;
            if (first) {
                first_delta = last_delta;
                first = false;
            }
            free_before = free_now;
        }
        std::size_t free_after = 0;
        check(cudaMemGetInfo(&free_after, &total_bytes), "cudaMemGetInfo");
        std::printf("  first alloc delta: %zu B (%zu MiB)\n", first_delta, first_delta >> 20);
        std::printf("  steady alloc delta: %zu B (%zu MiB)\n", last_delta, last_delta >> 20);
        std::printf("  total consumed (free_before - free_after): %zu B (%zu MiB)\n",
                    free_bytes - free_after, (free_bytes - free_after) >> 20);
        for (void* ptr : many) {
            check(cudaFree(ptr), "cudaFree (4 MiB)");
        }
    }
    std::printf("\n");
}

void measure_query_cost() {
    std::printf("--- experiment 2: cudaEventQuery cost ---\n");

    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");

    cudaEvent_t done = nullptr;
    check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming),
          "cudaEventCreateWithFlags (done)");
    check(cudaEventRecord(done, stream), "cudaEventRecord (done)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    cudaEvent_t pending = nullptr;
    check(cudaEventCreateWithFlags(&pending, cudaEventDisableTiming),
          "cudaEventCreateWithFlags (pending)");
    spin_kernel<<<1, 1, 0, stream>>>(4000000000ull);  // ~2.7 s of SM-clock spin
    check(cudaEventRecord(pending, stream), "cudaEventRecord (pending)");
    check(cudaGetLastError(), "kernel launch");
    const bool pending_at_start = (cudaEventQuery(pending) == cudaErrorNotReady);
    std::printf("  pending event is NotReady at start: %s\n", pending_at_start ? "yes" : "no");

    const int kRuns = 5;
    const int kQueries = 100000;

    Aggregated done_stats = measure(kRuns, kQueries, [&]() {
        const Clock::time_point start = Clock::now();
        (void)cudaEventQuery(done);
        return ns_between(start, Clock::now());
    });
    print_stats("query (completed event)", done_stats);

    Aggregated pending_stats = measure(kRuns, kQueries, [&]() {
        const Clock::time_point start = Clock::now();
        (void)cudaEventQuery(pending);
        return ns_between(start, Clock::now());
    });
    print_stats("query (pending event)", pending_stats);

    const bool still_pending = (cudaEventQuery(pending) == cudaErrorNotReady);
    std::printf("  pending event still NotReady after runs: %s\n", still_pending ? "yes" : "no");
    if (!still_pending) {
        std::printf("  WARNING: the spin ended before the runs finished; the pending\n"
                    "           number is contaminated with completed-query latency.\n");
    }

    check(cudaEventDestroy(done), "cudaEventDestroy (done)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize (drain)");
    check(cudaEventDestroy(pending), "cudaEventDestroy (pending)");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::printf("\n");
}

void measure_pool_cost() {
    std::printf("--- experiment 3: pool allocate/free, hit vs growth ---\n");
    const int kRuns = 5;
    const int kOps = 10000;

    poolkit::CudaBackend backend;
    poolkit::Pool pool(backend);

        // Every allocate reuses (promote_pending -> ready -> reuse), every free records an event on an idle stream.
    {
        poolkit::DevicePtr seed = pool.allocate(1024);
        pool.free(std::move(seed));

        Aggregated alloc_stats = measure(kRuns, kOps, [&]() {
            const Clock::time_point start = Clock::now();
            poolkit::DevicePtr ptr = pool.allocate(1024);
            pool.free(std::move(ptr));
            return ns_between(start, Clock::now());
        });
        std::printf("  hit path (alloc+free, one block, %d ops):\n", kOps);
        print_stats("allocate+free total", alloc_stats);

        Aggregated free_stats = measure(kRuns, kOps, [&]() {
            poolkit::DevicePtr ptr = pool.allocate(1024);
            const Clock::time_point start = Clock::now();
            pool.free(std::move(ptr));
            return ns_between(start, Clock::now());
        });
        print_stats("free (event record)", free_stats);
    }

    {
        poolkit::CudaBackend growth_backend;
        poolkit::Pool growth(growth_backend);
        std::vector<poolkit::DevicePtr> live;
        live.reserve(static_cast<std::size_t>(kOps));

        Aggregated growth_stats = measure(kRuns, kOps, [&]() {
            const Clock::time_point start = Clock::now();
            live.push_back(growth.allocate(1024));
            return ns_between(start, Clock::now());
        });
        std::printf("  growth path (alloc only, %d blocks, never freed):\n", kOps);
        print_stats("allocate (growth)", growth_stats);

        for (auto& ptr : live) {
            growth.free(std::move(ptr));
        }
    }
    std::printf("\n");
}

}  // namespace

int main() {
    print_header();
    measure_granularity();
    measure_query_cost();
    measure_pool_cost();
    return 0;
}

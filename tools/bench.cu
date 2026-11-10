// poolkit_bench: replay one trace through poolkit, cudaMalloc and cudaMallocAsync and
// report wall time, peak and per-operation latency. A number without its config is a
// number to throw away, so GPU, driver, runtime, arch and pool config travel with it.

#include <poolkit/pool.cuh>
#include <poolkit/pool.hpp>
#include <poolkit/trace.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "advise.hpp"
#include "report_json.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using poolkit::DevicePtr;
using poolkit::Pool;
using poolkit::Trace;
using poolkit::TraceOp;
using poolkit::TraceOpKind;
using poolkit::tools::BenchDistribution;
using poolkit::tools::BenchMode;

// Every 64 operations: enough to catch a peak that lasts longer, and 64x cheaper than sampling every op.
constexpr std::size_t kSampleEveryOps = 64;

struct SampleStats {
    double median = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

SampleStats stats_of(std::vector<double>& samples) {
    SampleStats stats;
    if (samples.empty()) {
        return stats;
    }
    std::sort(samples.begin(), samples.end());
    const auto at = [&](double p) {
        const std::size_t index =
            static_cast<std::size_t>(p * static_cast<double>(samples.size() - 1));
        return samples[index];
    };
    stats.median = at(0.50);
    stats.p99 = at(0.99);
    stats.min = samples.front();
    stats.max = samples.back();
    return stats;
}

// Rule 2 applied to a distribution: median of the per-run medians, smallest min, largest max.
// An empty run is skipped, not counted as zero, which would hide a failed run.
class Bucket {
public:
    void add(const std::vector<double>& samples) {
        if (samples.empty()) {
            return;
        }
        std::vector<double> copy = samples;
        const SampleStats stats = stats_of(copy);
        medians_.push_back(stats.median);
        p99s_.push_back(stats.p99);
        mins_.push_back(stats.min);
        maxs_.push_back(stats.max);
    }

    BenchDistribution finish() const {
        BenchDistribution distribution;
        if (medians_.empty()) {
            return distribution;
        }
        std::vector<double> medians = medians_;
        std::vector<double> p99s = p99s_;
        std::sort(medians.begin(), medians.end());
        std::sort(p99s.begin(), p99s.end());
        distribution.median = medians[medians.size() / 2];
        distribution.p99 = p99s[p99s.size() / 2];
        distribution.min = *std::min_element(mins_.begin(), mins_.end());
        distribution.max = *std::max_element(maxs_.begin(), maxs_.end());
        return distribution;
    }

private:
    std::vector<double> medians_;
    std::vector<double> p99s_;
    std::vector<double> mins_;
    std::vector<double> maxs_;
};

double ns_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::nano>(b - a).count();
}

std::string ns_text(double ns) {
    char buffer[64];
    if (ns >= 1.0e6) {
        std::snprintf(buffer, sizeof(buffer), "%.3f ms", ns / 1.0e6);
    } else if (ns >= 1.0e3) {
        std::snprintf(buffer, sizeof(buffer), "%.1f us", ns / 1.0e3);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.0f ns", ns);
    }
    return std::string(buffer);
}

std::string mib_text(std::size_t bytes) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.2f MiB",
                  static_cast<double>(bytes) / 1048576.0);
    return std::string(buffer);
}

std::string arch_text() {
#if defined(_M_X64) || defined(__x86_64__)
    return "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "aarch64";
#else
    return "unknown";
#endif
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "poolkit_bench: CUDA error in %s: %s\n", what,
                     cudaGetErrorString(status));
        std::exit(2);
    }
}

// Mirrors trace.hpp's replay_trace() with the two things the library replay does not need: a
// clock around each operation and the exact peak. Keeping it here keeps the portable replay
// free of <chrono>; tests/test_trace.cpp still pins replay_trace().
struct Latencies {
    std::vector<double> alloc;
    std::vector<double> free_ns;
    // hit vs growth, growth split into slot / chunk / large: a median alone leaves the cost unexplained.
    std::vector<double> hit;
    std::vector<double> growth;
    std::vector<double> slot;
    std::vector<double> chunk;
    std::vector<double> large;
};

// kTiming false is the output contract: nothing between the caller's two clock reads but the
// trace, and the handles the trace leaves live are released after the clock stops.
template <bool kTiming>
std::size_t replay_pool(Pool& pool, const Trace& trace, Latencies* latencies,
                        std::vector<DevicePtr>* leftovers) {
    std::unordered_map<std::uint64_t, DevicePtr> live;
    std::unordered_map<std::uint64_t, std::size_t> live_large;
    std::size_t large_live_bytes = 0;
    std::size_t peak = 0;
    const std::size_t max_class_bytes = pool.config().max_class_bytes;

    for (const TraceOp& op : trace.ops) {
        if (op.kind == TraceOpKind::alloc) {
            std::size_t reuse_before = 0;
            std::size_t capacity_before = 0;
            Clock::time_point start{};
            if (kTiming) {
                reuse_before = pool.reuse_count();
                capacity_before = pool.capacity();
                start = Clock::now();
            }
            DevicePtr ptr = pool.allocate(op.size, poolkit::trace_stream(op.stream));
            if (kTiming) {
                const double ns = ns_between(start, Clock::now());
                latencies->alloc.push_back(ns);
                if (pool.reuse_count() != reuse_before) {
                    latencies->hit.push_back(ns);
                } else {
                    latencies->growth.push_back(ns);
                    if (op.size > max_class_bytes) {
                        latencies->large.push_back(ns);
                    } else if (pool.capacity() != capacity_before) {
                        latencies->chunk.push_back(ns);
                    } else {
                        latencies->slot.push_back(ns);
                    }
                }
            }
            live.emplace(op.ptr, std::move(ptr));
            if (op.size > max_class_bytes) {
                live_large.emplace(op.ptr, op.size);
                large_live_bytes += op.size;
            }
        } else {
            const auto it = live.find(op.ptr);
            if (it == live.end()) {
                std::fprintf(stderr, "poolkit_bench: free of inactive ptr %llu\n",
                             static_cast<unsigned long long>(op.ptr));
                std::exit(2);
            }
            Clock::time_point start{};
            if (kTiming) {
                start = Clock::now();
            }
            pool.free(std::move(it->second), poolkit::trace_stream(op.stream));
            if (kTiming) {
                latencies->free_ns.push_back(ns_between(start, Clock::now()));
            }
            live.erase(it);
            const auto large = live_large.find(op.ptr);
            if (large != live_large.end()) {
                large_live_bytes -= large->second;
                live_large.erase(large);
            }
        }
        const std::size_t current = pool.capacity() + large_live_bytes;
        if (current > peak) {
            peak = current;
        }
    }

    for (auto& entry : live) {
        leftovers->push_back(std::move(entry.second));
    }
    return peak;
}

struct DriverRun {
    std::size_t peak = 0;
    Latencies latencies;
};

// cudaMalloc / cudaFree; the peak is sampled every kSampleEveryOps operations.
DriverRun replay_cudamalloc(const Trace& trace, bool timing) {
    DriverRun run;
    std::unordered_map<std::uint64_t, void*> live;
    std::size_t free_before = 0;
    std::size_t total = 0;
    check(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo (baseline)");

    std::size_t op_index = 0;
    const auto sample = [&]() {
        ++op_index;
        if (op_index % kSampleEveryOps != 0) {
            return;
        }
        std::size_t free_now = 0;
        check(cudaMemGetInfo(&free_now, &total), "cudaMemGetInfo (sample)");
        if (free_now <= free_before) {
            const std::size_t used = free_before - free_now;
            if (used > run.peak) {
                run.peak = used;
            }
        }
    };

    for (const TraceOp& op : trace.ops) {
        if (op.kind == TraceOpKind::alloc) {
            const Clock::time_point start = timing ? Clock::now() : Clock::time_point{};
            void* ptr = nullptr;
            check(cudaMalloc(&ptr, op.size), "cudaMalloc");
            if (timing) {
                run.latencies.alloc.push_back(ns_between(start, Clock::now()));
            }
            live.emplace(op.ptr, ptr);
        } else {
            const auto it = live.find(op.ptr);
            if (it == live.end()) {
                std::fprintf(stderr, "poolkit_bench: free of inactive ptr %llu\n",
                             static_cast<unsigned long long>(op.ptr));
                std::exit(2);
            }
            const Clock::time_point start = timing ? Clock::now() : Clock::time_point{};
            check(cudaFree(it->second), "cudaFree");
            if (timing) {
                run.latencies.free_ns.push_back(ns_between(start, Clock::now()));
            }
            live.erase(it);
        }
        sample();
    }
    for (auto& entry : live) {
        check(cudaFree(entry.second), "cudaFree (teardown)");
    }
    std::size_t free_end = 0;
    check(cudaMemGetInfo(&free_end, &total), "cudaMemGetInfo (final)");
    if (free_end <= free_before && free_before - free_end > run.peak) {
        run.peak = free_before - free_end;
    }
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize (cudamalloc)");
    return run;
}

// The latency measured is the cost of *enqueueing*, not of performing: an async allocation
// that has to ask the driver returns before the memory exists.
DriverRun replay_cudaasync(const Trace& trace, bool timing) {
    DriverRun run;
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    std::unordered_map<std::uint64_t, void*> live;

    std::size_t free_before = 0;
    std::size_t total = 0;
    check(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo (baseline)");

    std::size_t op_index = 0;
    const auto sample = [&]() {
        ++op_index;
        if (op_index % kSampleEveryOps != 0) {
            return;
        }
        std::size_t free_now = 0;
        check(cudaMemGetInfo(&free_now, &total), "cudaMemGetInfo (sample)");
        if (free_now <= free_before) {
            const std::size_t used = free_before - free_now;
            if (used > run.peak) {
                run.peak = used;
            }
        }
    };

    for (const TraceOp& op : trace.ops) {
        if (op.kind == TraceOpKind::alloc) {
            const Clock::time_point start = timing ? Clock::now() : Clock::time_point{};
            void* ptr = nullptr;
            check(cudaMallocAsync(&ptr, op.size, stream), "cudaMallocAsync");
            if (timing) {
                run.latencies.alloc.push_back(ns_between(start, Clock::now()));
            }
            live.emplace(op.ptr, ptr);
        } else {
            const auto it = live.find(op.ptr);
            if (it == live.end()) {
                std::fprintf(stderr, "poolkit_bench: free of inactive ptr %llu\n",
                             static_cast<unsigned long long>(op.ptr));
                std::exit(2);
            }
            const Clock::time_point start = timing ? Clock::now() : Clock::time_point{};
            check(cudaFreeAsync(it->second, stream), "cudaFreeAsync");
            if (timing) {
                run.latencies.free_ns.push_back(ns_between(start, Clock::now()));
            }
            live.erase(it);
        }
        sample();
    }
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize (cudaasync)");
    for (auto& entry : live) {
        check(cudaFreeAsync(entry.second, stream), "cudaFreeAsync (teardown)");
    }
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize (teardown)");
    std::size_t free_end = 0;
    check(cudaMemGetInfo(&free_end, &total), "cudaMemGetInfo (final)");
    if (free_end <= free_before && free_before - free_end > run.peak) {
        run.peak = free_before - free_end;
    }
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return run;
}

// Wall time comes from uninstrumented replays, latency from instrumented ones; between modes
// nothing stays resident.

struct Options {
    std::string trace_path;
    std::string json_path;
    std::string mode = "all";
    int repeat = 5;
    int warmup = 1;
};

struct ModeOutcome {
    BenchMode mode;
    bool ok = false;
};

ModeOutcome measure_pool(const Trace& trace, const Pool::Config& config, const Options& options) {
    ModeOutcome outcome;
    BenchMode& mode = outcome.mode;
    mode.mode = "pool";
    mode.ops = trace.ops.size();
    mode.allocs = 0;
    mode.frees = 0;
    for (const TraceOp& op : trace.ops) {
        if (op.kind == TraceOpKind::alloc) {
            ++mode.allocs;
        } else {
            ++mode.frees;
        }
    }

    std::vector<double> wall_runs;
    std::size_t peak_exact = 0;
    for (int run = 0; run < options.warmup + options.repeat; ++run) {
        poolkit::CudaBackend backend;
        Pool pool(backend, config);
        std::vector<DevicePtr> leftovers;
        const Clock::time_point start = Clock::now();
        const std::size_t peak = replay_pool<false>(pool, trace, nullptr, &leftovers);
        const double ns = ns_between(start, Clock::now());
        for (DevicePtr& ptr : leftovers) {
            ptr.release();
        }
        if (run >= options.warmup) {
            wall_runs.push_back(ns);
        }
        peak_exact = std::max(peak_exact, peak);
    }
    Bucket wall_bucket;
    wall_bucket.add(wall_runs);
    mode.wall_ns = wall_bucket.finish();
    mode.peak_bytes = peak_exact;
    mode.peak_sampled = false;
    mode.sample_every_ops = 0;

    Bucket alloc_bucket;
    Bucket free_bucket;
    Bucket hit_bucket;
    Bucket growth_bucket;
    Bucket slot_bucket;
    Bucket chunk_bucket;
    Bucket large_bucket;
    for (int run = 0; run < options.warmup + options.repeat; ++run) {
        poolkit::CudaBackend backend;
        Pool pool(backend, config);
        Latencies latencies;
        std::vector<DevicePtr> leftovers;
        (void)replay_pool<true>(pool, trace, &latencies, &leftovers);
        for (DevicePtr& ptr : leftovers) {
            ptr.release();
        }
        if (run >= options.warmup) {
            alloc_bucket.add(latencies.alloc);
            free_bucket.add(latencies.free_ns);
            hit_bucket.add(latencies.hit);
            growth_bucket.add(latencies.growth);
            slot_bucket.add(latencies.slot);
            chunk_bucket.add(latencies.chunk);
            large_bucket.add(latencies.large);
            mode.hit_ops = latencies.hit.size();
            mode.growth_ops = latencies.growth.size();
            mode.slot_ops = latencies.slot.size();
            mode.chunk_ops = latencies.chunk.size();
            mode.large_ops = latencies.large.size();
        }
    }
    mode.alloc_ns = alloc_bucket.finish();
    mode.free_ns = free_bucket.finish();
    mode.hit_ns = hit_bucket.finish();
    mode.growth_ns = growth_bucket.finish();
    mode.slot_ns = slot_bucket.finish();
    mode.chunk_ns = chunk_bucket.finish();
    mode.large_ns = large_bucket.finish();
    outcome.ok = true;
    return outcome;
}

ModeOutcome measure_cudamalloc(const Trace& trace, const Options& options) {
    ModeOutcome outcome;
    BenchMode& mode = outcome.mode;
    mode.mode = "cudamalloc";
    mode.ops = trace.ops.size();
    mode.allocs = 0;
    mode.frees = 0;
    for (const TraceOp& op : trace.ops) {
        if (op.kind == TraceOpKind::alloc) {
            ++mode.allocs;
        } else {
            ++mode.frees;
        }
    }

    std::vector<double> wall_runs;
    std::size_t peak = 0;
    for (int run = 0; run < options.warmup + options.repeat; ++run) {
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize (before cudamalloc run)");
        const Clock::time_point start = Clock::now();
        const DriverRun driver = replay_cudamalloc(trace, false);
        const double ns = ns_between(start, Clock::now());
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize (after cudamalloc run)");
        if (run >= options.warmup) {
            wall_runs.push_back(ns);
        }
        peak = std::max(peak, driver.peak);
    }
    Bucket wall_bucket;
    wall_bucket.add(wall_runs);
    mode.wall_ns = wall_bucket.finish();
    mode.peak_bytes = peak;
    mode.peak_sampled = true;
    mode.sample_every_ops = kSampleEveryOps;

    Bucket alloc_bucket;
    Bucket free_bucket;
    for (int run = 0; run < options.warmup + options.repeat; ++run) {
        const DriverRun driver = replay_cudamalloc(trace, true);
        if (run >= options.warmup) {
            alloc_bucket.add(driver.latencies.alloc);
            free_bucket.add(driver.latencies.free_ns);
        }
    }
    mode.alloc_ns = alloc_bucket.finish();
    mode.free_ns = free_bucket.finish();
    outcome.ok = true;
    return outcome;
}

ModeOutcome measure_cudaasync(const Trace& trace, const Options& options) {
    ModeOutcome outcome;
    BenchMode& mode = outcome.mode;
    mode.mode = "cudaasync";
    mode.ops = trace.ops.size();
    mode.allocs = 0;
    mode.frees = 0;
    for (const TraceOp& op : trace.ops) {
        if (op.kind == TraceOpKind::alloc) {
            ++mode.allocs;
        } else {
            ++mode.frees;
        }
    }

    std::vector<double> wall_runs;
    std::size_t peak = 0;
    for (int run = 0; run < options.warmup + options.repeat; ++run) {
        const Clock::time_point start = Clock::now();
        const DriverRun driver = replay_cudaasync(trace, false);
        const double ns = ns_between(start, Clock::now());
        if (run >= options.warmup) {
            wall_runs.push_back(ns);
        }
        peak = std::max(peak, driver.peak);
    }
    Bucket wall_bucket;
    wall_bucket.add(wall_runs);
    mode.wall_ns = wall_bucket.finish();
    mode.peak_bytes = peak;
    mode.peak_sampled = true;
    mode.sample_every_ops = kSampleEveryOps;

    Bucket alloc_bucket;
    Bucket free_bucket;
    for (int run = 0; run < options.warmup + options.repeat; ++run) {
        const DriverRun driver = replay_cudaasync(trace, true);
        if (run >= options.warmup) {
            alloc_bucket.add(driver.latencies.alloc);
            free_bucket.add(driver.latencies.free_ns);
        }
    }
    mode.alloc_ns = alloc_bucket.finish();
    mode.free_ns = free_bucket.finish();
    outcome.ok = true;
    return outcome;
}

const char* peak_source(const BenchMode& mode) {
    return mode.peak_sampled ? "sampled" : "exact (pool)";
}

void print_table(const std::vector<BenchMode>& modes) {
    std::printf("| mode | wall median | wall min | wall max | peak | peak source "
                "| alloc p50 | alloc p99 | free p50 | free p99 |\n");
    std::printf("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|\n");
    for (const BenchMode& mode : modes) {
        std::printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", mode.mode.c_str(),
                    ns_text(mode.wall_ns.median).c_str(), ns_text(mode.wall_ns.min).c_str(),
                    ns_text(mode.wall_ns.max).c_str(), mib_text(mode.peak_bytes).c_str(),
                    peak_source(mode), ns_text(mode.alloc_ns.median).c_str(),
                    ns_text(mode.alloc_ns.p99).c_str(), ns_text(mode.free_ns.median).c_str(),
                    ns_text(mode.free_ns.p99).c_str());
    }
    std::printf("\n");
}

void print_pool_split(const BenchMode& mode) {
    const auto share = [&](std::size_t ops) {
        char buffer[32];
        const double percent = mode.allocs == 0
                                   ? 0.0
                                   : 100.0 * static_cast<double>(ops) /
                                         static_cast<double>(mode.allocs);
        std::snprintf(buffer, sizeof(buffer), "%zu (%.1f%%)", ops, percent);
        return std::string(buffer);
    };
    std::printf("pool allocation paths (same runs, %zu allocs):\n", mode.allocs);
    std::printf("| path | ops | median | p99 | min | max |\n");
    std::printf("|---|---:|---:|---:|---:|---:|\n");
    std::printf("| hit (recycled block) | %s | %s | %s | %s | %s |\n", share(mode.hit_ops).c_str(),
                ns_text(mode.hit_ns.median).c_str(), ns_text(mode.hit_ns.p99).c_str(),
                ns_text(mode.hit_ns.min).c_str(), ns_text(mode.hit_ns.max).c_str());
    std::printf("| growth, total | %s | %s | %s | %s | %s |\n",
                share(mode.growth_ops).c_str(), ns_text(mode.growth_ns.median).c_str(),
                ns_text(mode.growth_ns.p99).c_str(), ns_text(mode.growth_ns.min).c_str(),
                ns_text(mode.growth_ns.max).c_str());
    std::printf("|   new slot, existing chunk | %s | %s | %s | %s | %s |\n",
                share(mode.slot_ops).c_str(), ns_text(mode.slot_ns.median).c_str(),
                ns_text(mode.slot_ns.p99).c_str(), ns_text(mode.slot_ns.min).c_str(),
                ns_text(mode.slot_ns.max).c_str());
    std::printf("|   fresh chunk (cudaMalloc + carve) | %s | %s | %s | %s | %s |\n",
                share(mode.chunk_ops).c_str(), ns_text(mode.chunk_ns.median).c_str(),
                ns_text(mode.chunk_ns.p99).c_str(), ns_text(mode.chunk_ns.min).c_str(),
                ns_text(mode.chunk_ns.max).c_str());
    std::printf("|   large path (cudaMalloc, no reuse) | %s | %s | %s | %s | %s |\n",
                share(mode.large_ops).c_str(), ns_text(mode.large_ns.median).c_str(),
                ns_text(mode.large_ns.p99).c_str(), ns_text(mode.large_ns.min).c_str(),
                ns_text(mode.large_ns.max).c_str());
    std::printf("| free (event record) | %s | %s | %s | %s | %s |\n\n",
                share(mode.frees).c_str(), ns_text(mode.free_ns.median).c_str(),
                ns_text(mode.free_ns.p99).c_str(), ns_text(mode.free_ns.min).c_str(),
                ns_text(mode.free_ns.max).c_str());
}

void print_header(const cudaDeviceProp& properties, int driver_version, int runtime_version,
                  const Trace& trace, const Pool::Config& config, const std::string& trace_path,
                  const Options& options) {
    poolkit::SizeClassTable::Config table_config;
    table_config.min_class_bytes = config.min_class_bytes;
    table_config.max_class_bytes = config.max_class_bytes;
    table_config.growth = config.growth;
    table_config.include_power_of_two = config.include_power_of_two;
    table_config.extra_sizes = config.extra_sizes;
    const poolkit::SizeClassTable table(table_config);

    std::printf("poolkit_bench\n");
    std::printf("  GPU            : %s (sm_%d%d)\n", properties.name, properties.major,
                properties.minor);
    std::printf("  driver         : %d   runtime: %d   arch: %s\n", driver_version,
                runtime_version, arch_text().c_str());
    std::printf("  trace          : %s (%zu ops)\n", trace_path.c_str(), trace.ops.size());
    std::printf("  pool config    : chunk %s, growth %.2f, min class %zu B, max class %s, "
                "%zu classes\n",
                mib_text(config.chunk_bytes).c_str(), config.growth, config.min_class_bytes,
                mib_text(config.max_class_bytes).c_str(), table.count());
    std::printf("  method         : %d warmup + %d runs, steady_clock, median of run medians, "
                "min/max\n",
                options.warmup, options.repeat);
    std::printf("  peak sampling  : every %zu ops for the driver modes; pool peak exact\n",
                kSampleEveryOps);
    std::printf("  footprint      : one mode at a time, fresh pool per run, everything "
                "released between modes\n\n");
}

// Schema 2: the analysis block of schema 1 plus the measured bench section. `waste` is the same
// histogram integral docs/measurements-m4.md quotes, on this trace with this table.

poolkit::tools::Report build_report(const std::vector<BenchMode>& modes, const Trace& trace,
                                    const Options& options, const Pool::Config& config,
                                    const cudaDeviceProp& properties, int driver_version,
                                    int runtime_version) {
    poolkit::tools::Report report;
    report.schema = poolkit::tools::kBenchReportSchema;
    report.tool = "poolkit_bench";
    report.has_bench = true;

    report.trace.path = options.trace_path;
    report.trace.ops = trace.ops.size();
    report.trace.allocs = 0;
    for (const TraceOp& op : trace.ops) {
        if (op.kind == TraceOpKind::alloc) {
            ++report.trace.allocs;
        }
    }
    report.trace.sha256 = poolkit::tools::sha256_file(options.trace_path);

    report.config.growth = config.growth;
    report.config.chunk_bytes = config.chunk_bytes;
    report.config.min_class_bytes = config.min_class_bytes;
    report.config.max_class_bytes = config.max_class_bytes;
    report.config.power_of_two = config.include_power_of_two;
    report.config.extra_sizes = config.extra_sizes;

    // A Pool cannot be built without a backend, and the report only needs the table, from the same config.
    poolkit::SizeClassTable::Config table_config;
    table_config.min_class_bytes = config.min_class_bytes;
    table_config.max_class_bytes = config.max_class_bytes;
    table_config.growth = config.growth;
    table_config.include_power_of_two = config.include_power_of_two;
    table_config.extra_sizes = config.extra_sizes;
    const poolkit::SizeClassTable table(table_config);
    report.classes = table.count();

    std::size_t alloc_requests = 0;
    std::size_t largest = 0;
    const auto histogram = poolkit::tools::advise_detail::build_histogram(trace, alloc_requests,
                                                                         largest);
    report.waste.internal =
        poolkit::tools::advise_detail::integrate_internal_waste(table, histogram);
    report.waste.tail = poolkit::tools::advise_detail::integrate_tail_waste(
        table, histogram, config.chunk_bytes);
    report.retention_bound_bytes = table.count() * config.chunk_bytes;

    // The exact pool peak when it ran, otherwise the sampled driver peak; peak_sampled says which.
    report.peak.resident_bytes = 0;
    for (const BenchMode& mode : modes) {
        if (!mode.peak_sampled) {
            report.peak.resident_bytes = mode.peak_bytes;
            break;
        }
    }
    if (report.peak.resident_bytes == 0 && !modes.empty()) {
        report.peak.resident_bytes = modes.front().peak_bytes;
    }
    // The bench tracks bytes, not simultaneous handles, so this stays 0 rather than carrying a guess.
    report.peak.blocks = 0;

    poolkit::tools::BenchSection& bench = report.bench;
    bench.schema = poolkit::tools::kBenchSectionSchema;
    bench.env.gpu = properties.name;
    bench.env.compute = "sm_" + std::to_string(properties.major) +
                        std::to_string(properties.minor);
    bench.env.driver_version = driver_version;
    bench.env.runtime_version = runtime_version;
    bench.env.arch = arch_text();
    bench.env.chunk_bytes = config.chunk_bytes;
    bench.env.growth = config.growth;
    bench.env.min_class_bytes = config.min_class_bytes;
    bench.env.max_class_bytes = config.max_class_bytes;
    bench.env.warmup = options.warmup;
    bench.env.repeat = options.repeat;
    bench.modes = modes;
    return report;
}

void print_usage(const char* program) {
    std::printf("usage: %s --trace <trace.jsonl> [options]\n"
                "\n"
                "Replays one trace through poolkit, cudaMalloc and cudaMallocAsync and reports\n"
                "wall time, peak memory and per-operation latency for each.\n"
                "\n"
                "Options:\n"
                "  --trace <f>     trace to replay (required)\n"
                "  --repeat <n>    timed runs per mode (default 5)\n"
                "  --warmup <n>    discarded runs before them (default 1)\n"
                "  --mode <m>      pool | cudamalloc | cudaasync | all (default all)\n"
                "  --json <f>      write the schema-2 report\n"
                "  -h, --help      this help\n"
                "\n"
                "Exit codes: 0 success, 2 bad arguments, unreadable trace or CUDA error.\n",
                program);
}

bool parse_count(const std::string& text, int& out) {
    if (text.empty()) {
        return false;
    }
    long long value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        value = value * 10 + (ch - '0');
        if (value > 1000000) {
            return false;
        }
    }
    out = static_cast<int>(value);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto take = [&](std::string& target) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "poolkit_bench: missing value for %s\n", arg.c_str());
                return false;
            }
            target = argv[++i];
            return true;
        };
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--trace") {
            if (!take(options.trace_path)) {
                return 2;
            }
        } else if (arg == "--json") {
            if (!take(options.json_path)) {
                return 2;
            }
        } else if (arg == "--mode") {
            if (!take(options.mode)) {
                return 2;
            }
            if (options.mode != "all" && options.mode != "pool" &&
                options.mode != "cudamalloc" && options.mode != "cudaasync") {
                std::fprintf(stderr, "poolkit_bench: invalid --mode '%s'\n", options.mode.c_str());
                return 2;
            }
        } else if (arg == "--repeat") {
            std::string value;
            if (!take(value) || !parse_count(value, options.repeat) || options.repeat == 0) {
                std::fprintf(stderr, "poolkit_bench: invalid --repeat\n");
                return 2;
            }
        } else if (arg == "--warmup") {
            std::string value;
            if (!take(value) || !parse_count(value, options.warmup)) {
                std::fprintf(stderr, "poolkit_bench: invalid --warmup\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "poolkit_bench: unknown argument '%s'\n\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }
    if (options.trace_path.empty()) {
        std::fprintf(stderr, "poolkit_bench: --trace is required\n\n");
        print_usage(argv[0]);
        return 2;
    }

    Trace trace;
    try {
        trace = poolkit::parse_trace_file(options.trace_path);
    } catch (const poolkit::TraceError& error) {
        std::fprintf(stderr, "poolkit_bench: %s\n", error.what());
        return 2;
    }

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "poolkit_bench: no CUDA device (%s)\n",
                     cudaGetErrorString(count_status == cudaSuccess ? cudaErrorNoDevice
                                                                   : count_status));
        return 77;
    }
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    int driver_version = 0;
    check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    int runtime_version = 0;
    check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");

    Pool::Config config;  // the shipped default: this is the configuration the plan quotes
    print_header(properties, driver_version, runtime_version, trace, config, options.trace_path,
                 options);

    std::vector<BenchMode> modes;
    std::vector<std::string> failed;
    const auto run = [&](const char* name) {
        if (options.mode != "all" && options.mode != name) {
            return;
        }
        std::printf("measuring %s...\n", name);
        ModeOutcome outcome;
        if (std::strcmp(name, "pool") == 0) {
            outcome = measure_pool(trace, config, options);
        } else if (std::strcmp(name, "cudamalloc") == 0) {
            outcome = measure_cudamalloc(trace, options);
        } else {
            outcome = measure_cudaasync(trace, options);
        }
        if (outcome.ok) {
            modes.push_back(outcome.mode);
        } else {
            failed.emplace_back(name);
        }
    };
    run("pool");
    run("cudamalloc");
    run("cudaasync");

    std::printf("\n");
    print_table(modes);
    for (const BenchMode& mode : modes) {
        if (mode.mode == "pool") {
            print_pool_split(mode);
        }
    }
    for (const std::string& name : failed) {
        std::printf("%s: measurement failed and is not in the table or the JSON\n", name.c_str());
    }

    if (!options.json_path.empty()) {
        const poolkit::tools::Report report =
            build_report(modes, trace, options, config, properties, driver_version,
                         runtime_version);
        std::ofstream out;
        out.open(options.json_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "poolkit_bench: cannot open '%s' for writing\n",
                         options.json_path.c_str());
            return 2;
        }
        out << poolkit::tools::to_json(report);
        out.flush();
        std::printf("wrote %s (schema %d)\n", options.json_path.c_str(),
                    poolkit::tools::kBenchReportSchema);
    }
    return 0;
}

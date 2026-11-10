// Validates the trace poolkit_record produced from tests/device/record_target.cu, with no GPU and no
// CUPTI: parse, replay, check what the workload promised. --injected is looser on purpose: the shim
// subscribes before the context exists, so the driver's own allocations may be in the file.

#include <poolkit/pool.hpp>
#include <poolkit/sim_backend.hpp>
#include <poolkit/trace.hpp>

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Must match tests/device/record_target.cu: the three tracked sizes and the tiny one.
constexpr std::size_t kExpectedSizes[] = {4u << 20, 64u << 10, 1u << 20};
constexpr std::size_t kTinySize = 1024;

bool file_exists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

}  // namespace

int main(int argc, char** argv) {
    const bool injected = argc == 3 && std::strcmp(argv[2], "--injected") == 0;
    if (argc != 2 && !injected) {
        std::fprintf(stderr, "usage: %s <trace.jsonl> [--injected]\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    if (!file_exists(path)) {
        std::fprintf(stderr, "record_check: no trace at '%s' (no recorder run?)\n", path.c_str());
        return 77;
    }

    poolkit::Trace trace;
    try {
        trace = poolkit::parse_trace_file(path);
    } catch (const poolkit::TraceError& error) {
        std::fprintf(stderr, "record_check: FAILED to parse: %s\n", error.what());
        return 1;
    }

    std::vector<std::size_t> alloc_sizes;
    std::size_t allocs = 0;
    std::size_t frees = 0;
    for (const poolkit::TraceOp& op : trace.ops) {
        if (op.kind == poolkit::TraceOpKind::alloc) {
            ++allocs;
            alloc_sizes.push_back(op.size);
            if (op.size == kTinySize) {
                std::fprintf(stderr,
                             "record_check: FAILED: the %zu B allocation below --min-size is in "
                             "the trace\n",
                             kTinySize);
                return 1;
            }
        } else {
            ++frees;
        }
    }

    const std::size_t expected_count = sizeof(kExpectedSizes) / sizeof(kExpectedSizes[0]);

    if (injected) {
        // A context allocation matching one size would be a coincidence; three of them is the workload.
        for (std::size_t i = 0; i < expected_count; ++i) {
            if (std::find(alloc_sizes.begin(), alloc_sizes.end(), kExpectedSizes[i]) ==
                alloc_sizes.end()) {
                std::fprintf(stderr,
                             "record_check: FAILED: the injected trace has no %zu B alloc; the "
                             "workload's allocations are missing\n",
                             kExpectedSizes[i]);
                return 1;
            }
        }
        if (frees == 0) {
            std::fprintf(stderr, "record_check: FAILED: the injected trace has no free at all\n");
            return 1;
        }
    } else {
        if (allocs != expected_count || frees != expected_count) {
            std::fprintf(stderr, "record_check: FAILED: %zu allocs / %zu frees, expected %zu each\n",
                         allocs, frees, expected_count);
            return 1;
        }

        std::vector<std::size_t> expected(std::begin(kExpectedSizes), std::end(kExpectedSizes));
        std::vector<std::size_t> actual = alloc_sizes;
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        if (expected != actual) {
            std::fprintf(stderr, "record_check: FAILED: sizes differ from the workload's");
            for (std::size_t size : actual) {
                std::fprintf(stderr, " %zu", size);
            }
            std::fprintf(stderr, " (expected");
            for (std::size_t size : expected) {
                std::fprintf(stderr, " %zu", size);
            }
            std::fprintf(stderr, ")\n");
            return 1;
        }
    }

    // The pairing check: a free without a live alloc, a double free or a duplicate ptr throws.
    poolkit::SimBackend backend;
    poolkit::Pool pool(backend);
    poolkit::TraceReplayResult replay;
    try {
        replay = poolkit::replay_trace(pool, trace);
    } catch (const poolkit::TraceError& error) {
        std::fprintf(stderr, "record_check: FAILED to replay: %s\n", error.what());
        return 1;
    }
    if (replay.peak_bytes == 0 || replay.peak_live_blocks < expected_count) {
        std::fprintf(stderr, "record_check: FAILED: peak %zu B, %zu live blocks at the peak\n",
                     replay.peak_bytes, replay.peak_live_blocks);
        return 1;
    }

    std::printf("record_check: OK %zu ops, %zu allocs (%zu B, %zu B, %zu B), %zu frees, "
                "peak %zu B, %zu blocks\n",
                trace.ops.size(), allocs, kExpectedSizes[0], kExpectedSizes[1], kExpectedSizes[2],
                frees, replay.peak_bytes, replay.peak_live_blocks);
    return 0;
}

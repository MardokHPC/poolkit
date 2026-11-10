// Randomised stress test: 10 000 allocate/free with a fixed seed, checked against an independent
// model after every operation — overlap, double free, the resident identity, the request total,
// alignment. Under 2 s by design: it is a unit test, not a benchmark.

#include <poolkit/backend.hpp>
#include <poolkit/device_ptr.hpp>
#include <poolkit/metrics.hpp>
#include <poolkit/pool.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <map>
#include <utility>

namespace {

constexpr std::size_t kOperations = 10000;
constexpr std::uint64_t kSeed = 0x9E3779B97F4A7C15ull;

// xorshift64: the same bytes on every compiler, unlike std::uniform_int_distribution.
std::uint64_t next_random(std::uint64_t& state) noexcept {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// Mostly small blocks, a long tail, and 1 % above s_max to keep the large path in the loop.
std::size_t pick_size(std::uint64_t& state, std::size_t max_class_bytes) noexcept {
    const std::uint64_t roll = next_random(state) % 100;
    if (roll < 60) {
        return 256 + static_cast<std::size_t>(next_random(state) % 4096);
    }
    if (roll < 85) {
        return 256 + static_cast<std::size_t>(next_random(state) % (std::size_t{64} << 10));
    }
    if (roll < 95) {
        return 256 + static_cast<std::size_t>(next_random(state) % (std::size_t{1} << 20));
    }
    if (roll < 99) {
        return (std::size_t{1} << 20) + static_cast<std::size_t>(next_random(state) % (std::size_t{1} << 20));
    }
    return max_class_bytes + 1 +
           static_cast<std::size_t>(next_random(state) % (std::size_t{1} << 20));
}

struct LiveBlock {
    poolkit::DevicePtr ptr;
    std::size_t size = 0;
};

}  // namespace

TEST(Stress, TenThousandRandomOperationsKeepEveryInvariant) {
    poolkit::FakeBackend backend;
    // Events are not all ready: freed blocks spend a few operations pending, then poll promotes them.
    backend.set_events_complete_on_record(false);

    const auto start = std::chrono::steady_clock::now();
    {
        poolkit::Pool pool(backend);
        const std::size_t align = pool.config().min_class_bytes;
        const std::size_t max_class = pool.config().max_class_bytes;

        // begin -> end of every live block, maintained by the test alone.
        std::map<std::uintptr_t, std::uintptr_t> intervals;
        std::map<std::uintptr_t, LiveBlock> live;
        std::size_t requested_sum = 0;

        std::uint64_t state = kSeed;
        for (std::size_t step = 0; step < kOperations; ++step) {
            const bool do_alloc =
                live.empty() || (next_random(state) % 100 < 55 && live.size() < 700);

            if (do_alloc) {
                const std::size_t size = pick_size(state, max_class);
                poolkit::DevicePtr ptr = pool.allocate(size);
                ASSERT_TRUE(ptr.valid());

                const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(ptr.get());
                const std::uintptr_t end = begin + ptr.capacity();
                EXPECT_EQ(begin % align, std::size_t{0})
                    << "step " << step << ": block is not " << align << " B aligned";
                // A duplicate address is the failure an interval map sees and counters do not.
                const auto next = intervals.lower_bound(begin);
                if (next != intervals.end()) {
                    EXPECT_LE(end, next->first) << "step " << step << ": overlaps the next block";
                }
                if (next != intervals.begin()) {
                    EXPECT_LE(std::prev(next)->second, begin)
                        << "step " << step << ": overlaps the previous block";
                }
                const auto [it, inserted] = intervals.emplace(begin, end);
                ASSERT_TRUE(inserted) << "step " << step << ": block " << begin
                                      << " is already live";
                (void)it;

                requested_sum += size;
                const auto [live_it, live_inserted] =
                    live.emplace(begin, LiveBlock{std::move(ptr), size});
                ASSERT_TRUE(live_inserted);
                (void)live_it;
            } else {
                const std::size_t index =
                    static_cast<std::size_t>(next_random(state) % live.size());
                auto victim = live.begin();
                std::advance(victim, static_cast<std::ptrdiff_t>(index));

                const std::uintptr_t begin = victim->first;
                const std::size_t size = victim->second.size;
                // The victim must be live in the test's own map, or the test's bookkeeping is wrong.
                ASSERT_EQ(intervals.erase(begin), std::size_t{1}) << "step " << step;
                pool.free(std::move(victim->second.ptr));
                live.erase(victim);
                ASSERT_GE(requested_sum, size);
                requested_sum -= size;
            }

            // Keeps the pending list bounded and exercises the recover path.
            if (step % 13 == 0) {
                backend.make_all_events_ready();
                (void)pool.poll();
            }

            const poolkit::PoolSnapshot snap = pool.snapshot();

            EXPECT_EQ(snap.bytes_requested_live, requested_sum)
                << "step " << step << ": bytes_requested_live drifts";
            // The identity, byte for byte: nothing resident that is not live capacity, free, quarantined or tail.
            EXPECT_EQ(snap.bytes_resident,
                      snap.bytes_capacity_live + snap.bytes_free_in_pool +
                          snap.quarantined_bytes + snap.tail_bytes)
                << "step " << step << ": resident identity broken";
            EXPECT_EQ(snap.bytes_free_in_pool,
                      snap.bytes_free_ready + snap.bytes_free_pending + snap.bytes_unused_slots)
                << "step " << step << ": free-block split broken";
            EXPECT_GE(snap.bytes_capacity_live, snap.bytes_live_in_pool) << "step " << step;
            std::size_t quarantined_bytes_sum = 0;
            std::size_t quarantined_blocks_sum = 0;
            for (const poolkit::PoolSnapshot::ClassStats& cls : snap.classes) {
                quarantined_bytes_sum += cls.quarantined_blocks * cls.class_size;
                quarantined_blocks_sum += cls.quarantined_blocks;
            }
            EXPECT_EQ(snap.quarantined_bytes, quarantined_bytes_sum) << "step " << step;
            EXPECT_EQ(snap.quarantined_blocks, quarantined_blocks_sum) << "step " << step;
        }

        // Drain: everything the test still holds goes back through the ordinary path.
        for (auto& entry : live) {
            const std::size_t size = entry.second.size;
            ASSERT_EQ(intervals.erase(entry.first), std::size_t{1});
            pool.free(std::move(entry.second.ptr));
            requested_sum -= size;
        }
        live.clear();
        ASSERT_TRUE(intervals.empty());

        backend.make_all_events_ready();
        (void)pool.poll();
        const poolkit::PoolSnapshot end = pool.snapshot();
        EXPECT_EQ(requested_sum, std::size_t{0});
        EXPECT_EQ(end.bytes_requested_live, std::size_t{0});
        EXPECT_EQ(end.bytes_capacity_live, std::size_t{0});
        EXPECT_EQ(end.bytes_large_live, std::size_t{0});
        EXPECT_EQ(end.bytes_resident,
                  end.bytes_free_in_pool + end.quarantined_bytes + end.tail_bytes);
        EXPECT_EQ(backend.bytes_resident(), end.bytes_resident)
            << "the pool leaks chunks relative to what the backend thinks it owns";
    }

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count();
    std::cout << "[stress] " << kOperations << " operations, seed 0x" << std::hex << kSeed
              << std::dec << ", " << elapsed_ms << " ms\n";
    EXPECT_LT(elapsed_ms, 2000.0) << "the stress test is a unit test, not a benchmark";

    // Pool destroyed: every chunk it was allowed to return is back.
    EXPECT_EQ(backend.chunk_count(), std::size_t{0});
}

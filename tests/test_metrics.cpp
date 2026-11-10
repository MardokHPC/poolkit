// The three fragmentation components, the per-request one, hit rate and host timings, each on a state
// small enough to compute on paper. The worked example of docs/metrics.md is reproduced here: chunk
// 1000 B, classes {256, 512}, one live request per class.

#include <poolkit/metrics.hpp>
#include <poolkit/pool.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace {

// Two classes and a chunk that divides neither, so both a tail and an internal waste are non-zero
// and hand-checkable: 256 -> 3 blocks + 232 tail, 512 -> 1 block + 488 tail.
poolkit::Pool::Config two_class_config() {
    poolkit::Pool::Config cfg;
    cfg.chunk_bytes = 1000;
    cfg.min_class_bytes = 256;
    cfg.max_class_bytes = 512;
    cfg.growth = 2.0;
    return cfg;
}

poolkit::PoolSnapshot::ClassStats
make_class(std::size_t index, std::size_t class_size, std::size_t ready, std::size_t pending,
           std::size_t unused) {
    poolkit::PoolSnapshot::ClassStats stats;
    stats.index = index;
    stats.class_size = class_size;
    stats.ready_blocks = ready;
    stats.pending_blocks = pending;
    stats.unused_blocks = unused;
    return stats;
}

}  // namespace

TEST(Metrics, ThreeComponentsMatchHandComputation) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, two_class_config());
    ASSERT_EQ(pool.size_classes().count(), std::size_t{2});

    poolkit::DevicePtr small = pool.allocate(100);   // -> class 0 (256 B)
    poolkit::DevicePtr large = pool.allocate(300);   // -> class 1 (512 B)
    const poolkit::PoolSnapshot snap = pool.snapshot();

    EXPECT_EQ(snap.bytes_capacity_live, std::size_t{768});
    EXPECT_EQ(snap.bytes_live_in_pool, std::size_t{400});
    EXPECT_DOUBLE_EQ(snap.internal_fragmentation, 368.0 / 400.0);

    EXPECT_EQ(snap.bytes_resident, std::size_t{2000});
    EXPECT_EQ(snap.tail_bytes, std::size_t{720});
    EXPECT_DOUBLE_EQ(snap.tail_fragmentation, 720.0 / 2000.0);

    // slack: memory paid and not live. Class 0 has 2 unused slots (512 B), class
    // 1 is fully used (1 slot, live).
    EXPECT_EQ(snap.bytes_free_in_pool, std::size_t{512});
    EXPECT_DOUBLE_EQ(snap.resident_slack, 512.0 / 2000.0);
    EXPECT_DOUBLE_EQ(snap.free_ratio, snap.resident_slack);

    // The identity the three components have to add up to, byte for byte.
    EXPECT_EQ(snap.bytes_resident,
              snap.bytes_capacity_live + snap.bytes_free_in_pool + snap.tail_bytes);

    ASSERT_EQ(snap.classes.size(), std::size_t{2});
    EXPECT_EQ(snap.classes[0].class_size, std::size_t{256});
    EXPECT_EQ(snap.classes[0].live_blocks, std::size_t{1});
    EXPECT_EQ(snap.classes[0].unused_blocks, std::size_t{2});
    EXPECT_EQ(snap.classes[0].tail_bytes_per_slab, std::size_t{232});
    EXPECT_EQ(snap.classes[1].class_size, std::size_t{512});
    EXPECT_EQ(snap.classes[1].live_blocks, std::size_t{1});
    EXPECT_EQ(snap.classes[1].unused_blocks, std::size_t{0});
    EXPECT_EQ(snap.classes[1].tail_bytes_per_slab, std::size_t{488});
}

TEST(Metrics, ExternalFragmentationDependsOnTheRequest) {
    // Built by hand: the property under test is arithmetic on the snapshot, not the pool.
    poolkit::PoolSnapshot snap;
    snap.classes.push_back(make_class(0, 256, 4, 0, 0));
    snap.classes.push_back(make_class(1, 512, 0, 0, 0));
    snap.bytes_free_in_pool = 1024;
    snap.bytes_free_ready = 1024;

    EXPECT_EQ(snap.usable_free(256), std::size_t{1024});
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(256), 0.0);

    // A 512-byte request cannot: all the free memory is in the wrong class.
    EXPECT_EQ(snap.usable_free(512), std::size_t{0});
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(512), 1.0);

    EXPECT_NE(snap.external_fragmentation(256), snap.external_fragmentation(512));

    snap.bytes_free_in_pool = 0;
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(512), 0.0);

    snap.classes[0].ready_blocks = 0;
    snap.classes[0].pending_blocks = 4;
    snap.bytes_free_in_pool = 1024;
    EXPECT_EQ(snap.usable_free(256), std::size_t{0});
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(256), 1.0);
}

TEST(Metrics, FreeMemoryInABiggerClassIsNotUsableForASmallerRequest) {
    // The pool maps a request to exactly one class and never borrows from a larger one, so summing over
    // "every class >= request" would report 0.0 where the honest answer is 1.0.
    poolkit::PoolSnapshot snap;
    snap.classes.push_back(make_class(0, 256, 0, 0, 0));   // empty
    snap.classes.push_back(make_class(1, 512, 2, 0, 0));   // 2 ready blocks
    snap.bytes_free_in_pool = 1024;
    snap.bytes_free_ready = 1024;

    EXPECT_EQ(snap.usable_free(256), std::size_t{0});
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(256), 1.0);

    // The class that is free serves its own request, which makes the 1.0 above a statement about the class.
    EXPECT_EQ(snap.usable_free(512), std::size_t{1024});
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(512), 0.0);

    // Above the largest class the pool takes the large path: no class serves it, so no usable free memory.
    EXPECT_EQ(snap.usable_free(1024), std::size_t{0});
}

TEST(Metrics, UnusedSlotsCountAsFreeAndAsUsable) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, two_class_config());

    poolkit::DevicePtr ptr = pool.allocate(100);
    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_free_in_pool, std::size_t{512});
    EXPECT_EQ(snap.bytes_unused_slots, std::size_t{512});
    EXPECT_EQ(snap.usable_free(100), std::size_t{512});
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(100), 0.0);
    EXPECT_EQ(snap.usable_free(512), std::size_t{0});
}

TEST(Metrics, WorkedExampleOfTheDocumentation) {
    // docs/metrics.md §4, verbatim: chunk 1000, classes {256, 512}, one live request per class, then
    // free the small one.
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, two_class_config());

    poolkit::DevicePtr small = pool.allocate(100);
    poolkit::DevicePtr large = pool.allocate(300);
    pool.free(std::move(small));

    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_free_in_pool, std::size_t{768});
    EXPECT_EQ(snap.bytes_free_pending, std::size_t{256});
    EXPECT_EQ(snap.bytes_unused_slots, std::size_t{512});
    EXPECT_EQ(snap.usable_free(256), std::size_t{512});
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(256), 1.0 - 512.0 / 768.0);
    EXPECT_EQ(snap.usable_free(512), std::size_t{0});
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(512), 1.0);
}

TEST(Metrics, HitRateAndPerClassStatsFollowTheAllocationOutcome) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, two_class_config());

    poolkit::DevicePtr first = pool.allocate(100);
    pool.free(std::move(first));
    // The fake stream is idle, so the next allocation of the same class promotes the pending block.
    poolkit::DevicePtr second = pool.allocate(100);

    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.growth_count, std::size_t{1});
    EXPECT_EQ(snap.reuse_count, std::size_t{1});
    EXPECT_DOUBLE_EQ(snap.hit_rate, 0.5);
    ASSERT_GE(snap.classes.size(), std::size_t{1});
    EXPECT_EQ(snap.classes[0].alloc_count, std::size_t{2});
    EXPECT_EQ(snap.classes[0].reuse_count, std::size_t{1});
    EXPECT_DOUBLE_EQ(snap.classes[0].hit_rate, 0.5);
    EXPECT_DOUBLE_EQ(snap.classes[1].hit_rate, 0.0);  // never used
    EXPECT_EQ(snap.recovered_count, std::size_t{1});
}

TEST(Metrics, HighWaterMarkIsMonotoneAndSurvivesReleaseAll) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, two_class_config());

    EXPECT_EQ(pool.snapshot().high_water_mark_bytes, std::size_t{0});

    poolkit::DevicePtr a = pool.allocate(100);
    const std::size_t first_peak = pool.snapshot().high_water_mark_bytes;
    EXPECT_EQ(first_peak, std::size_t{1000});

    poolkit::DevicePtr b = pool.allocate(300);
    const std::size_t second_peak = pool.snapshot().high_water_mark_bytes;
    EXPECT_EQ(second_peak, std::size_t{2000});
    EXPECT_GT(second_peak, first_peak);

    pool.free(std::move(a));
    pool.free(std::move(b));
    EXPECT_EQ(pool.snapshot().high_water_mark_bytes, second_peak);

    pool.release_all();
    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_resident, std::size_t{0});
    EXPECT_EQ(snap.high_water_mark_bytes, second_peak);
    // Every ratio is 0 when its denominator is 0, never NaN.
    EXPECT_DOUBLE_EQ(snap.internal_fragmentation, 0.0);
    EXPECT_DOUBLE_EQ(snap.tail_fragmentation, 0.0);
    EXPECT_DOUBLE_EQ(snap.resident_slack, 0.0);
    EXPECT_DOUBLE_EQ(snap.hit_rate, 0.0);
}

TEST(Metrics, HostTimingsArePopulated) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, two_class_config());

    std::vector<poolkit::DevicePtr> live;
    for (int i = 0; i < 100; ++i) {
        live.push_back(pool.allocate(100));
    }
    for (auto& ptr : live) {
        pool.free(std::move(ptr));
    }

    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_GT(snap.alloc_max_ns, 0.0);
    EXPECT_GE(snap.alloc_max_ns, snap.alloc_avg_ns);
    EXPECT_GT(snap.free_max_ns, 0.0);
    EXPECT_GE(snap.free_max_ns, snap.free_avg_ns);
}

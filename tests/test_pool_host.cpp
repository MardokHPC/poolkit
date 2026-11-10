// Pool behaviour on the host, with FakeBackend standing in for the driver: the
// contract M4 has to keep once cudaMalloc replaces the fake. Every number asserted
// is a definition from docs/metrics.md or an integer this test can add up itself.

#include <poolkit/backend.hpp>
#include <poolkit/device_ptr.hpp>
#include <poolkit/metrics.hpp>
#include <poolkit/pool.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

std::uintptr_t address(const void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr);
}

// No two live blocks may overlap: catches slab arithmetic a counter check cannot see.
class IntervalSet {
public:
    void add(std::uintptr_t begin, std::size_t size) {
        const std::uintptr_t end = begin + size;
        const auto next = intervals_.lower_bound(begin);
        if (next != intervals_.end()) {
            EXPECT_LE(end, next->first) << "overlaps the block that starts after it";
        }
        if (next != intervals_.begin()) {
            const auto previous = std::prev(next);
            EXPECT_LE(previous->second, begin) << "overlaps the block that starts before it";
        }
        intervals_.emplace(begin, end);
    }

    void remove(std::uintptr_t begin) { intervals_.erase(begin); }

    std::size_t count() const { return intervals_.size(); }

private:
    std::map<std::uintptr_t, std::uintptr_t> intervals_;
};

poolkit::Pool::Config small_chunk_config() {
    poolkit::Pool::Config cfg;
    cfg.chunk_bytes = 4096;
    return cfg;
}

bool g_fail_host_allocations = false;

// Throws where the CUDA backend allocates: the shape of bad_alloc inside free().
class ThrowingEventBackend final : public poolkit::FakeBackend {
public:
    void* record_event(poolkit::StreamHandle) override { throw std::bad_alloc(); }
};

class ThrowingAllocBackend final : public poolkit::FakeBackend {
public:
    void* allocate_raw(std::size_t) override { throw std::bad_alloc(); }
};

// Arms the host allocator as it hands over a chunk, so the pool's own bookkeeping is what fails; the allocation that must still succeed comes later.
class ArmOnChunkBackend final : public poolkit::FakeBackend {
public:
    void* allocate_raw(std::size_t bytes) override {
        void* ptr = FakeBackend::allocate_raw(bytes);
        if (ptr != nullptr && !armed_) {
            armed_ = true;
            g_fail_host_allocations = true;
        }
        return ptr;
    }

private:
    bool armed_ = false;
};

class DisarmOnExit {
public:
    // Scope exit, including the path where an ASSERT returns early: a leaked arming would fail every later test.
    DisarmOnExit() = default;
    ~DisarmOnExit() { g_fail_host_allocations = false; }
    DisarmOnExit(const DisarmOnExit&) = delete;
    DisarmOnExit& operator=(const DisarmOnExit&) = delete;
};

}  // namespace

// Replacing the host allocator is the only injection point that reaches a vector growth after the
// chunk exists, without a fault hook in a shipped header.
void* operator new(std::size_t size) {
    if (g_fail_host_allocations) {
        throw std::bad_alloc();
    }
    void* ptr = std::malloc(size == 0 ? 1 : size);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }

TEST(PoolHost, EveryBlockIsAlignedToTheAlignmentFloor) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend);

    std::vector<poolkit::DevicePtr> live;
    live.reserve(600);
    for (std::size_t i = 1; i <= 600; ++i) {
        const std::size_t size = 1 + (i * 4093) % 65536;
        poolkit::DevicePtr ptr = pool.allocate(size);
        ASSERT_TRUE(ptr.valid()) << "iteration " << i;
        EXPECT_EQ(address(ptr.get()) % 256, std::uintptr_t{0})
            << "size " << size << " class " << pool.size_classes().index_for(size);
        EXPECT_GE(ptr.capacity(), ptr.size());
        EXPECT_EQ(ptr.capacity() % 256, std::size_t{0});
        live.push_back(std::move(ptr));
    }
    EXPECT_EQ(live.size(), std::size_t{600});
}

TEST(PoolHost, AccountingMatchesTheDefinitions) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, small_chunk_config());

    std::vector<poolkit::DevicePtr> live;
    std::vector<std::size_t> sizes{100, 300, 1000, 5000, 20000, 250};
    for (const std::size_t size : sizes) {
        live.push_back(pool.allocate(size));
    }

    pool.free(std::move(live[1]));
    pool.free(std::move(live[4]));

    const poolkit::PoolSnapshot snap = pool.snapshot();

    std::size_t expected_live = 0;
    for (const auto& ptr : live) {
        if (ptr.valid()) {
            expected_live += ptr.size();
        }
    }
    EXPECT_EQ(snap.bytes_requested_live, expected_live);
    EXPECT_EQ(snap.bytes_live_in_pool, expected_live);
    EXPECT_EQ(snap.bytes_large_live, std::size_t{0});

    EXPECT_EQ(snap.bytes_resident, backend.bytes_resident());
    EXPECT_EQ(snap.bytes_resident, pool.capacity());

    // resident = capacity_live + free_in_pool + tail
    EXPECT_EQ(snap.bytes_resident,
              snap.bytes_capacity_live + snap.bytes_free_in_pool + snap.tail_bytes);
    EXPECT_GE(snap.bytes_capacity_live, snap.bytes_live_in_pool);
    EXPECT_EQ(snap.bytes_free_in_pool,
              snap.bytes_free_ready + snap.bytes_free_pending + snap.bytes_unused_slots);
    EXPECT_EQ(snap.chunk_count, backend.chunk_count());
    EXPECT_LE(snap.bytes_resident, snap.high_water_mark_bytes);
}

TEST(PoolHost, ActiveBlocksNeverOverlap) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend);

    IntervalSet intervals;
    std::vector<poolkit::DevicePtr> live;

    // Reuse is where a carving bug shows up: a block handed out twice, or at a stale class size.
    for (std::size_t i = 1; i <= 300; ++i) {
        const std::size_t size = 1 + (i * 7919) % 40000;
        live.push_back(pool.allocate(size));
        intervals.add(address(live.back().get()), live.back().capacity());
        if (i % 3 == 0) {
            intervals.remove(address(live.back().get()));
            pool.free(std::move(live.back()));
        }
    }
    for (std::size_t i = 0; i < live.size(); ++i) {
        if (live[i].valid()) {
            intervals.remove(address(live[i].get()));
            pool.free(std::move(live[i]));
        }
    }
    EXPECT_EQ(intervals.count(), std::size_t{0});
}

TEST(PoolHost, AFreedBlockIsNotReusedUntilItsEventCompletes) {
    poolkit::FakeBackend backend;
    backend.set_events_complete_on_record(false);  // the stream still has work
    poolkit::Pool pool(backend, small_chunk_config());

    poolkit::DevicePtr first = pool.allocate(100);
    EXPECT_EQ(pool.snapshot().growth_count, std::size_t{1});
    EXPECT_EQ(pool.snapshot().reuse_count, std::size_t{0});

    pool.free(std::move(first));
    const poolkit::PoolSnapshot after_free = pool.snapshot();
    EXPECT_EQ(after_free.bytes_free_pending, std::size_t{256});
    EXPECT_EQ(after_free.bytes_free_ready, std::size_t{0});

    // Same class but pending: the pool takes a never-used slot, so growth, not reuse.
    poolkit::DevicePtr second = pool.allocate(100);
    EXPECT_EQ(pool.snapshot().growth_count, std::size_t{2});
    EXPECT_EQ(pool.snapshot().reuse_count, std::size_t{0});

    backend.make_ready(backend.last_event());
    poolkit::DevicePtr third = pool.allocate(100);
    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.reuse_count, std::size_t{1});
    EXPECT_EQ(snap.growth_count, std::size_t{2});
    EXPECT_EQ(snap.recovered_count, std::size_t{1});
    EXPECT_EQ(backend.chunk_count(), std::size_t{1});  // one slab, not three
}

TEST(PoolHost, PendingBlocksAreNotCountedAsUsable) {
    poolkit::FakeBackend backend;
    backend.set_events_complete_on_record(false);
    poolkit::Pool pool(backend, small_chunk_config());

    poolkit::DevicePtr ptr = pool.allocate(100);
    pool.free(std::move(ptr));

    // 4096 B = 16 slots: one pending, 3840 B usable.
    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_free_in_pool, std::size_t{4096});
    EXPECT_EQ(snap.bytes_free_pending, std::size_t{256});
    EXPECT_EQ(snap.bytes_free_ready, std::size_t{0});
    EXPECT_EQ(snap.usable_free(100), std::size_t{3840});
    EXPECT_DOUBLE_EQ(snap.external_fragmentation(100), 256.0 / 4096.0);
}

TEST(PoolHost, OomThrowsLeavesNoLeakAndThePoolKeepsWorking) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend);

    poolkit::DevicePtr small = pool.allocate(1000);
    const std::size_t resident_before = pool.snapshot().bytes_resident;

    backend.fail_next_alloc();
    try {
        (void)pool.allocate(600000);
        FAIL() << "expected PoolError";
    } catch (const poolkit::PoolError& error) {
        const std::string message = error.what();
        // The message must carry the request, the class, and what is already resident.
        EXPECT_NE(message.find("600000"), std::string::npos) << message;
        EXPECT_NE(message.find("class"), std::string::npos) << message;
        EXPECT_NE(message.find("resident"), std::string::npos) << message;
    }

    EXPECT_EQ(pool.snapshot().bytes_resident, resident_before);
    EXPECT_EQ(pool.snapshot().bytes_requested_live, std::size_t{1000});

    poolkit::DevicePtr other = pool.allocate(200);
    poolkit::DevicePtr retry = pool.allocate(600000);  // the backend works again
    EXPECT_TRUE(retry.valid());

    pool.free(std::move(small));
    pool.free(std::move(other));
    pool.free(std::move(retry));
    EXPECT_EQ(pool.snapshot().bytes_requested_live, std::size_t{0});
}

TEST(PoolHost, MaxBytesRefusesTheChunkInsteadOfOvercommitting) {
    poolkit::FakeBackend backend;
    poolkit::Pool::Config cfg;
    cfg.chunk_bytes = 4096;
    cfg.max_bytes = 16 * 1024;
    poolkit::Pool pool(backend, cfg);

    std::vector<poolkit::DevicePtr> live;
    live.push_back(pool.allocate(256));
    live.push_back(pool.allocate(1000));
    live.push_back(pool.allocate(5000));
    EXPECT_LE(pool.snapshot().bytes_resident, cfg.max_bytes);
    EXPECT_THROW((void)pool.allocate(50000), poolkit::PoolError);
    EXPECT_LE(pool.snapshot().bytes_resident, cfg.max_bytes);
}

TEST(PoolHost, LargePathBypassesThePoolAndIsReleasedForReal) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend);
    const std::size_t big = pool.config().max_class_bytes + 1;

    poolkit::DevicePtr ptr = pool.allocate(big);
    ASSERT_TRUE(ptr.valid());
    EXPECT_EQ(ptr.capacity(), ptr.size());  // the large path does not round

    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.large_path_count, std::size_t{1});
    EXPECT_EQ(snap.bytes_resident, std::size_t{0});  // no chunk was carved
    EXPECT_EQ(snap.bytes_large_live, big);
    EXPECT_EQ(snap.bytes_requested_live, big);
    EXPECT_EQ(backend.chunk_count(), std::size_t{1});
    EXPECT_DOUBLE_EQ(snap.internal_fragmentation, 0.0);

    pool.free(std::move(ptr));
    const poolkit::PoolSnapshot after = pool.snapshot();
    EXPECT_EQ(after.bytes_requested_live, std::size_t{0});
    EXPECT_EQ(after.bytes_large_live, std::size_t{0});
    EXPECT_EQ(backend.chunk_count(), std::size_t{0});  // really given back
}

static_assert(!std::is_copy_constructible<poolkit::DevicePtr>::value,
              "a copied DevicePtr is a double free waiting for a scope exit");
static_assert(!std::is_copy_assignable<poolkit::DevicePtr>::value,
              "a copied DevicePtr is a double free waiting for a scope exit");
static_assert(std::is_move_constructible<poolkit::DevicePtr>::value,
              "DevicePtr must be movable");
static_assert(std::is_move_assignable<poolkit::DevicePtr>::value,
              "DevicePtr must be move-assignable");
static_assert(!std::is_move_constructible<poolkit::Pool>::value, "Pool is deliberately pinned");
static_assert(!std::is_copy_constructible<poolkit::Pool>::value, "Pool owns its chunks");

TEST(PoolHost, MovingAHandleTransfersItExactlyOnce) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, small_chunk_config());

    poolkit::DevicePtr first = pool.allocate(100);
    void* raw = first.get();
    EXPECT_TRUE(first.valid());

    poolkit::DevicePtr second = std::move(first);
    EXPECT_FALSE(first.valid()) << "the moved-from handle must be empty";
    EXPECT_EQ(second.get(), raw);
    EXPECT_EQ(pool.snapshot().bytes_requested_live, std::size_t{100});

    // Destroying the moved-from handle must not release: the live count would drop twice.
    {
        poolkit::DevicePtr third = std::move(second);
        EXPECT_FALSE(second.valid());
        EXPECT_EQ(pool.snapshot().bytes_requested_live, std::size_t{100});
    }
    EXPECT_EQ(pool.snapshot().bytes_requested_live, std::size_t{0});

    poolkit::DevicePtr a = pool.allocate(200);
    poolkit::DevicePtr b = pool.allocate(300);
    EXPECT_EQ(pool.snapshot().bytes_requested_live, std::size_t{500});
    a = std::move(b);
    EXPECT_FALSE(b.valid());
    EXPECT_EQ(pool.snapshot().bytes_requested_live, std::size_t{300});
}

TEST(PoolHost, ExplicitReleaseMatchesTheDestructor) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, small_chunk_config());

    poolkit::DevicePtr ptr = pool.allocate(100);
    ptr.release(0);
    EXPECT_FALSE(ptr.valid());
    EXPECT_EQ(pool.snapshot().bytes_requested_live, std::size_t{0});

    ptr.release(0);
    EXPECT_EQ(pool.snapshot().free_errors, std::size_t{0});
    EXPECT_EQ(backend.invalid_frees(), std::size_t{0});
}

TEST(PoolHost, FreeFailureQuarantinesTheBlockInsteadOfReusingIt) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, small_chunk_config());  // 4096 B chunks

    poolkit::DevicePtr first = pool.allocate(100);  // class 0: 16 slots of 256 B
    backend.fail_next_event();
    pool.free(std::move(first));  // the event could not be recorded

    poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.free_errors, std::size_t{1});
    EXPECT_EQ(snap.quarantined_blocks, std::size_t{1});
    EXPECT_EQ(snap.quarantined_bytes, std::size_t{256});
    // Fail-closed: neither ready nor pending, so it is never handed out again.
    EXPECT_EQ(snap.bytes_free_ready, std::size_t{0});
    EXPECT_EQ(snap.bytes_free_pending, std::size_t{0});

    std::vector<poolkit::DevicePtr> live;
    for (int i = 0; i < 3; ++i) {
        live.push_back(pool.allocate(100));
    }
    snap = pool.snapshot();
    EXPECT_EQ(snap.reuse_count, std::size_t{0});
    EXPECT_EQ(snap.growth_count, std::size_t{4});  // the quarantined one + these three
    EXPECT_EQ(snap.quarantined_bytes, std::size_t{256});

    poolkit::DevicePtr other = pool.allocate(300);  // class 512, own 4096 B chunk
    for (auto& ptr : live) {
        pool.free(std::move(ptr));
    }
    pool.free(std::move(other));
    EXPECT_EQ(pool.snapshot().bytes_resident, std::size_t{8192});

    // That chunk is kept: no event is left to prove nothing is still in flight.
    EXPECT_EQ(pool.release_all(), std::size_t{4096});
    snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_resident, std::size_t{4096});
    EXPECT_EQ(snap.quarantined_bytes, std::size_t{256});
    EXPECT_EQ(pool.release_all(), std::size_t{0});
    EXPECT_EQ(backend.chunk_count(), std::size_t{1});
}

TEST(PoolHost, ReleaseAllRefusesWhileAHandleIsAlive) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, small_chunk_config());

    poolkit::DevicePtr live_ptr = pool.allocate(1000);
    const std::size_t resident = pool.snapshot().bytes_resident;
    ASSERT_GT(resident, std::size_t{0});

    EXPECT_EQ(pool.release_all(), std::size_t{0});
    poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.release_all_refused, std::size_t{1});
    EXPECT_EQ(snap.bytes_resident, resident);
    EXPECT_EQ(backend.chunk_count(), std::size_t{1});
    EXPECT_TRUE(live_ptr.valid());

    poolkit::DevicePtr big = pool.allocate(pool.config().max_class_bytes + 1);
    EXPECT_EQ(pool.release_all(), std::size_t{0});
    EXPECT_EQ(pool.snapshot().release_all_refused, std::size_t{2});

    pool.free(std::move(big));
    pool.free(std::move(live_ptr));
    EXPECT_EQ(pool.poll(), std::size_t{1});
    EXPECT_EQ(pool.release_all(), resident);
    snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_resident, std::size_t{0});
    EXPECT_EQ(backend.chunk_count(), std::size_t{0});
}

TEST(PoolHost, PollPromotesCompletedEventsAndReportsHowMany) {
    poolkit::FakeBackend backend;
    backend.set_events_complete_on_record(false);
    poolkit::Pool pool(backend, small_chunk_config());

    poolkit::DevicePtr first = pool.allocate(100);
    poolkit::DevicePtr second = pool.allocate(100);
    pool.free(std::move(first));
    pool.free(std::move(second));

    EXPECT_EQ(pool.snapshot().bytes_free_pending, std::size_t{512});
    EXPECT_EQ(pool.snapshot().bytes_free_ready, std::size_t{0});
    EXPECT_EQ(pool.poll(), std::size_t{0});  // nothing has completed
    EXPECT_EQ(pool.snapshot().bytes_free_ready, std::size_t{0});

    backend.make_all_events_ready();
    EXPECT_EQ(pool.poll(), std::size_t{2});
    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_free_ready, std::size_t{512});
    EXPECT_EQ(snap.bytes_free_pending, std::size_t{0});
    EXPECT_EQ(pool.poll(), std::size_t{0});  // idempotent
}

TEST(PoolHost, ReleaseAllReturnsEveryChunkToTheDriver) {
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, small_chunk_config());

    std::vector<poolkit::DevicePtr> live;
    for (const std::size_t size : {100, 400, 1200, 9000, 30000}) {
        live.push_back(pool.allocate(size));
    }
    EXPECT_GT(pool.snapshot().bytes_resident, std::size_t{0});
    EXPECT_GT(backend.chunk_count(), std::size_t{0});
    const std::size_t high_water = pool.snapshot().high_water_mark_bytes;

    for (auto& ptr : live) {
        pool.free(std::move(ptr));
    }
    // Freed blocks stay resident on purpose: that is the whole point of the pool.
    EXPECT_GT(pool.snapshot().bytes_resident, std::size_t{0});

    pool.release_all();
    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_resident, std::size_t{0});
    EXPECT_EQ(snap.chunk_count, std::size_t{0});
    EXPECT_EQ(backend.chunk_count(), std::size_t{0});
    EXPECT_EQ(backend.bytes_resident(), std::size_t{0});
    EXPECT_EQ(pool.capacity(), std::size_t{0});
    EXPECT_EQ(snap.high_water_mark_bytes, high_water);
}

TEST(PoolHost, DestructorReturnsEveryChunkToTheDriver) {
    poolkit::FakeBackend backend;
    {
        poolkit::Pool pool(backend, small_chunk_config());
        std::vector<poolkit::DevicePtr> live;
        live.push_back(pool.allocate(100));
        live.push_back(pool.allocate(1000));
        for (auto& ptr : live) {
            pool.free(std::move(ptr));
        }
    }
    EXPECT_EQ(backend.chunk_count(), std::size_t{0});
    EXPECT_EQ(backend.invalid_frees(), std::size_t{0});
}

TEST(PoolHost, ReleaseAllKeepsAChunkWhoseFreeEventIsStillPending) {
    poolkit::FakeBackend backend;
    backend.set_events_complete_on_record(false);  // a stream still has work queued
    poolkit::Pool pool(backend, small_chunk_config());  // 4096 B chunks

    poolkit::DevicePtr pending = pool.allocate(100);  // class 256: chunk 0
    poolkit::DevicePtr completed = pool.allocate(300);  // class 512: chunk 1
    ASSERT_EQ(backend.chunk_count(), std::size_t{2});
    const std::size_t resident = pool.capacity();
    ASSERT_EQ(resident, std::size_t{8192});

    pool.free(std::move(pending));  // its event stays pending
    pool.free(std::move(completed));
    backend.make_ready(backend.last_event());  // the second one has completed
    ASSERT_EQ(backend.events_created(), std::size_t{2});

    // Fail-closed: the chunk holding the in-flight block stays, event and all.
    EXPECT_EQ(pool.release_all(), std::size_t{4096});
    EXPECT_EQ(pool.capacity(), resident - 4096);
    EXPECT_EQ(pool.capacity(), backend.bytes_resident());
    EXPECT_EQ(backend.free_calls(), std::size_t{1});      // exactly one chunk returned
    EXPECT_EQ(backend.events_created(), std::size_t{1});  // the pending event survives
    poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_resident, std::size_t{4096});
    EXPECT_EQ(snap.chunk_count, std::size_t{1});
    EXPECT_EQ(snap.bytes_free_pending, std::size_t{256});

    // Nothing is live now: the event is destroyed only after it was queried.
    backend.make_all_events_ready();
    EXPECT_EQ(pool.release_all(), std::size_t{4096});
    EXPECT_EQ(pool.capacity(), std::size_t{0});
    EXPECT_EQ(backend.bytes_resident(), std::size_t{0});
    EXPECT_EQ(backend.chunk_count(), std::size_t{0});
    EXPECT_EQ(backend.events_created(), std::size_t{0});
}

TEST(PoolHost, FreeQuarantinesTheBlockWhenTheBackendThrows) {
    ThrowingEventBackend backend;
    poolkit::Pool pool(backend, small_chunk_config());  // 4096 B = 16 x 256 B

    poolkit::DevicePtr first = pool.allocate(100);
    ASSERT_TRUE(first.valid());

    // free() is noexcept: bad_alloc must not become terminate, and must not be swallowed either.
    pool.free(std::move(first));

    poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.free_errors, std::size_t{1});
    EXPECT_EQ(snap.quarantined_blocks, std::size_t{1});
    EXPECT_EQ(snap.quarantined_bytes, std::size_t{256});
    EXPECT_EQ(snap.bytes_free_pending, std::size_t{0});
    EXPECT_EQ(snap.bytes_free_ready, std::size_t{0});
    EXPECT_EQ(snap.bytes_requested_live, std::size_t{0});
    EXPECT_EQ(snap.bytes_live_in_pool, std::size_t{0});

    // The quarantined slot never comes back: the 15 remaining slots are growth.
    std::vector<poolkit::DevicePtr> live;
    for (int i = 0; i < 15; ++i) {
        live.push_back(pool.allocate(100));
        ASSERT_TRUE(live.back().valid()) << "slot " << i;
    }
    snap = pool.snapshot();
    EXPECT_EQ(snap.reuse_count, std::size_t{0});
    EXPECT_EQ(snap.growth_count, std::size_t{16});  // 1 quarantined + 15 unused
    EXPECT_EQ(backend.chunk_count(), std::size_t{1});

    live.push_back(pool.allocate(100));
    snap = pool.snapshot();
    EXPECT_EQ(backend.chunk_count(), std::size_t{2});  // grew, did not recycle
    EXPECT_EQ(snap.reuse_count, std::size_t{0});
    EXPECT_EQ(snap.growth_count, std::size_t{17});
}

TEST(PoolHost, AllocateLeavesNoTraceWhenTheBackendThrowsBadAlloc) {
    ThrowingAllocBackend backend;
    poolkit::Pool pool(backend, small_chunk_config());

    EXPECT_THROW((void)pool.allocate(100), std::bad_alloc);

    const poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_resident, std::size_t{0});
    EXPECT_EQ(snap.chunk_count, std::size_t{0});
    EXPECT_EQ(snap.high_water_mark_bytes, std::size_t{0});
    EXPECT_EQ(pool.capacity(), std::size_t{0});
    EXPECT_EQ(backend.chunk_count(), std::size_t{0});

    // Must not subtract bytes never added: an unsigned underflow would report ~2^64 B.
    EXPECT_EQ(pool.release_all(), std::size_t{0});
    EXPECT_EQ(pool.capacity(), std::size_t{0});
    EXPECT_EQ(pool.snapshot().bytes_resident, std::size_t{0});
}

TEST(PoolHost, AddSlabReturnsTheChunkWhenTheInsertionAfterItThrows) {
    // The chunk is obtained first and the vectors grow after, so the failure is injected
    // at the host allocator: that is the only way to make the second step throw.
    ArmOnChunkBackend backend;
    poolkit::Pool pool(backend, small_chunk_config());  // 4096 B chunks

    bool threw = false;
    {
        DisarmOnExit disarm;
        try {
            (void)pool.allocate(256);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
    }
    ASSERT_TRUE(threw) << "the injection did not fire where the test expects it";

    // Both halves of "no leak": the chunk is back and the pool does not think it holds it.
    EXPECT_EQ(backend.chunk_count(), std::size_t{0});
    EXPECT_EQ(backend.bytes_resident(), std::size_t{0});
    EXPECT_EQ(backend.free_calls(), std::size_t{1});
    EXPECT_EQ(pool.capacity(), std::size_t{0});
    poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.bytes_resident, std::size_t{0});
    EXPECT_EQ(snap.chunk_count, std::size_t{0});
    EXPECT_EQ(snap.high_water_mark_bytes, std::size_t{0});
    EXPECT_EQ(snap.bytes_requested_live, std::size_t{0});

    EXPECT_EQ(pool.release_all(), std::size_t{0});
    EXPECT_EQ(pool.capacity(), std::size_t{0});
    EXPECT_EQ(pool.snapshot().bytes_resident, std::size_t{0});

    poolkit::DevicePtr small = pool.allocate(256);
    ASSERT_TRUE(small.valid());
    EXPECT_EQ(pool.capacity(), std::size_t{4096});
    EXPECT_EQ(backend.chunk_count(), std::size_t{1});

    // Same schema on the large path: large_.push_back throws and the chunk is not lost.
    ArmOnChunkBackend large_backend;
    poolkit::Pool large_pool(large_backend);
    bool large_threw = false;
    {
        DisarmOnExit disarm;
        try {
            (void)large_pool.allocate(large_pool.config().max_class_bytes + 1);
        } catch (const std::bad_alloc&) {
            large_threw = true;
        }
    }
    ASSERT_TRUE(large_threw);
    EXPECT_EQ(large_backend.chunk_count(), std::size_t{0});
    EXPECT_EQ(large_backend.bytes_resident(), std::size_t{0});
    snap = large_pool.snapshot();
    EXPECT_EQ(snap.large_path_count, std::size_t{0});
    EXPECT_EQ(snap.bytes_large_live, std::size_t{0});
    EXPECT_EQ(large_pool.release_all(), std::size_t{0});
    EXPECT_EQ(large_pool.capacity(), std::size_t{0});
}

TEST(PoolHost, MaxBytesAppliesToTheLargePathToo) {
    poolkit::FakeBackend backend;
    poolkit::Pool::Config cfg;
    cfg.max_bytes = std::size_t{64} << 20;  // 64 MiB, well above one chunk
    poolkit::Pool pool(backend, cfg);

    // The headline case: only add_slab() consulted max_bytes, so a 4 GiB request sailed past.
    const std::size_t big = std::size_t{4} << 30;
    bool threw = false;
    try {
        (void)pool.allocate(big);
    } catch (const poolkit::PoolError& error) {
        threw = true;
        const std::string message = error.what();
        // It must say which limit was hit: the driver did not refuse this one.
        EXPECT_NE(message.find("4294967296"), std::string::npos) << message;
        EXPECT_NE(message.find("max_bytes"), std::string::npos) << message;
        EXPECT_NE(message.find("ceiling"), std::string::npos) << message;
        EXPECT_NE(message.find("resident"), std::string::npos) << message;
    }
    ASSERT_TRUE(threw) << "a 4 GiB request under a 64 MiB ceiling was accepted";

    // The ceiling is decided above the backend: the driver was never asked.
    EXPECT_EQ(backend.alloc_calls(), std::size_t{0});
    EXPECT_EQ(backend.bytes_resident(), std::size_t{0});
    poolkit::PoolSnapshot snap = pool.snapshot();
    EXPECT_EQ(snap.large_path_count, std::size_t{0});
    EXPECT_EQ(snap.bytes_large_live, std::size_t{0});
    EXPECT_EQ(snap.bytes_resident, std::size_t{0});

    // The ceiling counts what the large path already holds, not only carved chunks.
    const std::size_t half = std::size_t{32} << 20;
    poolkit::DevicePtr live_large = pool.allocate(half);
    ASSERT_TRUE(live_large.valid());
    EXPECT_EQ(pool.snapshot().bytes_large_live, half);
    EXPECT_THROW((void)pool.allocate(std::size_t{48} << 20), poolkit::PoolError);
    EXPECT_EQ(pool.snapshot().bytes_large_live, half);
    EXPECT_EQ(backend.alloc_calls(), std::size_t{1});  // only the accepted one

    poolkit::DevicePtr small = pool.allocate(1024);
    EXPECT_TRUE(small.valid());
    EXPECT_EQ(pool.snapshot().bytes_resident, pool.config().chunk_bytes);
}


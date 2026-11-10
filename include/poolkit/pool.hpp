#pragma once

// Host-side allocator core: slab carving, per-class free lists, deferred reuse.
// Metadata lives on the host, never in device memory, and v1.0 is single-threaded
// (declared in the README, not hidden).

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <poolkit/backend.hpp>
#include <poolkit/config.hpp>
#include <poolkit/device_ptr.hpp>
#include <poolkit/metrics.hpp>
#include <poolkit/pool_error.hpp>
#include <poolkit/size_classes.hpp>
#include <poolkit/slab.hpp>

namespace poolkit {

class Pool {
public:
    struct Config {
        std::size_t chunk_bytes = std::size_t{2} << 20;
        // Also the alignment of every block.
        std::size_t min_class_bytes = 256;
        std::size_t max_class_bytes = std::size_t{2} << 20;
        double growth = 1.5;
        bool include_power_of_two = true;
        std::vector<std::size_t> extra_sizes{};
        std::size_t max_bytes = 0;  // 0 = no ceiling
        StreamHandle default_stream = nullptr;
    };

    // Injected, not created: the pool cannot know whether it talks to cudaMalloc or to a fake.
    explicit Pool(Backend& backend, Config config);

    // Overload, not a default argument: GCC rejects Config{} inside the class that defines it.
    Pool(Backend& backend);

    // Contract: no live handles. release_all() enforces it by refusing, not by freeing.
    ~Pool() noexcept;

    // Not movable: every live DevicePtr stores a Pool* to find its free list again.
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&) = delete;
    Pool& operator=(Pool&&) = delete;

    // Above max_class_bytes: one backend allocation, never reused. Throws PoolError, never nullptr.
    POOLKIT_NODISCARD DevicePtr allocate(std::size_t size, StreamHandle stream = nullptr);

    // Never blocks: the block becomes reusable only once that event has completed.
    // Fail-closed: if the event cannot be recorded the block is quarantined, never reused.
    void free(DevicePtr&& ptr, StreamHandle last_use_stream) noexcept;
    // Assumes the work is ordered with the handle's allocation stream.
    void free(DevicePtr&& ptr) noexcept;

    // Non-blocking: one query per pending block, never a wait. Not noexcept: the ready list may grow.
    std::size_t poll();

    // Not noexcept: a bad_alloc here must not turn the only diagnostic into terminate().
    POOLKIT_NODISCARD PoolSnapshot snapshot() const;

    POOLKIT_NODISCARD std::size_t capacity() const noexcept { return bytes_resident_; }

    POOLKIT_NODISCARD std::size_t reuse_count() const noexcept { return reuse_count_; }
    POOLKIT_NODISCARD std::size_t growth_count() const noexcept { return growth_count_; }
    POOLKIT_NODISCARD std::size_t large_path_count() const noexcept { return large_path_count_; }

    // Fail-closed in three ways: any live handle and it frees nothing and returns 0;
    // a chunk holding a quarantined block stays; a chunk whose pending event has not
    // completed stays too, asked again on a later call. Never waits for a stream.
    std::size_t release_all() noexcept;

    POOLKIT_NODISCARD const SizeClassTable& size_classes() const noexcept { return table_; }
    POOLKIT_NODISCARD const Config& config() const noexcept { return config_; }
    POOLKIT_NODISCARD Backend& backend() const noexcept { return backend_; }

private:
    using Clock = std::chrono::steady_clock;

    static SizeClassTable::Config make_table_config(const Config& config);

    enum class State : std::uint8_t {
        unused,
        free_ready,
        free_pending,
        live,
        // Whether the stream still writes here is unknown: terminal, never reused.
        quarantined
    };

    struct Block {
        std::size_t offset = 0;
        std::uint16_t class_index = 0;
        State state = State::unused;
        void* pending_event = nullptr;
        // Bumped per allocation: a stale handle is detectable in a debugger.
        std::uint32_t generation = 0;
    };

    struct Slab {
        void* base = nullptr;
        std::size_t bytes = 0;
        std::size_t class_index = 0;
        std::size_t tail = 0;
        std::size_t next_unused = 0;
        std::vector<Block> blocks;
    };

    struct ClassState {
        std::vector<std::uint64_t> free_ready;
        std::vector<std::uint64_t> free_pending;
        std::vector<std::uint32_t> slabs;
        std::size_t live_blocks = 0;
        std::size_t quarantined_blocks = 0;
        std::size_t total_blocks = 0;
        std::size_t alloc_count = 0;
        std::size_t reuse_count = 0;
        std::size_t recovered_count = 0;
    };

    struct LargeBlock {
        void* ptr = nullptr;
        std::size_t size = 0;
    };

    // Slab index in the high half, block index in the low half: neither reaches 2^32.
    static std::uint64_t make_handle(std::size_t slab, std::size_t block) noexcept {
        return (static_cast<std::uint64_t>(slab) << 32) | static_cast<std::uint64_t>(block);
    }

    Block& block_of(std::uint64_t handle) noexcept {
        return slabs_[static_cast<std::size_t>(handle >> 32)]
            .blocks[static_cast<std::size_t>(handle & 0xFFFFFFFFu)];
    }

    void promote_pending(std::size_t class_index);

    bool take_unused(std::size_t class_index, std::uint64_t& handle) noexcept;

    // Strong guarantee: on throw the pool state is untouched and the chunk is already back with the driver.
    void add_slab(std::size_t class_index, std::size_t request);

    static bool slab_is_quarantined(const Slab& slab) noexcept;

    // A query that throws answers "not ready", because release_all() is noexcept.
    bool slab_can_be_released(const Slab& slab) noexcept;

    // Terminal: a stream may still be writing there. noexcept because free() is.
    void quarantine_block(Block& block) noexcept;

    // Owns the chunk until release(); every non-success exit gives it back to the driver.
    class RawChunk {
    public:
        RawChunk(Backend& backend, void* ptr) noexcept : backend_(backend), ptr_(ptr) {}
        ~RawChunk() noexcept {
            if (ptr_ != nullptr) {
                backend_.free_raw(ptr_);
            }
        }
        RawChunk(const RawChunk&) = delete;
        RawChunk& operator=(const RawChunk&) = delete;

        void release() noexcept { ptr_ = nullptr; }

    private:
        Backend& backend_;
        void* ptr_;
    };

    POOLKIT_NODISCARD std::string oom_message(std::size_t request,
                                              std::size_t class_index) const;
    POOLKIT_NODISCARD std::string large_oom_message(std::size_t request) const;
    POOLKIT_NODISCARD std::string large_ceiling_message(std::size_t request) const;

    static double elapsed_ns(Clock::time_point start) noexcept;
    void record_alloc_time(double ns) noexcept;
    void record_free_time(double ns) noexcept;

    Backend& backend_;
    Config config_;
    SizeClassTable table_;

    std::vector<Slab> slabs_;
    std::vector<ClassState> classes_;
    // Doubles as the double-free detector: a pointer that is not here was never handed out.
    std::unordered_map<void*, std::uint64_t> live_;
    std::vector<LargeBlock> large_;

    std::size_t bytes_resident_ = 0;
    std::size_t bytes_requested_live_ = 0;
    std::size_t bytes_live_in_pool_ = 0;
    std::size_t bytes_capacity_live_ = 0;
    std::size_t bytes_large_live_ = 0;
    std::size_t high_water_mark_ = 0;
    std::size_t chunk_count_ = 0;

    std::size_t reuse_count_ = 0;
    std::size_t growth_count_ = 0;
    std::size_t recovered_count_ = 0;
    std::size_t large_path_count_ = 0;
    std::size_t free_errors_ = 0;
    std::size_t quarantined_blocks_ = 0;
    std::size_t quarantined_bytes_ = 0;
    std::size_t release_all_refused_ = 0;

    double alloc_sum_ns_ = 0.0;
    double alloc_max_ns_ = 0.0;
    std::size_t alloc_timed_ = 0;
    double free_sum_ns_ = 0.0;
    double free_max_ns_ = 0.0;
    std::size_t free_timed_ = 0;
};

inline SizeClassTable::Config Pool::make_table_config(const Config& config) {
    SizeClassTable::Config table_config;
    table_config.min_class_bytes = config.min_class_bytes;
    table_config.max_class_bytes = config.max_class_bytes;
    table_config.growth = config.growth;
    table_config.include_power_of_two = config.include_power_of_two;
    table_config.extra_sizes = config.extra_sizes;
    return table_config;
}

inline Pool::Pool(Backend& backend, Config config)
    : backend_(backend),
      config_(config),
      table_(make_table_config(config)) {
    if (config_.chunk_bytes == 0) {
        throw PoolError("poolkit: chunk_bytes must be > 0");
    }
    if (config_.chunk_bytes < config_.min_class_bytes) {
        throw PoolError("poolkit: chunk_bytes must be >= min_class_bytes");
    }
    if (config_.max_bytes != 0 && config_.max_bytes < config_.chunk_bytes) {
        throw PoolError("poolkit: max_bytes must be 0 or >= chunk_bytes");
    }
    classes_.resize(table_.count());
}

inline Pool::Pool(Backend& backend) : Pool(backend, Config{}) {}

inline Pool::~Pool() noexcept {
    release_all();
}

inline DevicePtr Pool::allocate(std::size_t size, StreamHandle stream) {
    const Clock::time_point start = Clock::now();

    if (size == 0) {
        // A zero-byte request has no class; padding it to 256 B would hide a caller bug.
        throw PoolError("poolkit: a zero-byte allocation has no size class");
    }

    const StreamHandle effective_stream = stream == nullptr ? config_.default_stream : stream;

    if (size > config_.max_class_bytes) {
        if (config_.max_bytes != 0) {
            const std::size_t used = bytes_resident_ + bytes_large_live_;
            if (used > config_.max_bytes || size > config_.max_bytes - used) {
                throw PoolError(large_ceiling_message(size));
            }
        }

        void* raw = backend_.allocate_raw(size);
        if (raw == nullptr) {
            throw PoolError(large_oom_message(size));
        }
        // large_.push_back() allocates: on throw the chunk goes back instead of leaking untracked.
        RawChunk owner(backend_, raw);
        large_.push_back(LargeBlock{raw, size});
        owner.release();
        bytes_large_live_ += size;
        bytes_requested_live_ += size;
        ++large_path_count_;
        record_alloc_time(elapsed_ns(start));
        // capacity == size: the large path neither rounds nor reuses, so there is no event either.
        return DevicePtr(this, raw, size, size, effective_stream, 0, true);
    }

    const std::size_t index = table_.index_for(size);
    ClassState& state = classes_[index];

    std::uint64_t handle = 0;
    bool reused = false;

    if (!state.free_ready.empty()) {
        handle = state.free_ready.back();
        state.free_ready.pop_back();
        reused = true;
    } else {
        promote_pending(index);
        if (!state.free_ready.empty()) {
            handle = state.free_ready.back();
            state.free_ready.pop_back();
            reused = true;
        }
    }

    if (!reused) {
        if (!take_unused(index, handle)) {
            add_slab(index, size);
            if (!take_unused(index, handle)) {
                throw PoolError("poolkit: internal error: a fresh slab has no slot");
            }
        }
        ++growth_count_;
    }

    Block& block = block_of(handle);
    block.state = State::live;
    ++block.generation;
    const std::size_t class_size = table_.class_size(index);
    const std::size_t slab_index = static_cast<std::size_t>(handle >> 32);
    void* data = static_cast<char*>(slabs_[slab_index].base) + block.offset;

    live_.emplace(data, handle);
    ++state.live_blocks;
    ++state.alloc_count;
    if (reused) {
        ++state.reuse_count;
        ++reuse_count_;
    }
    bytes_requested_live_ += size;
    bytes_live_in_pool_ += size;
    bytes_capacity_live_ += class_size;

    record_alloc_time(elapsed_ns(start));
    return DevicePtr(this, data, size, class_size, effective_stream, handle, false);
}

inline void Pool::promote_pending(std::size_t class_index) {
    ClassState& state = classes_[class_index];
    std::size_t kept = 0;
    for (std::size_t i = 0; i < state.free_pending.size(); ++i) {
        const std::uint64_t handle = state.free_pending[i];
        Block& block = block_of(handle);
        if (backend_.query(block.pending_event)) {
            backend_.destroy_event(block.pending_event);
            block.pending_event = nullptr;
            block.state = State::free_ready;
            state.free_ready.push_back(handle);
            ++state.recovered_count;
            ++recovered_count_;
        } else {
            state.free_pending[kept] = handle;
            ++kept;
        }
    }
    state.free_pending.resize(kept);
}

inline bool Pool::take_unused(std::size_t class_index, std::uint64_t& handle) noexcept {
    ClassState& state = classes_[class_index];
    for (std::size_t i = state.slabs.size(); i-- > 0;) {
        Slab& slab = slabs_[static_cast<std::size_t>(state.slabs[i])];
        if (slab.next_unused < slab.blocks.size()) {
            const std::size_t block = slab.next_unused;
            ++slab.next_unused;
            slab.blocks[block].state = State::live;
            handle = make_handle(static_cast<std::size_t>(state.slabs[i]), block);
            return true;
        }
    }
    return false;
}

inline void Pool::add_slab(std::size_t class_index, std::size_t request) {
    const std::size_t class_size = table_.class_size(class_index);
    const std::size_t chunk = chunk_bytes_for(class_size, config_.chunk_bytes);

    if (config_.max_bytes != 0 && bytes_resident_ + chunk > config_.max_bytes) {
        throw PoolError(oom_message(request, class_index));
    }

    void* base = backend_.allocate_raw(chunk);
    if (base == nullptr) {
        throw PoolError(oom_message(request, class_index));
    }

    RawChunk owner(backend_, base);

    const std::size_t count = blocks_per_chunk(class_size, chunk);
    ClassState& state = classes_[class_index];

    // Reserved here, the one place allowed to throw: free()'s push_back must not reallocate in a noexcept function.
    const std::size_t handles = state.total_blocks + count;
    state.free_ready.reserve(handles);
    state.free_pending.reserve(handles);
    state.slabs.reserve(state.slabs.size() + 1);

    Slab slab;
    slab.base = base;
    slab.bytes = chunk;
    slab.class_index = class_index;
    slab.tail = chunk - count * class_size;
    slab.blocks.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        slab.blocks[i].offset = i * class_size;
        slab.blocks[i].class_index = static_cast<std::uint16_t>(class_index);
    }

    slabs_.push_back(std::move(slab));
    const std::size_t slab_index = slabs_.size() - 1;
    state.slabs.push_back(static_cast<std::uint32_t>(slab_index));
    state.total_blocks += count;

    owner.release();
    bytes_resident_ += chunk;
    ++chunk_count_;
    if (bytes_resident_ > high_water_mark_) {
        high_water_mark_ = bytes_resident_;
    }
}

inline std::string Pool::oom_message(std::size_t request, std::size_t class_index) const {
    return "poolkit: out of memory: request " + std::to_string(request) + " B, class " +
           std::to_string(class_index) + " (" + std::to_string(table_.class_size(class_index)) +
           " B), resident " + std::to_string(bytes_resident_) + " B";
}

inline std::string Pool::large_oom_message(std::size_t request) const {
    return "poolkit: out of memory: large-path request " + std::to_string(request) +
           " B (above max_class_bytes), resident " + std::to_string(bytes_resident_) + " B";
}

inline std::string Pool::large_ceiling_message(std::size_t request) const {
    return "poolkit: Config::max_bytes exceeded: large-path request " + std::to_string(request) +
           " B (above max_class_bytes) does not fit under the " +
           std::to_string(config_.max_bytes) + " B ceiling, resident " +
           std::to_string(bytes_resident_) + " B, large live " +
           std::to_string(bytes_large_live_) + " B";
}

inline void Pool::free(DevicePtr&& ptr, StreamHandle last_use_stream) noexcept {
    if (ptr.pool_ != this) {
        // Foreign or moved-from handle: leave it to its owner and count it; free() is noexcept.
        if (ptr.valid()) {
            ++free_errors_;
        }
        return;
    }
    if (!ptr.valid()) {
        ptr.detach();
        return;
    }

    const Clock::time_point start = Clock::now();
    const std::size_t size = ptr.size_;
    void* data = ptr.data_;
    const bool large = ptr.large_;
    const std::uint64_t handle = ptr.block_;

    if (large) {
        // No event: the large path never reuses, so there is nothing to wait for.
        for (std::size_t i = 0; i < large_.size(); ++i) {
            if (large_[i].ptr == data) {
                large_[i] = large_.back();
                large_.pop_back();
                break;
            }
        }
        bytes_requested_live_ -= size;
        bytes_large_live_ -= size;
        backend_.free_raw(data);
        ptr.detach();
        record_free_time(elapsed_ns(start));
        return;
    }

    const auto it = live_.find(data);
    if (it == live_.end()) {
        // Not in live_: only a forged handle gets here, so count it and do not touch the block.
        ++free_errors_;
        ptr.detach();
        return;
    }
    live_.erase(it);

    Block& block = block_of(handle);
    ClassState& state = classes_[block.class_index];
    if (state.live_blocks > 0) {
        --state.live_blocks;
    }
    bytes_requested_live_ -= size;
    bytes_live_in_pool_ -= size;
    bytes_capacity_live_ -= table_.class_size(block.class_index);

    // From here nothing may escape: free() is noexcept. A record_event that returns
    // nullptr and one that throws mean the same thing — the pool cannot prove when a
    // stream is done — so both quarantine the block instead of reusing it.
    try {
        void* event = backend_.record_event(last_use_stream);
        if (event == nullptr) {
            quarantine_block(block);
        } else {
            // Cannot reallocate: add_slab() reserved it. Assigned first so quarantine would still find it.
            block.pending_event = event;
            block.state = State::free_pending;
            state.free_pending.push_back(handle);
        }
    } catch (...) {
        quarantine_block(block);
    }

    ptr.detach();
    record_free_time(elapsed_ns(start));
}

inline void Pool::quarantine_block(Block& block) noexcept {
    if (block.pending_event != nullptr) {
        // Recorded but never linked into the free list: nothing will ever query it, so it goes.
        backend_.destroy_event(block.pending_event);
        block.pending_event = nullptr;
    }
    block.state = State::quarantined;
    ClassState& state = classes_[block.class_index];
    ++state.quarantined_blocks;
    ++quarantined_blocks_;
    quarantined_bytes_ += table_.class_size(block.class_index);
    ++free_errors_;
}

inline void Pool::free(DevicePtr&& ptr) noexcept {
    // Read the stream before moving: after the move the handle is empty.
    const StreamHandle stream = ptr.stream();
    free(std::move(ptr), stream);
}

inline std::size_t Pool::poll() {
    std::size_t promoted = 0;
    for (std::size_t index = 0; index < classes_.size(); ++index) {
        if (classes_[index].free_pending.empty()) {
            continue;
        }
        const std::size_t before = classes_[index].free_ready.size();
        promote_pending(index);
        promoted += classes_[index].free_ready.size() - before;
    }
    return promoted;
}

inline bool Pool::slab_is_quarantined(const Slab& slab) noexcept {
    for (const Block& block : slab.blocks) {
        if (block.state == State::quarantined) {
            return true;
        }
    }
    return false;
}

inline bool Pool::slab_can_be_released(const Slab& slab) noexcept {
    if (slab_is_quarantined(slab)) {
        return false;
    }
    // Query and destruction are separate passes: a slab that must stay keeps its events.
    for (const Block& block : slab.blocks) {
        if (block.state != State::free_pending) {
            continue;
        }
        bool ready = false;
        try {
            ready = backend_.query(block.pending_event);
        } catch (...) {
            ready = false;
        }
        if (!ready) {
            return false;
        }
    }
    return true;
}

inline PoolSnapshot Pool::snapshot() const {
    PoolSnapshot snap;
    snap.bytes_requested_live = bytes_requested_live_;
    snap.bytes_live_in_pool = bytes_live_in_pool_;
    snap.bytes_capacity_live = bytes_capacity_live_;
    snap.bytes_resident = bytes_resident_;
    snap.bytes_large_live = bytes_large_live_;
    snap.high_water_mark_bytes = high_water_mark_;
    snap.chunk_count = chunk_count_;

    snap.reuse_count = reuse_count_;
    snap.growth_count = growth_count_;
    snap.recovered_count = recovered_count_;
    snap.large_path_count = large_path_count_;
    snap.free_errors = free_errors_;
    snap.quarantined_blocks = quarantined_blocks_;
    snap.quarantined_bytes = quarantined_bytes_;
    snap.release_all_refused = release_all_refused_;
    const std::size_t served = reuse_count_ + growth_count_;
    snap.hit_rate = served == 0 ? 0.0 : static_cast<double>(reuse_count_) / static_cast<double>(served);

    if (bytes_live_in_pool_ > 0) {
        const double waste = static_cast<double>(bytes_capacity_live_ - bytes_live_in_pool_);
        snap.internal_fragmentation = waste / static_cast<double>(bytes_live_in_pool_);
    }

    snap.classes.reserve(table_.count());
    for (std::size_t index = 0; index < table_.count(); ++index) {
        const ClassState& state = classes_[index];
        PoolSnapshot::ClassStats stats;
        stats.index = index;
        stats.class_size = table_.class_size(index);
        stats.slab_count = state.slabs.size();
        stats.total_blocks = state.total_blocks;
        stats.live_blocks = state.live_blocks;
        stats.ready_blocks = state.free_ready.size();
        stats.pending_blocks = state.free_pending.size();
        stats.quarantined_blocks = state.quarantined_blocks;
        stats.unused_blocks = state.total_blocks - stats.live_blocks - stats.ready_blocks -
                              stats.pending_blocks - stats.quarantined_blocks;
        stats.tail_bytes_per_slab = slab_layout(stats.class_size, config_.chunk_bytes).tail;
        stats.alloc_count = state.alloc_count;
        stats.reuse_count = state.reuse_count;
        stats.hit_rate = state.alloc_count == 0
                             ? 0.0
                             : static_cast<double>(state.reuse_count) /
                                   static_cast<double>(state.alloc_count);

        snap.bytes_free_ready += stats.ready_blocks * stats.class_size;
        snap.bytes_free_pending += stats.pending_blocks * stats.class_size;
        snap.bytes_unused_slots += stats.unused_blocks * stats.class_size;
        snap.tail_bytes += stats.slab_count * stats.tail_bytes_per_slab;
        snap.classes.push_back(stats);
    }

    snap.bytes_free_in_pool =
        snap.bytes_free_ready + snap.bytes_free_pending + snap.bytes_unused_slots;

    if (bytes_resident_ > 0) {
        const double resident = static_cast<double>(bytes_resident_);
        snap.tail_fragmentation = static_cast<double>(snap.tail_bytes) / resident;
        snap.resident_slack = static_cast<double>(snap.bytes_free_in_pool) / resident;
        snap.free_ratio = snap.resident_slack;
    }

    if (alloc_timed_ > 0) {
        snap.alloc_avg_ns = alloc_sum_ns_ / static_cast<double>(alloc_timed_);
        snap.alloc_max_ns = alloc_max_ns_;
    }
    if (free_timed_ > 0) {
        snap.free_avg_ns = free_sum_ns_ / static_cast<double>(free_timed_);
        snap.free_max_ns = free_max_ns_;
    }
    return snap;
}

inline std::size_t Pool::release_all() noexcept {
    // Fail-closed: a live handle can still be written by the caller, so nothing is freed and it is counted.
    if (!live_.empty() || !large_.empty()) {
        ++release_all_refused_;
        return 0;
    }

    std::size_t freed_bytes = 0;
    std::size_t freed_chunks = 0;

    std::size_t kept = 0;
    for (std::size_t read = 0; read < slabs_.size(); ++read) {
        Slab& slab = slabs_[read];
        // Fail-closed: a chunk with a quarantined block, or with an event not yet complete, stays.
        if (!slab_can_be_released(slab)) {
            if (kept != read) {
                slabs_[kept] = std::move(slab);
            }
            ++kept;
            continue;
        }
        for (const Block& block : slab.blocks) {
            if (block.state == State::free_pending) {
                // After the query, never before: destroying it first would leave nothing to ask.
                backend_.destroy_event(block.pending_event);
            }
        }
        if (slab.base != nullptr) {
            backend_.free_raw(slab.base);
            slab.base = nullptr;
        }
        freed_bytes += slab.bytes;
        ++freed_chunks;
    }
    slabs_.resize(kept);

    for (ClassState& state : classes_) {
        state.free_ready.clear();
        state.free_pending.clear();
        state.slabs.clear();
        state.live_blocks = 0;
        state.quarantined_blocks = 0;
        state.total_blocks = 0;
    }
    for (std::size_t index = 0; index < slabs_.size(); ++index) {
        Slab& slab = slabs_[index];
        ClassState& state = classes_[slab.class_index];
        state.slabs.push_back(static_cast<std::uint32_t>(index));
        state.total_blocks += slab.blocks.size();
        for (std::size_t block = 0; block < slab.blocks.size(); ++block) {
            const std::uint64_t handle = make_handle(index, block);
            switch (slab.blocks[block].state) {
                case State::free_ready:
                    state.free_ready.push_back(handle);
                    break;
                case State::free_pending:
                    state.free_pending.push_back(handle);
                    break;
                case State::quarantined:
                    ++state.quarantined_blocks;
                    break;
                case State::unused:
                case State::live:
                    break;
            }
        }
    }

    for (const LargeBlock& entry : large_) {
        backend_.free_raw(entry.ptr);
    }
    large_.clear();

    bytes_resident_ -= freed_bytes;
    chunk_count_ -= freed_chunks;
    bytes_requested_live_ = 0;
    bytes_live_in_pool_ = 0;
    bytes_capacity_live_ = 0;
    bytes_large_live_ = 0;
    return freed_bytes;
}

inline double Pool::elapsed_ns(Clock::time_point start) noexcept {
    return std::chrono::duration<double, std::nano>(Clock::now() - start).count();
}

inline void Pool::record_alloc_time(double ns) noexcept {
    alloc_sum_ns_ += ns;
    ++alloc_timed_;
    if (ns > alloc_max_ns_) {
        alloc_max_ns_ = ns;
    }
}

inline void Pool::record_free_time(double ns) noexcept {
    free_sum_ns_ += ns;
    ++free_timed_;
    if (ns > free_max_ns_) {
        free_max_ns_ = ns;
    }
}

// DevicePtr bodies live here: releasing a block needs the complete Pool, and Pool needs DevicePtr.

inline DevicePtr::~DevicePtr() {
    if (pool_ != nullptr) {
        pool_->free(std::move(*this), stream_);
    }
}

inline void DevicePtr::release(StreamHandle last_use_stream) noexcept {
    if (pool_ != nullptr) {
        pool_->free(std::move(*this), last_use_stream);
    }
}

}  // namespace poolkit

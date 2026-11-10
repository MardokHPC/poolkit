#pragma once

// A Backend that answers with plausible addresses and allocates none of the bytes it promises:
// the advisor replays a trace ~1400 times and a trace may describe 8 GiB, so the analysis must
// cost the same whether the trace is 200 KiB or 20 GiB.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <unordered_map>

#include <poolkit/backend.hpp>

namespace poolkit {

class SimBackend final : public Backend {
public:
    // Same floor the public API promises; a less aligned fake block would let an alignment bug pass.
    static constexpr std::size_t alignment = 256;

    static constexpr std::uintptr_t arena_base = 0x7000'0000'0000ull;

    // Events are values too: one distinct non-null value per record, in a range that cannot collide.
    static constexpr std::uintptr_t event_base = 0x7fff'0000'0000ull;

    SimBackend() = default;
    ~SimBackend() override = default;

    SimBackend(const SimBackend&) = delete;
    SimBackend& operator=(const SimBackend&) = delete;

    void* allocate_raw(std::size_t bytes) override;
    void free_raw(void* ptr) noexcept override;
    std::size_t resident_step() const override { return alignment; }

    // Complete on record, like FakeBackend's default: a different answer would change the peak.
    void* record_event(StreamHandle stream) override;
    bool query(void* event) override { return event != nullptr; }
    void destroy_event(void* event) noexcept override;

    POOLKIT_NODISCARD std::size_t bytes_resident() const noexcept { return resident_; }
    POOLKIT_NODISCARD std::size_t chunk_count() const noexcept { return chunks_.size(); }
    POOLKIT_NODISCARD std::size_t alloc_calls() const noexcept { return alloc_calls_; }
    POOLKIT_NODISCARD std::size_t free_calls() const noexcept { return free_calls_; }
    POOLKIT_NODISCARD std::size_t invalid_frees() const noexcept { return invalid_frees_; }
    POOLKIT_NODISCARD std::size_t events_recorded() const noexcept { return events_recorded_; }

private:
    std::unordered_map<void*, std::size_t> chunks_;
    std::uintptr_t next_address_ = arena_base;
    std::uintptr_t next_event_ = event_base;
    std::size_t resident_ = 0;
    std::size_t alloc_calls_ = 0;
    std::size_t free_calls_ = 0;
    std::size_t invalid_frees_ = 0;
    std::size_t events_recorded_ = 0;
};

inline void* SimBackend::allocate_raw(std::size_t bytes) {
    ++alloc_calls_;
    const std::size_t rounded =
        bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)
            ? bytes
            : ((bytes + alignment - 1) / alignment) * alignment;
    if (rounded > std::numeric_limits<std::uintptr_t>::max() - next_address_) {
        // Reported as nullptr, the way the driver reports it: the pool must not see a difference in kind.
        return nullptr;
    }
    if (rounded < bytes) {
        return nullptr;
    }
    void* ptr = reinterpret_cast<void*>(next_address_);
    next_address_ += rounded;
    try {
        chunks_.emplace(ptr, bytes);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
    resident_ += bytes;
    return ptr;
}

inline void SimBackend::free_raw(void* ptr) noexcept {
    ++free_calls_;
    const auto it = chunks_.find(ptr);
    if (it == chunks_.end()) {
        // Same as FakeBackend: a pointer never handed out is counted, not ignored.
        ++invalid_frees_;
        return;
    }
    resident_ -= it->second;
    chunks_.erase(it);
}

inline void* SimBackend::record_event(StreamHandle /*stream*/) {
    ++events_recorded_;
    void* event = reinterpret_cast<void*>(next_event_);
    // 256 B apart: distinct even if something later aligns the value.
    next_event_ += alignment;
    if (next_event_ < event_base) {  // wrapped: keep the handle non-null
        next_event_ = event_base;
    }
    return event;
}

inline void SimBackend::destroy_event(void* /*event*/) noexcept {
    // Nothing to free: an event is a value here, not an allocation.
}

}  // namespace poolkit

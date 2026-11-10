#pragma once

// The seam between the allocation policy and the driver primitives: everything that decides
// where a block goes talks to this interface only, which is what makes the allocator testable
// on a CPU with no GPU and no nvcc. No CUDA header here, on purpose.

#include <cstddef>
#include <new>
#include <vector>

#include <poolkit/config.hpp>

namespace poolkit {

class Backend {
public:
    Backend() = default;
    virtual ~Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // nullptr on failure, not an exception, so the contract is visible from the signature: the pool
    // turns it into PoolError carrying the request, the class and the resident bytes.
    virtual void* allocate_raw(std::size_t bytes) = 0;

    // Must accept exactly the pointers allocate_raw returned, and be noexcept: ~Pool and free() call it.
    virtual void free_raw(void* ptr) noexcept = 0;

    // Observed allocation granularity, 0 when unknown. Informational only: correctness never depends on it.
    virtual std::size_t resident_step() const = 0;

    // Opaque event handle. nullptr is the failure contract; a throw is tolerated (the block is
    // quarantined), but a caller cannot tell the two apart and only one keeps the error countable.
    virtual void* record_event(StreamHandle stream) = 0;
    virtual bool query(void* event) = 0;
    virtual void destroy_event(void* event) noexcept = 0;
};

// Host backend with real memory and scriptable events: the memory is genuinely allocated and 256 B
// aligned, so a test proves two blocks do not overlap and a double free is caught by the allocator
// underneath. Events are the only simulated part, which is what makes deferred reuse deterministic.
class FakeBackend : public Backend {
public:
    explicit FakeBackend(std::size_t resident_step = 0) noexcept : resident_step_(resident_step) {}

    ~FakeBackend() override;

    FakeBackend(const FakeBackend&) = delete;
    FakeBackend& operator=(const FakeBackend&) = delete;

    void* allocate_raw(std::size_t bytes) override;
    void free_raw(void* ptr) noexcept override;
    std::size_t resident_step() const override { return resident_step_; }

    void* record_event(StreamHandle stream) override;
    bool query(void* event) override;
    void destroy_event(void* event) noexcept override;

    void set_events_complete_on_record(bool complete) noexcept { events_complete_on_record_ = complete; }
    void make_ready(void* event) noexcept;
    void keep_pending(void* event) noexcept;
    void make_all_events_ready() noexcept;

    void fail_next_alloc() noexcept { fail_next_alloc_ = true; }
    void fail_next_event() noexcept { fail_next_event_ = true; }

    POOLKIT_NODISCARD void* last_event() const noexcept { return last_event_; }
    POOLKIT_NODISCARD std::size_t chunk_count() const noexcept { return chunks_.size(); }
    POOLKIT_NODISCARD std::size_t bytes_resident() const noexcept;
    POOLKIT_NODISCARD std::size_t alloc_calls() const noexcept { return alloc_calls_; }
    POOLKIT_NODISCARD std::size_t free_calls() const noexcept { return free_calls_; }
    POOLKIT_NODISCARD std::size_t invalid_frees() const noexcept { return invalid_frees_; }
    POOLKIT_NODISCARD std::size_t events_created() const noexcept { return events_.size(); }

private:
    struct Chunk {
        void* ptr = nullptr;
        std::size_t bytes = 0;
    };

    struct Event {
        bool ready = true;
    };

    static constexpr std::size_t alignment_ = 256;

    std::vector<Chunk> chunks_;
    std::vector<Event*> events_;
    std::size_t resident_step_ = 0;
    std::size_t alloc_calls_ = 0;
    std::size_t free_calls_ = 0;
    std::size_t invalid_frees_ = 0;
    bool events_complete_on_record_ = true;
    bool fail_next_alloc_ = false;
    bool fail_next_event_ = false;
    void* last_event_ = nullptr;
};

inline FakeBackend::~FakeBackend() {
    for (const Chunk& chunk : chunks_) {
        ::operator delete(chunk.ptr, std::align_val_t{alignment_});
    }
    for (Event* event : events_) {
        delete event;
    }
}

inline void* FakeBackend::allocate_raw(std::size_t bytes) {
    ++alloc_calls_;
    if (fail_next_alloc_) {
        fail_next_alloc_ = false;
        return nullptr;
    }
    try {
        void* ptr = ::operator new(bytes, std::align_val_t{alignment_});
        chunks_.push_back(Chunk{ptr, bytes});
        return ptr;
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

inline void FakeBackend::free_raw(void* ptr) noexcept {
    ++free_calls_;
    for (std::size_t i = 0; i < chunks_.size(); ++i) {
        if (chunks_[i].ptr == ptr) {
            ::operator delete(chunks_[i].ptr, std::align_val_t{alignment_});
            chunks_[i] = chunks_.back();
            chunks_.pop_back();
            return;
        }
    }
    // Counted, never silently ignored: this is exactly the shape of a double free.
    ++invalid_frees_;
}

inline std::size_t FakeBackend::bytes_resident() const noexcept {
    std::size_t total = 0;
    for (const Chunk& chunk : chunks_) {
        total += chunk.bytes;
    }
    return total;
}

inline void* FakeBackend::record_event(StreamHandle /*stream*/) {
    if (fail_next_event_) {
        fail_next_event_ = false;
        return nullptr;
    }
    Event* event = new Event{events_complete_on_record_};
    events_.push_back(event);
    last_event_ = event;
    return event;
}

inline bool FakeBackend::query(void* event) {
    if (event == nullptr) {
        return true;
    }
    return static_cast<Event*>(event)->ready;
}

inline void FakeBackend::destroy_event(void* event) noexcept {
    if (event == nullptr) {
        return;
    }
    for (std::size_t i = 0; i < events_.size(); ++i) {
        if (events_[i] == event) {
            delete events_[i];
            events_[i] = events_.back();
            events_.pop_back();
            if (last_event_ == event) {
                last_event_ = nullptr;
            }
            return;
        }
    }
}

inline void FakeBackend::make_ready(void* event) noexcept {
    if (event != nullptr) {
        static_cast<Event*>(event)->ready = true;
    }
}

inline void FakeBackend::keep_pending(void* event) noexcept {
    if (event != nullptr) {
        static_cast<Event*>(event)->ready = false;
    }
}

inline void FakeBackend::make_all_events_ready() noexcept {
    for (Event* event : events_) {
        event->ready = true;
    }
}

}  // namespace poolkit

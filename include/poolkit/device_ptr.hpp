#pragma once

// A move-only, non-owning handle to one block of the pool: releasing it returns the block to its
// class free list, it never calls cudaFree. The handle must not outlive its Pool (the destructor
// calls back into it), and a live handle keeps release_all() from freeing anything.

#include <cstddef>
#include <cstdint>
#include <utility>

#include <poolkit/config.hpp>

namespace poolkit {

class Pool;

class DevicePtr {
public:
    DevicePtr() noexcept = default;

    // Move-only: a copy would be two handles on one block, i.e. a double free waiting for a scope exit.
    DevicePtr(DevicePtr&& other) noexcept;
    DevicePtr& operator=(DevicePtr&& other) noexcept;
    DevicePtr(const DevicePtr&) = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;

    // Defined in pool.hpp, because releasing needs the complete Pool: destroying a handle without
    // pool.hpp is a link error, not a silently empty destructor.
    ~DevicePtr();

    POOLKIT_NODISCARD void* get() const noexcept { return data_; }
    POOLKIT_NODISCARD std::size_t size() const noexcept { return size_; }
    POOLKIT_NODISCARD std::size_t capacity() const noexcept { return capacity_; }
    POOLKIT_NODISCARD StreamHandle stream() const noexcept { return stream_; }
    POOLKIT_NODISCARD bool valid() const noexcept { return data_ != nullptr; }

    void release(StreamHandle last_use_stream) noexcept;
    // Assumes the work is ordered with the allocation stream, as the README says.
    void release() noexcept { release(stream_); }

private:
    friend class Pool;

    DevicePtr(Pool* pool, void* data, std::size_t size, std::size_t capacity,
              StreamHandle stream, std::uint64_t block, bool large) noexcept;

    // Detaches without releasing: the pool calls it once it has taken the block back.
    void detach() noexcept;

    Pool* pool_ = nullptr;
    void* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
    StreamHandle stream_{};
    std::uint64_t block_ = 0;
    bool large_ = false;
};

inline DevicePtr::DevicePtr(Pool* pool, void* data, std::size_t size, std::size_t capacity,
                            StreamHandle stream, std::uint64_t block, bool large) noexcept
    : pool_(pool), data_(data), size_(size), capacity_(capacity), stream_(stream), block_(block),
      large_(large) {}

inline DevicePtr::DevicePtr(DevicePtr&& other) noexcept
    : pool_(other.pool_), data_(other.data_), size_(other.size_), capacity_(other.capacity_),
      stream_(other.stream_), block_(other.block_), large_(other.large_) {
    other.detach();
}

inline DevicePtr& DevicePtr::operator=(DevicePtr&& other) noexcept {
    if (this != &other) {
        release();
        pool_ = other.pool_;
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        stream_ = other.stream_;
        block_ = other.block_;
        large_ = other.large_;
        other.detach();
    }
    return *this;
}

inline void DevicePtr::detach() noexcept {
    pool_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
    stream_ = StreamHandle{};
    block_ = 0;
    large_ = false;
}

}  // namespace poolkit

#pragma once

// The recording core shared by tools/record.cu (poolkit_record) and tools/inject.cu
// (poolkit_inject, the CUDA_INJECTION64_PATH shim): same callbacks, filters, JSONL
// writer and lifecycle, so the two tools cannot produce different formats.

// MSVC reports C4324 inside CUPTI's own header: /W4 /WX is right for this code, not
// for that header, so the warning is silenced around the include and nowhere else.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
#include <cupti.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace poolkit {
namespace recorder {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kDefaultBufferSize = 8 * 1024 * 1024;
constexpr std::size_t kDefaultMinSize = 4096;
constexpr std::size_t kDefaultMaxRecords = 2000000;

struct MemoryRecord {
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    bool is_alloc = false;
    std::uint64_t ordinal = 0;
};

struct Stats {
    std::size_t seen = 0;              // MEMORY2 records delivered, before filters
    std::size_t kept = 0;
    std::size_t allocs = 0;
    std::size_t frees = 0;
    std::size_t small_allocs = 0;
    std::size_t orphan_frees = 0;      // frees of blocks allocated before recording
    std::size_t duplicate_allocs = 0;
    std::size_t over_cap = 0;
    std::size_t cupti_dropped = 0;
    std::size_t write_errors = 0;
    std::size_t min_size = kDefaultMinSize;
    std::size_t max_records = kDefaultMaxRecords;
    double elapsed_ms = 0.0;
};

// Prefixed to every diagnostic: a message from inside somebody else's process must say who sent it.
inline const char*& program_name() {
    static const char* name = "poolkit";
    return name;
}

inline void set_program_name(const char* name) { program_name() = name; }

// POOLKIT_RECORD_DEBUG traces the injection path, one line per step: those failures happen
// inside somebody else's process, where attaching a debugger is not an option.
inline bool debug_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("POOLKIT_RECORD_DEBUG");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

inline void debug(const char* what) {
    if (debug_enabled()) {
        std::fprintf(stderr, "%s: [debug] %s\n", program_name(), what);
        std::fflush(stderr);
    }
}

// CUPTI delivers whole buffers on its own thread; everything below is under one mutex.
// The file is written incrementally, one line per accepted operation, so a process that
// dies without a clean shutdown still leaves what CUPTI already delivered.

class Recorder {
public:
    Recorder() = default;
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    // flush_period_ms > 0 asks CUPTI to flush every that many ms; on this stack it delivered
    // nothing with 8 MiB buffers, so the shim relies on small buffers and keeps it as a retry.
    bool start(const std::string& out_path, std::size_t min_size, std::size_t max_records,
               std::uint32_t flush_period_ms = 0);

    // Never call from a process exit handler: every CUPTI call can block there, use finish_at_exit().
    bool stop(Stats& stats);

    // Exit-path variant: no CUPTI call at all (see the note on the definition below).
    void finish_at_exit(Stats& stats);

    // Per-process, like the rest: CUPTI asks for buffers through a callback with no user data.
    void set_buffer_bytes(std::size_t bytes) { buffer_bytes_ = bytes; }
    std::size_t buffer_bytes() const noexcept { return buffer_bytes_; }

    void accept(bool is_alloc, std::uint64_t address, std::uint64_t bytes);

    void set_cupti_dropped(std::size_t dropped);

    bool started() const noexcept { return started_; }

private:
    void write_line(const MemoryRecord& record);

    void fill_stats(Stats& stats) const;

    mutable std::mutex mutex_;
    std::vector<MemoryRecord> records_;
    std::unordered_set<std::uint64_t> live_;
    std::unordered_map<std::uint64_t, std::uint64_t> dropped_;
    std::ofstream out_;
    std::string path_;

    std::size_t min_size_ = kDefaultMinSize;
    std::size_t max_records_ = kDefaultMaxRecords;
    std::uint64_t next_ordinal_ = 0;

    std::size_t seen_ = 0;
    std::size_t small_allocs_ = 0;
    std::size_t orphan_frees_ = 0;
    std::size_t duplicate_allocs_ = 0;
    std::size_t over_cap_ = 0;
    std::size_t cupti_dropped_ = 0;
    std::size_t write_errors_ = 0;
    bool write_failed_ = false;
    std::size_t buffer_bytes_ = kDefaultBufferSize;

    Clock::time_point start_time_{};
    bool started_ = false;
    bool closed_ = false;
};

inline void Recorder::write_line(const MemoryRecord& record) {
    if (write_failed_) {
        ++write_errors_;
        return;
    }
    out_ << "{\"op\":\"" << (record.is_alloc ? "alloc" : "free") << "\",\"size\":"
         << record.bytes << ",\"ptr\":" << record.address << ",\"stream\":0,\"ordinal\":"
         << record.ordinal << "}\n";
    if (!out_) {
        write_failed_ = true;
        ++write_errors_;
        std::fprintf(stderr, "%s: write to '%s' failed\n", program_name(), path_.c_str());
    }
}

// Pushes before writing, so kept == records_.size() even when a write fails.
inline void Recorder::accept(bool is_alloc, std::uint64_t address, std::uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return;
    }
    ++seen_;

    if (is_alloc) {
        if (bytes < min_size_) {
            ++small_allocs_;
            dropped_[address] = bytes;
            return;
        }
        if (live_.count(address) != 0) {
            // Two live blocks at one address cannot be represented and would make the replay throw.
            ++duplicate_allocs_;
            return;
        }
        if (records_.size() >= max_records_) {
            ++over_cap_;
            return;
        }
        dropped_.erase(address);  // the previous block at this address is gone
        MemoryRecord record;
        record.address = address;
        record.bytes = bytes;
        record.is_alloc = true;
        record.ordinal = next_ordinal_++;
        live_.insert(address);
        records_.push_back(record);
        write_line(record);
        return;
    }

    const auto dropped = dropped_.find(address);
    if (dropped != dropped_.end()) {
        dropped_.erase(dropped);
        return;
    }
    if (live_.count(address) == 0) {
        // Allocated before recording started: writing the free would break the trace.
        ++orphan_frees_;
        return;
    }
    if (records_.size() >= max_records_) {
        ++over_cap_;
        return;
    }
    MemoryRecord record;
    record.address = address;
    record.bytes = 0;
    record.is_alloc = false;
    record.ordinal = next_ordinal_++;
    live_.erase(address);
    records_.push_back(record);
    write_line(record);
}

inline void Recorder::set_cupti_dropped(std::size_t dropped) {
    std::lock_guard<std::mutex> lock(mutex_);
    cupti_dropped_ = dropped;
}

// Deliberately leaked, not a function-local static object: the shim flushes from an atexit
// handler, and an ofstream destroyed at module teardown before it would lose the last write.
inline Recorder& instance() {
    static Recorder* recorder = new Recorder();
    return *recorder;
}

inline void CUPTIAPI on_buffer_completed(CUcontext /*context*/, uint32_t /*stream_id*/,
                                        uint8_t* buffer, size_t /*size*/, size_t valid_size) {
    CUpti_Activity* activity = nullptr;
    for (;;) {
        const CUptiResult status = cuptiActivityGetNextRecord(buffer, valid_size, &activity);
        if (status != CUPTI_SUCCESS) {
            break;
        }
        if (activity->kind != CUPTI_ACTIVITY_KIND_MEMORY2) {
            continue;
        }
        const auto* memory = reinterpret_cast<const CUpti_ActivityMemory3*>(activity);
        if (memory->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION) {
            instance().accept(true, memory->address, memory->bytes);
        } else if (memory->memoryOperationType ==
                   CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE) {
            instance().accept(false, memory->address, memory->bytes);
        }
    }
    std::free(buffer);
}

inline void CUPTIAPI on_buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records) {
    const std::size_t bytes = instance().buffer_bytes();
    *buffer = static_cast<uint8_t*>(std::malloc(bytes));
    if (*buffer == nullptr) {
        // Nothing sensible to do inside a CUPTI callback: report and let the next CUDA call fail.
        std::fprintf(stderr, "%s: out of host memory for a CUPTI buffer\n", program_name());
        *size = 0;
        *max_records = 0;
        return;
    }
    *size = bytes;
    *max_records = 0;
}

inline bool cupti_ok(CUptiResult status, const char* what) {
    if (status == CUPTI_SUCCESS) {
        return true;
    }
    const char* text = nullptr;
    (void)cuptiGetResultString(status, &text);
    std::fprintf(stderr, "%s: CUPTI error in %s (result %d): %s\n", program_name(), what,
                 static_cast<int>(status), text != nullptr ? text : "unknown");
    return false;
}

inline void CUPTIAPI on_api_callback(void* /*userdata*/, CUpti_CallbackDomain /*domain*/,
                                     CUpti_CallbackId /*callback_id*/, const void* /*data*/) {
    // Subscribed because cuptiSubscribe needs a callback; MEMORY2 does not go through here.
}

inline CUpti_SubscriberHandle& subscriber() {
    static CUpti_SubscriberHandle handle{};
    return handle;
}

inline bool Recorder::start(const std::string& out_path, std::size_t min_size,
                            std::size_t max_records, std::uint32_t flush_period_ms) {
    if (started_) {
        std::fprintf(stderr, "%s: already recording\n", program_name());
        return false;
    }
    path_ = out_path;
    min_size_ = min_size;
    max_records_ = max_records;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Truncate: a trace is one run, and appending would mix two runs' ordinals into one file.
        out_.open(path_, std::ios::binary | std::ios::trunc);
        if (!out_) {
            std::fprintf(stderr, "%s: cannot open '%s' for writing\n", program_name(),
                         path_.c_str());
            return false;
        }
    }

    if (!cupti_ok(cuptiSubscribe(&subscriber(), on_api_callback, nullptr), "cuptiSubscribe")) {
        return false;
    }
    debug("cuptiSubscribe ok");
    if (!cupti_ok(cuptiActivityRegisterCallbacks(on_buffer_requested, on_buffer_completed),
                  "cuptiActivityRegisterCallbacks")) {
        return false;
    }
    debug("cuptiActivityRegisterCallbacks ok");
    if (flush_period_ms != 0) {
        // CUPTI's own timer thread, not ours: a thread we started could be mid-driver-call
        // during teardown. Failure is not fatal, so it is reported and ignored.
        if (cupti_ok(cuptiActivityFlushPeriod(flush_period_ms), "cuptiActivityFlushPeriod")) {
            debug("cuptiActivityFlushPeriod ok");
        }
    }
    if (!cupti_ok(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2),
                  "cuptiActivityEnable(MEMORY2)")) {
        return false;
    }
    debug("cuptiActivityEnable(MEMORY2) ok");
    start_time_ = Clock::now();
    started_ = true;
    return true;
}

inline bool Recorder::stop(Stats& stats) {
    bool ok = true;
    if (started_) {
        debug("stop: cuptiActivityFlushAll(1)");
        ok = cupti_ok(cuptiActivityFlushAll(1), "cuptiActivityFlushAll(force)");
        debug("stop: cuptiActivityDisable");
        ok = cupti_ok(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMORY2),
                      "cuptiActivityDisable(MEMORY2)") &&
             ok;

        std::size_t dropped = 0;
        if (cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped) == CUPTI_SUCCESS) {
            set_cupti_dropped(dropped);
        }
        debug("stop: cuptiUnsubscribe");
        ok = cupti_ok(cuptiUnsubscribe(subscriber()), "cuptiUnsubscribe") && ok;
        debug("stop: cuptiFinalize");
        (void)cuptiFinalize();
        debug("stop: cuptiFinalize done");
        started_ = false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    fill_stats(stats);

    out_.flush();
    const bool flushed = static_cast<bool>(out_);
    out_.close();

    if (!flushed) {
        std::fprintf(stderr, "%s: write to '%s' failed\n", program_name(), path_.c_str());
        ok = false;
    }
    return ok;
}

inline void Recorder::fill_stats(Stats& stats) const {
    stats.seen = seen_;
    stats.kept = records_.size();
    stats.allocs = 0;
    stats.frees = 0;
    for (const MemoryRecord& record : records_) {
        if (record.is_alloc) {
            ++stats.allocs;
        } else {
            ++stats.frees;
        }
    }
    stats.small_allocs = small_allocs_;
    stats.orphan_frees = orphan_frees_;
    stats.duplicate_allocs = duplicate_allocs_;
    stats.over_cap = over_cap_;
    stats.cupti_dropped = cupti_dropped_;
    stats.write_errors = write_errors_;
    stats.min_size = min_size_;
    stats.max_records = max_records_;
    stats.elapsed_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start_time_).count();
}

// No CUPTI call here, for a measured reason: from an atexit handler on this stack both
// cuptiActivityFlushAll(1) and (0) block forever and leave a process that cannot be killed
// (docs/record.md §9). Losing the last buffers beats wedging the program being observed.
inline void Recorder::finish_at_exit(Stats& stats) {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    if (out_.is_open()) {
        out_.flush();
        out_.close();
    }
    fill_stats(stats);
    started_ = false;
}

inline bool trace_has_records(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    return static_cast<bool>(in) && in.tellg() > 0;
}

inline std::string file_size_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return "?";
    }
    const std::streamoff bytes = in.tellg();
    return std::to_string(static_cast<long long>(bytes)) + " B";
}

// `wrote` is printed by the caller: the two tools quote it from different places.
inline void print_summary(const Stats& stats) {
    std::printf("  MEMORY2 records seen %zu, kept %zu (%zu allocs, %zu frees)\n", stats.seen,
                stats.kept, stats.allocs, stats.frees);
    std::printf("  filtered: %zu allocs below %zu B, %zu frees without an alloc, "
                "%zu duplicate allocs\n",
                stats.small_allocs, stats.min_size, stats.orphan_frees, stats.duplicate_allocs);
    if (stats.over_cap != 0) {
        std::printf("  TRUNCATED at --max-records %zu: %zu records refused\n", stats.max_records,
                    stats.over_cap);
    }
    if (stats.write_errors != 0) {
        std::printf("  WRITE ERRORS: %zu records could not be written\n", stats.write_errors);
    }
    if (stats.cupti_dropped != 0) {
        std::fprintf(stderr,
                     "%s: WARNING: CUPTI dropped %zu of its own records (buffers too small "
                     "for the activity rate); the trace is incomplete\n",
                     program_name(), stats.cupti_dropped);
    }
    std::printf("  elapsed %.1f ms\n", stats.elapsed_ms);
}

}  // namespace recorder
}  // namespace poolkit

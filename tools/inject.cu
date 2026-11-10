// poolkit_inject: the CUDA_INJECTION64_PATH shim, which traces a CUDA program that was not modified,
// recompiled or wrapped. CUPTI only sees the process that subscribed, so the driver loads this DLL
// and InitializeInjection subscribes from inside it. Known limits: docs/record.md §9.

#include "record_core.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {

// Two atomics, not a mutex: the exit handler can run while another thread is inside
// InitializeInjection, and a deadlock there would take the target process down with it.
std::atomic<bool> g_recording{false};
std::atomic<bool> g_finished{false};

std::size_t env_size(const char* name, std::size_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || value == 0) {
        std::fprintf(stderr, "poolkit_inject: ignoring invalid %s='%s'\n", name, text);
        return fallback;
    }
    return static_cast<std::size_t>(value);
}

const char* program() { return "poolkit_inject"; }

// Never throw: an exception escaping into the CRT exit sequence is std::terminate inside
// somebody else's program, a far worse outcome than a missing trace.
void finish_recording() noexcept {
    poolkit::recorder::debug("finish_recording: entered");
    if (!g_recording.exchange(false)) {
        return;
    }
    if (g_finished.exchange(true)) {
        return;
    }
    const char* path = std::getenv("POOLKIT_RECORD_OUT");
    poolkit::recorder::Stats stats;
    poolkit::recorder::instance().finish_at_exit(stats);
    poolkit::recorder::debug("finish_recording: file closed");
    std::fprintf(stderr, "poolkit_inject: recording stopped\n");
    poolkit::recorder::print_summary(stats);
    if (path != nullptr) {
        std::fprintf(stderr, "poolkit_inject: wrote %s (%s)\n", path,
                     poolkit::recorder::file_size_text(path).c_str());
    }
    std::fflush(stderr);
}

}  // namespace

// extern "C" so the name is not mangled, dllexport so a Windows DLL really exports it. The
// parameter: CUPTI's own sample declares (void), older docs say it takes the export-table pointer;
// on x64 the extra argument lands in a register the callee may ignore, so both forms work.
#if defined(_WIN32)
extern "C" __declspec(dllexport) int InitializeInjection(void* pCudaGetExportTable)
#else
extern "C" __attribute__((visibility("default"))) int InitializeInjection(
    void* pCudaGetExportTable)
#endif
{
    (void)pCudaGetExportTable;

    if (g_finished.load() || g_recording.exchange(true)) {
        return 1;
    }
    poolkit::recorder::set_program_name(program());
    poolkit::recorder::debug("InitializeInjection: entered");

    const char* out = std::getenv("POOLKIT_RECORD_OUT");
    if (out == nullptr || *out == '\0') {
        g_recording.store(false);
        return 1;
    }

    const std::size_t min_size =
        env_size("POOLKIT_RECORD_MIN_SIZE", poolkit::recorder::kDefaultMinSize);
    const std::size_t max_records =
        env_size("POOLKIT_RECORD_MAX_RECORDS", poolkit::recorder::kDefaultMaxRecords);
    const std::size_t flush_ms = env_size("POOLKIT_RECORD_FLUSH_MS", 100);
    const std::size_t buffer_kb = env_size("POOLKIT_RECORD_BUFFER_KB", 32);
    poolkit::recorder::instance().set_buffer_bytes(buffer_kb * 1024);

    if (std::atexit(&finish_recording) != 0) {
        std::fprintf(stderr,
                     "poolkit_inject: atexit registration failed; the trace would never be "
                     "flushed, so recording is NOT started\n");
        g_recording.store(false);
        return 1;
    }

    if (!poolkit::recorder::instance().start(out, min_size, max_records,
                                            static_cast<std::uint32_t>(flush_ms))) {
        std::fprintf(stderr,
                     "poolkit_inject: CUPTI could not be initialised; the program runs "
                     "unrecorded. Check that the CUPTI DLL is on PATH (or that CUPTI is "
                     "installed next to the driver).\n");
        g_recording.store(false);
        return 1;
    }

    std::fprintf(stderr,
                 "poolkit_inject: recording MEMORY2 into '%s' (min-size %zu B, max-records "
                 "%zu, flush every %zu ms)\n",
                 out, min_size, max_records, flush_ms);
    return 1;
}

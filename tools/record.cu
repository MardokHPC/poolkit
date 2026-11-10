// poolkit_record: turn a real CUDA workload into a trace the rest of the toolchain can read. The
// callbacks, filters and JSONL writer live in record_core.hpp; this file is the CLI. Exit codes:
// 0 trace written, 2 usage/CUPTI/load error, 4 no records, 77 no CUDA device, else the workload's.

#include "record_core.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cstdint>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <process.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using poolkit::recorder::Recorder;
using poolkit::recorder::Stats;

using WorkloadFn = int (*)(int, char**);

void* load_library(const std::string& path, std::string& error) {
#if defined(_WIN32)
    HMODULE handle = LoadLibraryA(path.c_str());
    if (handle == nullptr) {
        error = "LoadLibraryA failed (" + std::to_string(static_cast<long long>(GetLastError())) +
                "): " + path;
    }
    return reinterpret_cast<void*>(handle);
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW);
    if (handle == nullptr) {
        const char* reason = dlerror();
        error = std::string("dlopen failed: ") + (reason != nullptr ? reason : "unknown");
    }
    return handle;
#endif
}

WorkloadFn find_workload(void* handle, std::string& error) {
    WorkloadFn symbol = nullptr;
#if defined(_WIN32)
    FARPROC proc = GetProcAddress(reinterpret_cast<HMODULE>(handle), "poolkit_record_workload");
    // memcpy, not a cast: MSVC warns C4191 on FARPROC -> function pointer, and /W4 /WX is not negotiable.
    static_assert(sizeof(symbol) == sizeof(proc), "function pointer sizes differ");
    std::memcpy(&symbol, &proc, sizeof(symbol));
#else
    symbol = reinterpret_cast<WorkloadFn>(dlsym(handle, "poolkit_record_workload"));
#endif
    if (symbol == nullptr) {
        error = "the library does not export "
                "'extern \"C\" int poolkit_record_workload(int, char**)'";
    }
    return symbol;
}

void unload_library(void* handle) {
#if defined(_WIN32)
    (void)FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
    (void)dlclose(handle);
#endif
}

// Kept because the CLI has to behave honestly for the command form too: it runs the program, returns
// its code, and the caller then reports that no in-process CUPTI record can come from it.
int spawn_and_wait(const std::vector<std::string>& command) {
    std::vector<const char*> argv;
    argv.reserve(command.size() + 1);
    for (const std::string& token : command) {
        argv.push_back(token.c_str());
    }
    argv.push_back(nullptr);

#if defined(_WIN32)
    const intptr_t status = _spawnvp(_P_WAIT, argv[0], argv.data());
    if (status == -1) {
        std::fprintf(stderr, "poolkit_record: cannot run '%s'\n", argv[0]);
        return 2;
    }
    return static_cast<int>(status);
#else
    const pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "poolkit_record: fork failed\n");
        return 2;
    }
    if (pid == 0) {
        execvp(argv[0], const_cast<char**>(argv.data()));
        std::fprintf(stderr, "poolkit_record: cannot run '%s'\n", argv[0]);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return 2;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
}

void print_usage(const char* program) {
    std::printf(
        "usage: %s -o <trace.jsonl> [options] --workload <lib> [-- args...]\n"
        "\n"
        "Records every cudaMalloc/cudaFree a workload performs, through CUPTI\n"
        "CUPTI_ACTIVITY_KIND_MEMORY2, and writes a docs/trace-format.md JSONL.\n"
        "\n"
        "Options:\n"
        "  -o, --out <f>          trace to write (required)\n"
        "  --min-size <bytes>     drop allocations below this (default 4096)\n"
        "  --max-records <n>      stop after n records (default 2000000)\n"
        "  --drop-internal        not implemented: see docs/record.md\n"
        "  --workload <lib>       shared library exporting\n"
        "                         'extern \"C\" int poolkit_record_workload(int, char**)'\n"
        "  -- [args...]           arguments for --workload; without --workload, the\n"
        "                         command to spawn (invisible to CUPTI: exit 4)\n"
        "  -h, --help             this help\n"
        "\n"
        "CUPTI instruments the calling process only. That is why the workload is a\n"
        "library loaded in-process, not a child process. To record an unmodified\n"
        "program, use poolkit_inject with CUDA_INJECTION64_PATH (docs/record.md).\n"
        "\n"
        "Exit codes: 0 trace written, 2 usage/CUPTI/load error, 4 no records,\n"
        "77 no CUDA device (suite skips), otherwise the workload's own code.\n",
        program);
}

bool parse_size(const std::string& text, std::size_t& out) {
    if (text.empty()) {
        return false;
    }
    std::uint64_t value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(ch - '0');
        if (value > (18446744073709551615ull - digit) / 10ull) {
            return false;
        }
        value = value * 10ull + digit;
    }
    if (value > static_cast<std::uint64_t>(static_cast<std::size_t>(-1))) {
        return false;
    }
    out = static_cast<std::size_t>(value);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    poolkit::recorder::set_program_name("poolkit_record");

    std::string out_path;
    std::string workload_path;
    std::vector<std::string> command;
    bool drop_internal = false;
    std::size_t min_size = poolkit::recorder::kDefaultMinSize;
    std::size_t max_records = poolkit::recorder::kDefaultMaxRecords;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto take = [&](std::string& target) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "poolkit_record: missing value for %s\n", arg.c_str());
                return false;
            }
            target = argv[++i];
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-o" || arg == "--out") {
            if (!take(out_path)) {
                return 2;
            }
        } else if (arg == "--workload") {
            if (!take(workload_path)) {
                return 2;
            }
        } else if (arg == "--min-size") {
            std::string value;
            if (!take(value)) {
                return 2;
            }
            if (!parse_size(value, min_size)) {
                std::fprintf(stderr, "poolkit_record: invalid --min-size '%s'\n", value.c_str());
                return 2;
            }
        } else if (arg == "--max-records") {
            std::string value;
            if (!take(value)) {
                return 2;
            }
            if (!parse_size(value, max_records) || max_records == 0) {
                std::fprintf(stderr, "poolkit_record: invalid --max-records '%s'\n", value.c_str());
                return 2;
            }
        } else if (arg == "--drop-internal") {
            drop_internal = true;
        } else if (arg == "--") {
            for (int k = i + 1; k < argc; ++k) {
                command.emplace_back(argv[k]);
            }
            break;
        } else {
            std::fprintf(stderr, "poolkit_record: unknown argument '%s'\n\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (out_path.empty()) {
        std::fprintf(stderr, "poolkit_record: -o/--out is required\n\n");
        print_usage(argv[0]);
        return 2;
    }
    if (workload_path.empty() && command.empty()) {
        std::fprintf(stderr,
                     "poolkit_record: give --workload <lib> (with optional -- args) or "
                     "-- <command>\n\n");
        print_usage(argv[0]);
        return 2;
    }
    if (drop_internal) {
        std::fprintf(stderr,
                     "poolkit_record: --drop-internal is not implemented; recording every "
                     "allocation >= --min-size (see docs/record.md)\n");
    }

    // No device means the suite skips (exit 77), not that the recorder failed.
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "poolkit_record: no CUDA device (%s)\n",
                     cudaGetErrorString(count_status == cudaSuccess ? cudaErrorNoDevice
                                                                   : count_status));
        return 77;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        std::fprintf(stderr, "poolkit_record: cannot query device 0\n");
        return 2;
    }

    Recorder& recorder = poolkit::recorder::instance();
    if (!recorder.start(out_path, min_size, max_records)) {
        std::fprintf(stderr,
                     "poolkit_record: CUPTI could not be initialised; no trace written. "
                     "Check that the CUPTI DLL/SO directory is on PATH/LD_LIBRARY_PATH.\n");
        return 2;
    }

    int workload_status = 0;
    if (!workload_path.empty()) {
        std::string error;
        void* handle = load_library(workload_path, error);
        if (handle == nullptr) {
            std::fprintf(stderr, "poolkit_record: %s\n", error.c_str());
            Stats ignored;
            (void)recorder.stop(ignored);
            return 2;
        }
        WorkloadFn workload = find_workload(handle, error);
        if (workload == nullptr) {
            std::fprintf(stderr, "poolkit_record: '%s': %s\n", workload_path.c_str(),
                         error.c_str());
            unload_library(handle);
            Stats ignored;
            (void)recorder.stop(ignored);
            return 2;
        }

        std::vector<char*> workload_argv;
        workload_argv.reserve(command.size());
        for (std::string& token : command) {
            workload_argv.push_back(token.data());
        }
        workload_status = workload(static_cast<int>(workload_argv.size()),
                                   workload_argv.empty() ? nullptr : workload_argv.data());
        // Queued work is finished before the final flush, or a cudaFreeAsync record would be missing.
        (void)cudaDeviceSynchronize();
        unload_library(handle);
    } else {
        workload_status = spawn_and_wait(command);
    }

    Stats stats;
    const bool flushed = recorder.stop(stats);
    if (!flushed) {
        std::fprintf(stderr,
                     "poolkit_record: WARNING: the final CUPTI flush failed; records may be "
                     "missing from the end of the trace\n");
    }

    std::printf("poolkit_record: %s, sm_%d%d, %s\n", properties.name, properties.major,
                properties.minor, workload_path.empty() ? "spawned command" : "in-process workload");
    poolkit::recorder::print_summary(stats);

    if (stats.kept == 0) {
        (void)std::remove(out_path.c_str());
        if (!workload_path.empty()) {
            std::fprintf(stderr,
                         "poolkit_record: the workload produced no memory records. Was CUDA "
                         "actually used, and did it allocate at least --min-size %zu B?\n",
                         stats.min_size);
        } else {
            std::fprintf(stderr,
                         "poolkit_record: 0 records: CUPTI instruments this process only, and "
                         "the spawned child's CUDA calls are invisible here. Build the workload "
                         "as a shared library and pass it with --workload (docs/record.md "
                         "section 1), or record the program unmodified with "
                         "poolkit_inject.\n");
        }
        return 4;
    }

    std::printf("  wrote %s (%s)\n", out_path.c_str(),
                poolkit::recorder::file_size_text(out_path).c_str());

    // The workload's code wins, and the trace is already flushed and closed.
    return workload_status;
}

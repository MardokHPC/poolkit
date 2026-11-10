# poolkit

[![CI](https://github.com/MardokHPC/poolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/MardokHPC/poolkit/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
[![Release](https://img.shields.io/github/v/release/MardokHPC/poolkit?label=release&color=orange)](https://github.com/MardokHPC/poolkit/releases/latest)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76b900?logo=nvidia&logoColor=white)

**Size your GPU allocator from a trace, and gate it in CI.** Record what an unmodified CUDA program allocates,
find the size classes and chunk size that fit *that* workload, and fail the build when memory regresses — with the
growth attributed to one of three components. The analysis runs on a machine with no GPU.

> **poolkit is not the fastest.** On the committed trace `cudaMallocAsync` is 1.8× faster than our pool and uses less
> memory. Here is the comparison, with our loss in it:

| mode | wall (median) | peak | alloc p50 | free p50 |
|------|--------------:|-----:|----------:|---------:|
| **poolkit** | 4.96 ms | **68 MiB (exact)** | 400 ns | 600 ns |
| `cudaMalloc` per allocation | 10.00 ms | 46 MiB (sampled) | 2.3 µs | 1.8 µs |
| `cudaMallocAsync` | **2.72 ms** | 64 MiB (sampled) | 400 ns | 400 ns |

`traces/inference_transformer.jsonl`, 3 076 operations, RTX 2060 SUPER (sm_75), driver 610.88, CUDA 12.6, one process
at a time, 1 warmup + 5 runs per invocation, three invocations, median of the run medians. The three raw reports are
committed: [`docs/bench/final-run1.json`](docs/bench/final-run1.json),
[`…run2`](docs/bench/final-run2.json), [`…run3`](docs/bench/final-run3.json) — spread across the three: pool
4.21–5.05 ms, `cudaMalloc` 9.77–10.44 ms, `cudaMallocAsync` 2.68–2.86 ms.

We lose because every `free` records a `cudaEvent` (~600 ns × 1 538 frees ≈ 0.9 ms), while the driver's pool is
stream-ordered *inside* the driver and pays nothing on the host. That is architectural, not an implementation gap, and
we do not plan to chase it. What none of the alternatives can tell you is **where the bytes went**, **which component
grew between two releases**, and **why you are at 95%**.

---

## What it is

A measurement toolchain with an allocator inside it, not the other way round:

| Piece | Command | Answers |
|---|---|---|
| **recorder** | `poolkit_record` / `poolkit_inject` | "what does my application actually allocate?" — CUPTI `MEMORY2`. Injection takes a binary you did not rebuild (**Windows verified**, Linux untested); otherwise the workload is built as a shared library |
| **advisor** | `poolkit_advise` | "which size classes and chunk should I use for this workload, and what peak does that give?" |
| **gate** | `poolkit_gate` | "did this change grow memory, and in which component?" — verdict + exit code; the GPU job in CI is a template, because hosted runners have no GPU |
| **bench** | `poolkit_bench` | "pool vs `cudaMalloc` vs `cudaMallocAsync`, same trace, same footprint" |
| **library** | `#include <poolkit/pool.hpp>` | the reference implementation of the model: slab carving, deferred reuse, three metrics |

```bash
# 1. record a binary you did not rebuild (Windows injection, verified; Linux source-identical but untested)
set CUDA_INJECTION64_PATH=build-dev/poolkit_inject.dll
set POOLKIT_RECORD_OUT=trace.jsonl
my_inference_app.exe                       # runs exactly as before
#    no recompile, no wrapper. `poolkit_record -- <cmd>` does NOT record a child process:
#    CUPTI instruments only the process that subscribes to it, so a child yields 0 records and exit 4.

# 2. size the allocator for that workload, no GPU needed
poolkit_advise --trace trace.jsonl
#    -> 1 457 configurations swept, simulated peak per configuration,
#       one recommended config with the reason

# 3. gate it in CI
poolkit_gate --baseline base.json --current cur.json
#    exit 0 pass/improvement, 1 regression, 2 suspect (comparison not valid)
```

**What it is not.** Not a replacement for `cudaMallocAsync` (we are slower, see above). Not a profiler: it does not
time kernels, produce timelines or per-instruction counters — that is Nsight. Not multi-GPU, not thread-safe, not a
`realloc` implementation. Not framework-aware: PyTorch already reports on its own allocator, and poolkit's value there
is the trace-level sizing and the gate, not a replacement for `memory_snapshot()`.

The advisor refuses to recommend its own best row when that row wins by leaving half the bytes outside the pool:

```
| rank | growth | pow2 | chunk | max class | covered | k | internal | peak sim |
|    1 |   1.43 |   no | 1 MiB |     1 MiB | 49.650% | 23 |  6.555 % | 41.45 MiB |
overall best, not recommended. It reaches 41.45 MiB (2.18 MiB less), but only 49.650% of the requested bytes
would still be rounded by the table: the rest becomes one driver allocation per block, never reused.
The model prices bytes, not driver calls.
```

## When to use it

| You want | Use |
|---|---|
| To know **where** the memory went: class rounding, slab tail, resident-but-idle | **poolkit** (`snapshot()`, `poolkit_advise`) |
| CI to **fail** when peak memory regresses, with the component named | **poolkit** (`poolkit_gate`) |
| To record allocations of an **existing** CUDA program | **poolkit** (`poolkit_record`, `poolkit_inject`) |
| Size classes and chunk for *your* load, on a machine **without a GPU** | **poolkit** (`poolkit_advise` + `SimBackend`) |
| One-off allocation, no interest in fragmentation | `cudaMalloc` |
| Stream-ordered async allocation, opacity acceptable | `cudaMallocAsync` |
| Attribution **inside** a PyTorch process | `torch.cuda.memory_snapshot()` — but only there |
| Timeline, per-kernel counters, graph analysis | Nsight Systems / Compute |

poolkit does not replace `cudaMallocAsync` or a framework's caching allocator: it makes them **measurable**. PyTorch
tells you about PyTorch's allocator, inside PyTorch. `poolkit` takes a trace — recorded from a program you can run, or
built from your own allocation pattern — models the allocator, tells you which classes suit *that* load, and gives you
a reproducible verdict you can put in CI, including on a runner with no GPU.

## The three fragmentations

"Fragmentation" without an adjective means nothing, so the snapshot carries three numbers that answer three questions:

| Metric | Formula | Question |
|---|---|---|
| `internal_fragmentation` | `(Σ capacity_live − Σ size_live) / Σ size_live` | how much the class rounds a request up |
| `tail_fragmentation` | `Σ_chunk (chunk − n·c) / Σ bytes_resident` | bytes of a carved chunk no block can ever use |
| `resident_slack` | `bytes_free_in_pool / bytes_resident` | memory paid for and not used right now |

plus `external_fragmentation(request) = 1 − usable_free(request) / bytes_free_in_pool`, which cannot be a single number
because the same pool answers `0.0` for one request and `1.0` for another. `usable_free` counts only the class the pool
would actually use for that request — the smallest class ≥ `request`, exactly what `Pool::allocate` maps it to — so
free blocks sitting in a larger class are fragmentation, not free memory. Above `max_class_bytes` the request takes the
large path, `usable_free` is 0 and external fragmentation is not defined there. Verified identity, checked byte by byte
in the stress test:

```
bytes_resident == bytes_capacity_live + bytes_free_in_pool + quarantined_bytes + tail_bytes
```

## What we got wrong

Two numbers in this project were wrong, and both were caught by measuring.

**1. Predicted 18.03% internal waste, measured 23.9%.** (Both numbers capacity-normalised — waste over the capacity
handed out — because that is what the table can be scored against; the API's `internal_fragmentation` divides by the
*requested* bytes instead, and the two differ by construction.) The cost model assumed a log-uniform size distribution. On the
first realistic trace — large long-lived weights, many small short-lived activations — the measurement said
**23.915%**. Mechanism: 1 MiB weights landed just above the 920 KiB class and fell into the next one, paying 32%.
The fix was not to adjust the model but to add **powers of two** to the class table (as jemalloc does) plus
`extra_sizes` for known sizes. After: **15.234%** internal, replay tail waste 31.7% → 21.1%, and the *actually
resident* memory at the end of the replay **82 → 68 MiB**. Price, stated: 23 → 34 classes and a worst-case retention
bound of 46 → 68 MiB. Derivations in [`docs/cost-model.md`](docs/cost-model.md) §9.

**2. "`cudaMalloc` costs ~1.2 ms", measured 2.3 µs.** The 1.2 ms was the *maximum* over 10 000 samples — the tail of
heap expansion. The median per-allocation cost is **2.3 µs**; a fresh 2 MiB chunk is **~61 µs**. The honest headline is
**400 ns vs 2.3 µs (~6×)**, or **~152×** against a fresh chunk, not "a thousand times". See
[`docs/measurements-m6.md`](docs/measurements-m6.md).

## How it works

`cudaMalloc` granularity is 2 MiB (measured), so a block per allocation would waste 2 MiB for a 64 B request. Each
chunk is carved into fixed-size slots:

```
chunk C = 2 MiB ─┬─ slot c_i ─┬─ slot c_i ─┬─ ... ─┬─ tail (C mod c_i, never assignable)
                 └─────────────┴─────────────┴───────┘
class sizes: 256 B, 512 B, 768 B, 1 KiB, ... (geometric g=1.5 ∪ powers of two ∪ extra_sizes), every one a multiple of 256
free: a block returns to its class after a cudaEvent recorded on the stream of its last use reports completion
```

- `slab carving` keeps tail waste visible and bounded instead of hiding it in padding;
- `deferred reuse` never blocks and never synchronises: a pending block is skipped, not waited for;
- `Backend` is injected, so the entire pool logic — sizing, free lists, reuse, metrics — is tested on a CPU with
  `FakeBackend`/`SimBackend`, no GPU and no `nvcc`;
- failures fail closed: an unrecordable event **quarantines** the block forever (counted in `quarantined_bytes`) rather
  than handing over memory that might still be written, and `release_all()` **refuses** while any handle is alive.

## Install

```cmake
# A: add_subdirectory
add_subdirectory(third_party/poolkit)
target_link_libraries(my_app PRIVATE poolkit::poolkit)

# B: FetchContent, pinned to the released tag
FetchContent_Declare(poolkit GIT_REPOSITORY https://github.com/MardokHPC/poolkit.git GIT_TAG v1.0.0)
FetchContent_MakeAvailable(poolkit)

# C: installed package
find_package(poolkit 1.0 CONFIG REQUIRED)
```

Header-only INTERFACE target for the core (no CUDA header in `backend.hpp`, `metrics.hpp`, `size_classes.hpp`,
`slab.hpp`, `trace.hpp`); the CUDA backend is one translation unit (`src/pool.cu` + `pool.cuh`) and linking it brings
`cudart`; the tools (`bench`, `record`, `inject`) add `CUPTI`. Requires C++17;
CMake 3.18+ for the library, **3.25+** if you build the tools that need `CUDA::cupti`.

## Use

```cpp
#include <poolkit/pool.hpp>
#include <poolkit/pool.cuh>   // CudaBackend: the only CUDA header

poolkit::CudaBackend backend;
poolkit::Pool pool(backend, {.chunk_bytes = 2u << 20, .growth = 1.5});
auto buf = pool.allocate(1u << 20, stream);

launch_kernel<<<grid, block, 0, stream>>>(buf.get(), buf.size());

pool.free(std::move(buf), stream);   // records an event; reuse is deferred, never blocking

auto s = pool.snapshot();
printf("internal %.3f  tail %.3f  slack %.3f  hit %.1f%%\n",
       s.internal_fragmentation, s.tail_fragmentation, s.resident_slack, 100.0 * s.hit_rate);
```

## Limits

- **One stream per block.** Reuse is deferred with `cudaEvent`, but there is no cross-stream block migration and no
  `cudaStreamWaitEvent` between pools. `free()` without an argument **assumes** the work is ordered with the
  allocation stream — documented, not hidden.
- **Single device, single thread.** `Pool` is not thread-safe; no multi-GPU in v1.0.
- **No `realloc`, no coalescing.** A freed block stays in its class.
- **Memory is not returned to the driver** except through `release_all()`, which refuses while handles are alive.
  **Consequence, including at shutdown:** a `Pool` destroyed while a freed block's event is still pending keeps that
  chunk for the life of the process — the destructor does not fall back on a synchronising `cudaFree`. Queued work
  must be given a chance to complete (`poll()` after the stream is done) or the memory is not coming back. Quarantined
  blocks never come back at all: a leak, declared and counted.
- **Host-side metadata only** — no structure in device memory; cost is host RAM proportional to tracked blocks.
- **No "fast mode" without events.** It would put us at or ahead of `cudaMallocAsync` and move the synchronisation
  contract onto the caller. v1.1, cited and not implemented.
- **The recorder sees everything the process allocates**, runtime internals included, and cannot report the stream
  (`CUPTI_ACTIVITY_KIND_MEMORY2` has none: `stream` is always `0`). Injection loses the last partial CUPTI buffer
  (~570 operations): a program with fewer CUDA allocations than that may produce an **empty** trace, and there is no
  flush at exit — calling CUPTI from the exit path wedges the process on this stack.
- **No clock/throttle sampling.** The gate compares memory, not frequencies.

## Quality

- **108 GoogleTest cases** (103 host, 5 device) across **15 ctest entries in the host-only tree and 21 in the device
  tree**, plus a deterministic stress test: 10 000 random
  alloc/free operations with a fixed seed, an interval map checking for overlap and double free at every step, and the
  accounting identity above.
- Host-only suites run everywhere, including a CI job with **no CUDA toolkit installed**; device suites exit `77` when
  no GPU is present and are reported as `skipped`, never as green.
- Warnings are fatal everywhere: `/W4 /WX` (MSVC), `-Wall -Wextra -Wpedantic -Werror` (GCC/Clang host C++). The
  `.cu` files get `-Wall -Wextra -Werror` through `-Xcompiler`, **without `-Wpedantic`**: nvcc's own generated code
  emits `#line` directives that GCC flags, so the pedantic flag would fail every CUDA translation unit.
- `compute-sanitizer memcheck` and `racecheck` on the deferred-reuse tests: **0 errors, 0 hazards**, with the
  transcript and the exact commands in [`docs/sanitizer-1.0.md`](docs/sanitizer-1.0.md). That document also states what
  sanitizers **cannot** prove here — host-side lifetime bugs — and which test covers those instead.
- CI: `host-only` (ubuntu + windows, no CUDA), `cuda` (ubuntu 11.8 / 12.1, windows 12.9, sm_75), and a commented
  self-hosted GPU job template — hosted runners have no GPU, so device tests only ever *skip* there.
- Test counts and pass/fail are reproduced by `ctest`; the benchmark table is reproduced by `poolkit_bench`, and its
  three authoritative reports are committed in [`docs/bench/`](docs/bench) — `ctest` does not re-run the bench. Every
  other number cites the document that measured it. If a number here cannot be reproduced, that is a bug in this
  README.

## Reproduce

```bash
./build-host.bat        # VS 18, no CUDA needed -> configures, builds, runs ctest (15 entries)
./build-device.bat      # VS 2022 BuildTools + CUDA 12.6, sm_75 -> configures, builds, runs ctest (21 entries)

# the scripts set up the toolchain environment themselves; calling ctest by hand on
# Windows needs a Developer Prompt, otherwise poolkit_consumer_install cannot find rc.exe

# host tools live in build-host
build-host/poolkit_advise --trace traces/inference_transformer.jsonl
build-host/poolkit_gate   --baseline base.json --current cur.json

# GPU tools live in build-dev
build-dev/poolkit_bench  --trace traces/inference_transformer.jsonl --mode all --repeat 5 --warmup 1
build-dev/poolkit_record -o out.jsonl --workload <your_workload.dll>
```

The table at the top is the median over three invocations of `poolkit_bench --mode all --repeat 5 --warmup 1`; the
reports it produced are the three committed JSONs in [`docs/bench/`](docs/bench). Methodology (fixed footprint between
modes, one process at a time, warmup + medians, never a single run) in [`docs/bench.md`](docs/bench.md).

## Docs

- [`docs/cost-model.md`](docs/cost-model.md) — expected waste, slab tail, retention cost, why powers of two
- [`docs/metrics.md`](docs/metrics.md) — the three fragmentations with worked examples
- [`docs/trace-format.md`](docs/trace-format.md) — the JSONL schema and the replay rules
- [`docs/advise.md`](docs/advise.md) · [`docs/gate.md`](docs/gate.md) · [`docs/record.md`](docs/record.md) · [`docs/bench.md`](docs/bench.md)
- [`docs/measurements-m4.md`](docs/measurements-m4.md) · [`docs/measurements-m6.md`](docs/measurements-m6.md) — raw measurements
- [`docs/sanitizer-1.0.md`](docs/sanitizer-1.0.md) — memcheck/racecheck transcript, and what it does not prove
- [`docs/spike-cupti.md`](docs/spike-cupti.md) — the CUPTI feasibility spike that decided the recorder
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — two toolchains on Windows, how to regenerate every number

## License

MIT — see [LICENSE](LICENSE). Author: MardokHPC.

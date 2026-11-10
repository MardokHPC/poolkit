# Changelog

All notable changes to this project are documented in this file.

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every number quoted below is reproduced by a tool in this repository and pinned by
a test or by the document named next to it; `CONTRIBUTING.md` says how to
regenerate each one. Numbers measured on the reference machine (RTX 2060 SUPER,
sm_75, driver API 13.3 — display driver 610.88) say so.

## [Unreleased]

Nothing yet. `1.0.0` is the first release; the tag is created by the maintainer,
not by the build.

## [1.0.0] - 2026-11-10

First release. C++17, and a CUDA backend that is the only code in the project that
names `cudaMalloc`. What has to be linked, exactly, so nobody has to guess:

* the **core** (`poolkit::poolkit`) is an INTERFACE target: headers only, no
  link-time dependency, no toolkit required, nothing to add to your link line;
* the **CUDA backend** is a single translation unit, `src/pool.cu` (declarations in
  `pool.cuh`), built into the static library `poolkit_cuda`, which links `cudart`
  publicly — link `poolkit_cuda` and `cudart` comes with it;
* the **tools** link CUDA by construction: `poolkit_bench`, `poolkit_measure_m4`
  and `examples/basic_usage` through `poolkit_cuda` (hence `cudart`),
  `poolkit_record` against `CUDA::cupti` and `cudart`, and `poolkit_inject`
  against `CUDA::cupti` **only** — it must not initialise the runtime, or it would
  allocate before CUPTI is subscribed.

### Added

**Library**

* `poolkit::Pool` over an injected `Backend`: slab carving from a fixed chunk size,
  per-size-class free lists, deferred reuse through events instead of a device
  synchronisation (`free` never blocks), a large path for requests above
  `max_class_bytes`, and a fail-closed quarantine for blocks whose event could not
  be recorded. Metadata lives on the host, never in device memory.
* `poolkit::SizeClassTable`: the geometric progression with powers of two merged in
  and caller-supplied `extra_sizes`. 34 classes at `g = 1.5` on `[256 B, 2 MiB]`;
  `docs/cost-model.md` derives the table and §9 measures the union.
* `poolkit::CudaBackend` (`src/pool.cu`, a few dozen lines of driver glue) and
  `FakeBackend`/`SimBackend`, so the whole allocator runs on a CPU with no GPU, no
  toolkit and no nvcc.
* `PoolSnapshot`: the three fragmentation components as three numbers
  (`internal_fragmentation`, `tail_fragmentation`, `resident_slack`), the
  request-dependent `external_fragmentation(request)`, lifetime counters
  (reuse/growth/recovered/large path), host timings, and per-class statistics
  (`docs/metrics.md`).
* `DevicePtr`: move-only RAII handle; `free` is non-blocking, `release_all()`
  refuses (and counts the refusal) when a handle is still live.
* Trace format and reader (`docs/trace-format.md`, `include/poolkit/trace.hpp`):
  JSONL, `op/size/ptr/stream/ordinal`, strict parser with line numbers,
  deterministic CPU replay.

**Tools**

* `poolkit_advise`: sweeps growth, chunk size, `max class` and the power-of-two
  union on a trace, replays every candidate, and recommends a configuration with the
  peak as the criterion. Prints `covered` next to `internal` so a small `max class`
  cannot look good by sending the expensive requests to the large path
  (`docs/advise.md`).
* `poolkit_gate`: exit `0` pass / `1` regression / `2` not comparable, refusing to
  compare different traces or different configurations instead of guessing.
* `poolkit_bench`: the same trace through poolkit, `cudaMalloc` and
  `cudaMallocAsync`, 1 warmup + 5 runs, medians and p99, config in the header and in
  the JSON (`docs/bench.md`).
* `poolkit_record`: CUPTI `MEMORY2` recorder. Records a workload library loaded
  in-process, filters coherently (`--min-size`, orphan frees, `--max-records`) and
  writes a trace the parser accepts (`docs/record.md`).
* `poolkit_inject` (Windows): the `CUDA_INJECTION64_PATH` shim, which records a CUDA
  program that was not modified, recompiled or wrapped — `set
  CUDA_INJECTION64_PATH=...\poolkit_inject.dll`, `set POOLKIT_RECORD_OUT=trace.jsonl`,
  run the program. Shares the recording core with `poolkit_record`
  (`tools/record_core.hpp`), so the two cannot produce different trace dialects
  (`docs/record.md` §9).
* `poolkit_sweep_growth`, `poolkit_measure_classes`, `poolkit_measure_m4`: the
  regenerators for `docs/cost-model-sweep.md`, §9 of `docs/cost-model.md` and
  `docs/measurements-m4.md`.
* `examples/basic_usage.cu`: the shortest honest program (create, allocate, print
  the snapshot, free, `release_all()`), built and run by `ctest` so the example in
  the README cannot rot away from the headers.

**Build, tests, CI**

* `find_package(poolkit CONFIG)` from an installed tree, exercised by an
  out-of-tree consumer test.
* 15 host test cases in the host-only tree and 21 ctest entries in the device tree —
  the host-only 15 plus 6 that need a card; the device ones exit `77` and are
  registered as `SKIP_RETURN_CODE`, so a runner without a card reports SKIPPED and
  never a green that hides a test that did not run.
* Warnings are errors everywhere (`/W4 /WX`, also through `-Xcompiler` on the `.cu`
  files and on both CUPTI libraries).
* CI with three jobs: `host-only` (ubuntu-22.04 + windows-2022, no CUDA toolkit at
  all) and `cuda` (ubuntu-22.04 with CUDA 11.8 and 12.1, windows-2022 with 12.9),
  plus a commented `gpu` template for a self-hosted runner. `POOLKIT_REQUIRE_CUPTI`
  turns "CMake was too old to expose CUDA::cupti" into a configure error instead of
  a job that silently stops testing the recorder.
* `build-host.bat` / `build-device.bat`: the two toolchains on one machine
  (VS 18 for the host-only build, VS 2022 BuildTools + CUDA 12.6 for the device
  build), configure → build → ctest, no PATH surgery.
* `docs/`: cost model, metrics, trace format, advise, gate, bench, recorder, the M4
  and M6 measurement logs, the CUPTI spike write-up.

### Changed

* **The size class table is the union of the geometric progression and powers of
  two.** Measured on `traces/inference_transformer.jsonl`: `k` 23 → 34, internal
  waste 23.915 % → 15.234 % (−8.68 pp, −36.3 % relative, and back inside the ±20 %
  band around the log-uniform prediction), tail 0.962 % → 0.543 %, worst-case
  retention `k·C` 46 → 68 MiB, and the end-of-replay resident set 82.0 → 68.0 MiB.
  The +22 MiB of worst-case retention is the price and is stated in
  `docs/cost-model.md` §9.4, not hidden.
* **Peak accounting includes the large path.** `TraceReplayResult::peak_bytes` is
  now *chunks resident + bytes of live large-path blocks*: an allocation above
  `max_class_bytes` goes straight to the driver and is not part of
  `bytes_resident`, so a peak that ignored it rewarded a configuration for
  *stopping to pool* the expensive requests. `poolkit_advise` reports the share of
  traffic that is subject to it as `covered`.
* `poolkit_bench --json` writes **schema 2**: the schema-1 analysis document plus a
  required `bench` section. Schema-1 readers are unaffected, a schema-2 document
  without `bench` is rejected, and `poolkit_gate` accepts both schemas but refuses
  to compare one against the other.
* `cudaMallocAsync` is recorded as one `ALLOCATION`/`RELEASE` pair per application
  call: the driver's pool suballocations do not appear in the trace, and no special
  mapping was needed (`docs/record.md` §2).
* The recorder writes its JSONL incrementally, one line per accepted operation,
  instead of buffering the whole run: a process that dies without a clean shutdown
  keeps everything CUPTI already delivered.
* `poolkit_record -- <command>` runs the child and exits `4` saying it recorded
  nothing. It is a measurement of CUPTI's in-process scope, not a limitation to
  paper over; the answer for unmodified binaries is `poolkit_inject`.

### Fixed

* **"`cudaMalloc` costs 1.2 ms" was not the cost of a `cudaMalloc`.** That figure
  was the *maximum* of 10 000 growth-path samples from M4 — the tail where the
  driver expands its heap, including the process's first chunk — and it was quoted
  as if it were typical. Measured in M6b (`docs/measurements-m6.md`): the pool's
  growth step (a fresh 2 MiB chunk) has a **median of 62 µs** (p99 119 µs, max
  136–149 µs), while `cudaMalloc` called directly with the trace's own sizes has a
  median of **2.3 µs**. The honest statement is now "~62 µs per pool growth, ~2.3 µs
  per `cudaMalloc` on sizes the driver serves from its own cache", and the M4
  document keeps the old number labelled as a maximum.
* **The advisor could recommend a table by making its own metric look better.** A
  smaller `max class` lowers `internal` because the requests above it leave the
  histogram — and land on the large path, one driver allocation per block, never
  reused. The `covered` column, the peak including the large path, and the
  "overall best, not recommended" note exist because of that; `docs/advise.md` §3
  states what each column is *not*.
* **The recorder could write traces the rest of the toolchain refuses.** A free
  without its alloc, or two live blocks with the same address, made the parser
  throw. The filters are applied to both sides of an allocation (a dropped tiny
  alloc drops its free; a duplicate address is refused and counted), so every
  trace `poolkit_record` writes replays.
* **The injection shim's first design wedged the program it recorded.** Flushing
  CUPTI from an `atexit` handler blocks forever on Windows + CUDA 12.6
  (`cuptiActivityFlushAll(1)` and `(0)` alike) and leaves a process that
  `taskkill /F` cannot terminate. The exit path now makes no CUPTI call at all and
  relies on CUPTI's buffer-full delivery with small buffers; the cost — the last
  partial buffer — is documented in `docs/record.md` §9.1 with the transcript.
* Doubles in every JSON writer and reader are formatted and parsed in the classic
  locale: a locale with a decimal comma silently corrupted a CSV once, and `,` in a
  JSON number is not a number.

### Notes on scope

* Not thread-safe, one host thread, declared rather than hidden.
* The recorder records allocation traffic: no kernels, no memcpys, no streams
  (`stream` is always `0` because `CUpti_ActivityMemory3` does not carry it), no
  timestamps in the schema.
* `poolkit_inject` is built and tested on Windows only; the Linux `.so` compiles
  from the same source but has not been run on a Linux driver.
* `--drop-internal` is not implemented and says so: the heuristic also deletes a
  workload's long-lived weights, which is exactly what the tool exists to measure.

[Unreleased]: https://github.com/MardokHPC/poolkit/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/MardokHPC/poolkit/releases/tag/v1.0.0

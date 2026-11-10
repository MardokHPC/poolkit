# M6b measurements: the recorder and the benchmark

Every number on this page was produced on this machine, by the binaries in this
repository, with the commands shown. Nothing is estimated; where a measurement
could not be taken, the text says so. The methodology is
`DEVELOPMENT_PLAN.md` §Metodologia di misura, and the tools implement it —
[`bench.md`](bench.md) §1 maps each rule to the code that enforces it.

## Environment

| field | value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 SUPER, 8192 MiB |
| compute capability | sm_75 |
| driver (nvidia-smi) | 610.88 |
| driver (`cudaDriverGetVersion`) | 13030 (= CUDA 13.3) |
| toolkit / nvcc | CUDA 12.6.20 (`v12.6`), sm_75 |
| runtime actually loaded | **12060** (= 12.6) with `...\CUDA\v12.6\bin` first in `PATH`; the plain `PATH` on this machine resolves `cudart64_12.dll` from `v12.1` and the header prints **12010**. The bench reports the runtime it loaded for exactly this reason |
| host compiler | MSVC 19.44 (VS 2022 BuildTools, device build), MSVC 19.51 (host build) |
| free memory at measurement time | 7471 MiB |

Methodology for every timed number: **1 warmup + 5 runs**, `std::chrono::steady_clock`,
reported as the median of the per-run medians plus the smallest min and the
largest max. One process at a time, GPU otherwise idle (desktop only). Three
separate process invocations are reported for the benchmark, because the
between-process spread is larger than the within-process spread.

---

## 1. The recorder on `record_target`

Command (from `build-dev`):

```text
poolkit_record -o record_target.jsonl --workload poolkit_record_target_workload.dll
```

`tests/device/record_target.cu` allocates 4 MiB, 64 KiB, 1 MiB and 1 KiB, runs a
kernel on the first block, and frees everything.

Output, verbatim:

```text
record_target: 1 cycles, 4 allocations x 4194304 + 65536 + 1048576 B tracked
poolkit_record: NVIDIA GeForce RTX 2060 SUPER, sm_75, in-process workload
  MEMORY2 records seen 8, kept 6 (3 allocs, 3 frees)
  filtered: 1 allocs below 4096 B, 0 frees without an alloc, 0 duplicate allocs
  elapsed 135.3 ms
  wrote record_target.jsonl (403 B)
```

| quantity | value |
|---|---|
| CUPTI records delivered | 8 (4 allocs, 4 frees) |
| records written | 6 (3 allocs, 3 frees) |
| filtered | 1 allocation below `--min-size 4096` **and its free** — no orphan free in the file |
| orphan frees / duplicate allocs | 0 / 0 |
| file size | 403 B, 6 lines (67.2 B per line) |
| elapsed, tool's own clock | 115.3 / 105.1 / 100.2 ms on three consecutive runs (135.3 ms on the first run after a build); ~100 ms of that is CUDA context creation inside the workload |
| replay | `poolkit_record_check`: 6 ops, 3 allocs of the expected sizes, 3 paired frees, **peak 8.00 MiB**, 3 live blocks at the peak |

The replay peak is 8.00 MiB = 4 MiB (large path, `max_class_bytes` is 2 MiB) +
2 × 2 MiB chunks (one for the 64 KiB class, one for the 1 MiB class) — the same
arithmetic the trace describes, which is what makes "the trace is the workload"
a check and not a hope.

### `cudaMallocAsync` is reported by the same record kind

Same workload with `-- 1 async` (allocation through
`cudaMallocAsync`/`cudaFreeAsync` on a private stream):

```text
record_target: 1 cycles (cudaMallocAsync), 4 allocations x 4194304 + 65536 + 1048576 B tracked
poolkit_record: ...
  MEMORY2 records seen 8, kept 6 (3 allocs, 3 frees)
  wrote async.jsonl (397 B)
```

The three sizes are identical to the `cudaMalloc` run (4194304 / 65536 /
1048576 B), `poolkit_record_check` passes, and **no special mapping was needed**:
`CUPTI_ACTIVITY_KIND_MEMORY2` reports the driver's pool allocator as an ordinary
`ALLOCATION`/`RELEASE` pair with the requested size. The trace cannot see the
driver pool's own suballocations, which is a documented limit
([`record.md`](record.md) §4).

### Overhead of recording

Same workload with and without the recorder, median of the total process time.
The workload is built both ways from one source, so the only difference is the
recorder.

| workload | standalone | recorded | difference |
|---|---:|---:|---:|
| `record_target`, 1 cycle (8 memory ops) | 144.8 ms | 161.2 ms | **+16.5 ms** |
| `record_target 500`, 500 cycles (4000 memory ops) | 427.5 ms | 451.7 ms | **+24.2 ms** |

Decomposed: **~15–17 ms fixed** (CUPTI init + final flush + writing the file) and
**≈ 2.6–4.8 µs per recorded operation** — the range is two measurement sessions on
the same machine, 3000 written lines apart. The 500-cycle trace is 3000 lines /
209 390 B (69.8 B per line). Without subtracting the fixed cost the naive figure
is 8.1 µs per written line; the marginal one is the number that matters for a
long run. A 4-allocation program is therefore dominated by the fixed cost, and
that is the honest reading of the 115 ms above.

### Truncation and filtering keep the trace valid

| probe | result |
|---|---|
| `--max-records 4` | 3 allocs + 1 free written, `TRUNCATED at --max-records 4: 2 records refused`, and `poolkit_advise --trace` replays it (peak 8.00 MiB, 3 blocks) |
| `--min-size 0` | all 8 records written (4 allocs, 4 frees), file 535 B; `record_check` then correctly reports that the 1024 B allocation is present |
| `-- poolkit_record_target.exe` (child process) | 0 records, exit 4, and the message naming the reason: CUPTI instruments the calling process only |

---

## 2. The benchmark on the committed trace

Command (from `build-dev`, `...\CUDA\v12.6\bin` first in `PATH`):

```text
poolkit_bench --trace ..\traces\inference_transformer.jsonl --repeat 5 --warmup 1 --json bench.json
```

Trace: `traces/inference_transformer.jsonl`, 3076 ops (1538 allocs, 1538 frees),
sha256 `31fc9eff…73b10`, pool config = shipped default (`g = 1.5`, powers of two,
chunk 2 MiB, max class 2 MiB, 34 classes).

### Three invocations, wall time (ms)

| invocation | pool | cudamalloc | cudaasync |
|---|---:|---:|---:|
| 1 (`final-run1.json`) | 4.955 | 9.998 | 2.684 |
| 2 (`final-run2.json`) | 5.048 | 10.444 | 2.859 |
| 3 (`final-run3.json`) | 4.213 | 9.767 | 2.723 |
| **median of the three** | **4.955** | **9.998** | **2.723** |

Those three rows are read from the three committed reports under `docs/bench/`
(`final-run1.json`, `final-run2.json`, `final-run3.json`): one file per process
invocation, each `--repeat 5`, and every number here is the median of that file's
five run medians. **The JSON files are the authoritative artifacts** — if this
table and a bench JSON ever disagree, the JSON is the measurement.

A previous session on the same machine and with the same method reported
**4.811 / 10.520 / 2.656 ms** (median of three). The two sessions differ by
3.0 % / 5.0 % / 2.5 %, the same order as the run-to-run spread of this session
(below): they are two readings of the same measurement, not two different
results — and the committed JSON files are the ones to quote.

Within one invocation the five runs are tight (run 3: pool 4.010–4.321 ms); the
between-process spread is larger (pool medians 4.213–5.048 ms, `cudaMalloc`
9.767–10.444 ms, `cudaMallocAsync` 2.684–2.859 ms). That is why the headline is
a median of medians and not one run.

**Why the warmup is not optional.** The same command with `--warmup 0 --repeat 1`
— one cold run per mode — measures the pool at **103.8 ms** and `cudaMalloc` at
10.5 ms: the first replay pays the driver's one-time heap reservation for the 34
chunks, a 20× error on the pool's wall time. The plan's rule 2 exists precisely
to keep this number out of a published table, and the tool's default is 1 warmup
+ 5 runs for that reason.

### Peak memory

| mode | peak | how |
|---|---:|---|
| pool | **68.00 MiB** | exact: 34 chunks × 2 MiB, the retention bound of this trace |
| cudamalloc | **46.00 MiB** | `cudaMemGetInfo` every 64 ops (lower bound) |
| cudaasync | **64.00 MiB** | same sampling, plus the driver pool's lazy release |

Exact requested-bytes peak of the trace, computed from the file alone: 37.92 MiB.
The gap to `cudaMalloc`'s 46 MiB is the driver's 2 MiB granularity on the large
blocks, which a requested-bytes sum cannot see — i.e. the sampled measurement is
doing its job.

`poolkit_gate` on the bench JSON reports `peak_resident_bytes: 71303168` for the
analysis block, which is 68.00 MiB: the bench's exact peak and the advisor's
replay peak agree, as they must (same trace, same config, same replay rules).

### Per-operation latency (median / p99, median over the 3 invocations)

| mode | alloc p50 | alloc p99 | free p50 | free p99 |
|---|---:|---:|---:|---:|
| pool (all allocations) | 400 ns | ~62 µs | 600 ns | ~800 ns |
| cudamalloc | 2.3 µs | 57.2 µs | 1.8 µs | 83.4 µs |
| cudaasync (enqueue only) | 400 ns | 800 ns | 400 ns | 600 ns |

Pool paths, identical in every invocation (the split is a property of the trace
and the config: 1496 hit + 42 growth = 1538):

| path | ops | median | p99 | min | max |
|---|---:|---:|---:|---:|---:|
| hit (recycled block) | 1496 (97.3 %) | **400 ns** | 500 ns | 300 ns | 33.8 µs |
| growth, total | 42 (2.7 %) | **57.3 µs** | 114.2 µs | 100 ns | 244.3 µs |
|   new slot, existing chunk | 8 (0.5 %) | 300 ns | 300 ns | 100 ns | 700 ns |
|   fresh chunk (`cudaMalloc` + carve) | 34 (2.2 %) | **60.8 µs** | 114.2 µs | 47.9 µs | 244.3 µs |
|   large path | 0 (0.0 %) | — | — | — | — |
| free (event record) | 1538 (100 %) | **600 ns** | 800 ns | 500 ns | 18.9 µs |

**The split is the finding.** "Growth" is not one thing: 34 of its 42 operations
are chunk expansions — a 2 MiB `cudaMalloc` plus the carving of ~900 block
records — and they cost ~61 µs each; the other 8 are new slots in an existing
chunk at ~300 ns. Without the split, a single "growth" median of 57 µs would say
nothing except "this bucket is slow".

---

## 3. Comparison with the M4 numbers

| M4 number (`measurements-m4.md`) | M6b result | verdict |
|---|---|---|
| `cudaMalloc` ≈ **1.2 ms** (the max of the growth path) | fresh chunk: median **60.8 µs**, p99 114 µs, max 176–244 µs. `cudaMalloc` driven directly on the trace's own sizes: median **2.3 µs** | **not reproduced as a median**. 1.2 ms was the *max* of 10 000 growth-path samples, i.e. the tail where the driver expands its heap (and it includes the process's first chunk), not the cost of a `cudaMalloc`. The M4 doc already labelled it "growth max"; the M6b measurement shows where the rest of that distribution lives |
| hit `allocate`+`free` ≈ **1 µs** | hit allocate **400 ns** + free **600 ns** = **1000 ns** | **reproduced exactly**, and now decomposed |
| `free` alone ≈ **600 ns** | free median **600 ns** in every invocation | **reproduced exactly** |
| `cudaEventQuery` **100–200 ns** | not re-measured (the bench does not call it directly). Indirectly: the 8 new-slot growth ops at ~250 ns include the pending-list scan, so no evidence of a query storm | **unchanged, not re-verified** |
| granularity **2 MiB** | pool peak 68.00 MiB = exactly 34 × 2 MiB; `cudaMalloc` peak 46 MiB vs 37.92 MiB of requested bytes | **consistent** |

### The headline number, corrected

M4 proposed "~1 µs vs ~1.2 ms, a 1000× gap". The measured statement is:

* an allocation served from a recycled block costs **~400 ns**, a `free` **~600 ns**;
* an allocation that has to grow the pool costs **~61 µs** (median), with a p99
  near **114 µs** and a worst case of **244 µs** on this machine today;
* the ratio is **~152×** on the median (60.8 µs ÷ 400 ns), not 1000×. 1000× is the
  ratio against the *worst* single event M4 saw, which is a different claim.

That does not weaken the argument for the pool (2.0× faster end-to-end than
`cudaMalloc` on this trace, at the cost of 68 MiB held instead of 46), but a
published number should be the one the tool reproduces.

### End-to-end, which is what a user feels

| mode | wall (median of 3 invocations) | peak | speed vs `cudaMalloc` |
|---|---:|---:|---:|
| `cudaMalloc` | 9.998 ms | 46.00 MiB | 1.0× |
| poolkit | 4.955 ms | 68.00 MiB | **2.0× faster, +22 MiB** |
| `cudaMallocAsync` | 2.723 ms | 64.00 MiB | 3.7× faster, but asynchronous: the wall includes the final synchronisation, the latency columns are enqueue costs |

`cudaMallocAsync` is faster than poolkit on this trace and that is expected: the
pool is a synchronous allocator with per-block deferred reuse, while the async
pool overlaps its work with the host. poolkit's case is not "fastest wall clock";
it is a synchronous API, offline analysis of the same workload, and measured
fragmentation.

---

## 4. What could not be measured

* **A real application's trace.** CUPTI is in-process, so recording an
  unmodified binary needs the injection library that M6b2 will have to build
  ([`record.md`](record.md) §1). Every number in §1 is from
  `tests/device/record_target`, a workload written for the purpose.
* **A second GPU, or two processes at once.** Out of scope for v1.0 and, per the
  plan's rule 3, a way to make all of the above wrong.
* **Whether the 1.2 ms of M4 came from heap expansion.** The M6b tool does not
  record the moment inside the driver; the claim is "the median is 61 µs, the max
  is 244 µs", not a mechanism.

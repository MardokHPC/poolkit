# The benchmark: the same trace, three allocators, one card

`poolkit_bench` answers the only question that can make somebody adopt a
custom allocator: **on this trace, on this GPU, how long does the allocation
traffic take, and how much memory does it hold?** It replays one trace through
poolkit, through `cudaMalloc`/`cudaFree`, and through
`cudaMallocAsync`/`cudaFreeAsync`, and reports wall time, peak and per-operation
latency for each — with the GPU, driver, runtime, arch and pool config printed
next to the numbers.

```bash
poolkit_bench --trace traces/inference_transformer.jsonl \
              [--repeat 5] [--warmup 1] \
              [--mode pool|cudamalloc|cudaasync|all] [--json bench.json]
```

Exit codes: `0` success, `2` bad arguments / unreadable trace / CUDA error, `77`
no device.

---

## 1. The five rules, and where they live in the code

`DEVELOPMENT_PLAN.md` §Metodologia di misura says the benchmark does not get
published until these five rules are respected in `tools/`. Each one is a
mechanism in [`tools/bench.cu`](../tools/bench.cu), not a paragraph in a doc:

| rule (plan) | implementation |
|---|---|
| **Fixed footprint between the configurations** | every run builds a fresh pool and destroys it; every driver block is freed and `cudaDeviceSynchronize` is called between modes, so each mode starts from the same resident bytes. What cannot be equalised is stated in §5 |
| **1 warmup + 5 runs, median + min/max** | `Bucket::add` collects one distribution per run; `Bucket::finish` reports the median of the per-run medians, the median of the per-run p99s, the smallest min and the largest max. `--warmup 0` is possible, and it is a mistake: one cold pool replay measures 103.8 ms where the warm median is 4.8 ms (the driver's one-time heap reservation), see [`measurements-m6.md`](measurements-m6.md) §2 |
| **One process at a time** | the tool measures one mode at a time, frees everything before the next; the method line in the header says so. Running a build or a profiler next to it invalidates the numbers, and that is not detectable from inside |
| **Config declared next to the number** | header prints GPU, compute capability, driver, runtime, arch, chunk, growth, min/max class, class count; the JSON carries the same in `bench.env` |
| **Never the mean alone** | every number is a median plus p99, min and max; the table never shows an average |

## 2. What each column means

```text
| mode | wall median | wall min | wall max | peak | peak source | alloc p50 | alloc p99 | free p50 | free p99 |
```

* **wall** — `steady_clock` around the whole trace replay (all 3076 operations,
  in order), median of the 5 timed runs plus the smallest min and largest max of
  those runs. This is the number to compare across modes: it includes everything
  the allocator does for this workload.
* **peak** — bytes the mode held from the driver at its high-water mark. The
  source column says whether it is exact or sampled (§3).
* **alloc p50/p99** — per *allocation call*, not per operation pair. A median is
  meaningless when the distribution is bimodal, so the pool section below splits
  it; for the driver modes the p99 is the useful column (a chunk expansion lives
  in the tail).

The pool section that follows adds the split that makes the pool's median
interpretable:

| path | what it is |
|---|---|
| **hit** | `free_ready` non-empty: a recycled block, no driver call |
| **growth, total** | everything else (the plan's second path); decomposed below |
| new slot, existing chunk | a never-used block of a slab already resident: bookkeeping |
| fresh chunk (`cudaMalloc` + carve) | the chunk was empty: one `cudaMalloc` **plus** carving ~900 block records |
| large path | request above `max_class_bytes`: one `cudaMalloc`, never reused |

The counts next to each row are the denominator: "400 ns" means something
different at 1496 of 1538 operations than at 8 of 1538.

## 3. Why the peak is sampled for the driver modes, and what it costs

The pool knows its own footprint exactly (`pool.capacity()` plus the live
large-path bytes), so its peak is a counter, sampled after every operation at no
cost beyond a member read.

`cudaMalloc` and `cudaMallocAsync` have no counter to read: the only way to see
their footprint is `cudaMemGetInfo`. Calling it after **every** operation would
put a driver call between two measured operations, which inflates both the wall
time and every latency sample — the measurement would perturb what it measures.
So the driver modes sample every **64 operations** (`kSampleEveryOps`), and the
peak they report is a **lower bound**: the real peak can be higher by at most the
largest allocation that lives and dies entirely between two samples. On the
committed trace that bound is one activation (a few KiB); the interesting peak —
the weights resident across hundreds of operations — is always observed.

The exact requested-bytes peak of the committed trace is 37.92 MiB (computed
from the trace alone). `cudaMalloc` reports 46 MiB of *device* memory: the
difference is the driver's 2 MiB granularity on the large blocks, which the
requested-bytes sum cannot see. That is a feature of the sampled measurement, not
noise.

`cudaMallocAsync`'s peak is also a lower bound for a second reason: the driver's
pool releases memory lazily, so a drop in `cudaMemGetInfo` can lag the frees.

## 4. `cudaMallocAsync` latency is the cost of *enqueueing*

`cudaMallocAsync`/`cudaFreeAsync` return once the operation is queued on the
stream. The latency columns for that mode therefore measure the cost of asking,
not of obtaining memory: compare them with `cudaMalloc`'s only if you know that.
The wall time is comparable, because it includes the final
`cudaStreamSynchronize`, which waits for all the queued work — and that is why
`cudaasync` can show the smallest wall time while doing exactly the same
allocations: the work overlaps instead of serialising with the host.

## 5. What "fixed footprint" cannot mean here

The plan's first rule exists because the base address and the resident bytes
change performance: two runs with different memory held are not comparable. The
bench resets the world between runs, but the *pool's own design* holds more
memory than `cudaMalloc` does at the peak (68 MiB vs 46 MiB on the committed
trace, because the pool retains one chunk per class). Forcing equal footprints
would mean changing the pool's retention policy, i.e. measuring a different
allocator. The honest statement is the one the table already makes: the `peak`
column **is** the footprint, and the wall times are read next to it. `poolkit`
trades bytes for time; hiding the bytes would make the time meaningless.

The large-path row is the same trade-off in the other direction: a table with a
smaller `max_class_bytes` can look cheaper in bytes while moving allocations to
the large path, where every request is a `cudaMalloc` and nothing is reused. The
split exposes it as a count.

## 6. The JSON document

`--json out.json` writes **schema 2**: the schema-1 analysis document
(trace identity, config, classes, waste, peak) plus a required `bench` section
with `schema: 1` of its own. Compatibility, stated the way the reader implements
it:

* a schema-1 document (what `poolkit_advise` writes) has no `bench` key and is
  read exactly as before — `tests/test_tools.cpp` pins the byte-identical round
  trip;
* a schema-2 document **must** carry `bench`; a schema-2 file without one is
  rejected, so a truncated bench report cannot be mistaken for an analysis;
* the reader refuses a bench section in a schema-1 document, a distribution that
  is not ordered (`min <= median <= p99 <= max`), a "sampled" peak with a zero
  sampling period, and a pool split whose counts do not add up to the mode's
  allocation count;
* `poolkit_gate` accepts both schemas but refuses to compare them with each
  other, and refuses any document whose `tool` is not `poolkit_advise`: a bench
  run is not a regression report.

The analysis fields in a bench document are real, not placeholders: `waste` is
the same histogram integral `poolkit_advise` computes, on this trace and this
table; `peak.resident_bytes` is the exact pool peak when the pool mode ran,
otherwise the sampled peak of the first mode (the `bench.modes[*].peak_sampled`
field says which). `peak.blocks` stays `0`: the bench tracks bytes, not
simultaneous handles.

## 7. What it does not measure

* **Kernel time.** This is allocation traffic. A trace says nothing about the
  kernels between allocations, so the wall time is the allocator's contribution,
  not the application's runtime.
* **Multi-stream behaviour.** The traces carry one opaque stream id and the
  replays use the default stream; stream-ordered reuse across streams is out of
  scope for v1.0.
* **Another process on the GPU.** Any other tenant changes `cudaMemGetInfo` and
  the driver's behaviour; the numbers are only comparable on a quiet card.
* **A mode that fails.** If a mode cannot run (out of device memory, no async
  pool), it is named on stdout and left out of the table and the JSON — never
  written as a zero.

## 8. A run on the committed trace

Real output of one invocation (`--repeat 5 --warmup 1`), RTX 2060 SUPER, sm_75,
driver API 13.3 (display driver 610.88), CUDA runtime 12.6, one of the three consecutive invocations
tabulated in [`measurements-m6.md`](measurements-m6.md) (the third one, verbatim):

```text
poolkit_bench
  GPU            : NVIDIA GeForce RTX 2060 SUPER (sm_75)
  driver         : 13030   runtime: 12010   arch: x86_64
  trace          : ..\traces\inference_transformer.jsonl (3076 ops)
  pool config    : chunk 2.00 MiB, growth 1.50, min class 256 B, max class 2.00 MiB, 34 classes
  method         : 1 warmup + 5 runs, steady_clock, median of run medians, min/max
  peak sampling  : every 64 ops for the driver modes; pool peak exact

| mode | wall median | wall min | wall max | peak | peak source | alloc p50 | alloc p99 | free p50 | free p99 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| pool | 4.955 ms | 4.817 ms | 5.111 ms | 68.00 MiB | exact (pool) | 400 ns | 62.7 us | 600 ns | 800 ns |
| cudamalloc | 9.998 ms | 9.285 ms | 10.672 ms | 46.00 MiB | sampled | 2.4 us | 58.2 us | 1.8 us | 81.0 us |
| cudaasync | 2.684 ms | 2.348 ms | 2.821 ms | 64.00 MiB | sampled | 400 ns | 800 ns | 400 ns | 600 ns |

pool allocation paths (same runs, 1538 allocs):
| path | ops | median | p99 | min | max |
|---|---:|---:|---:|---:|---:|
| hit (recycled block) | 1496 (97.3%) | 400 ns | 500 ns | 300 ns | 33.8 us |
| growth, total | 42 (2.7%) | 57.3 us | 123.3 us | 200 ns | 244.3 us |
|   new slot, existing chunk | 8 (0.5%) | 300 ns | 300 ns | 200 ns | 700 ns |
|   fresh chunk (cudaMalloc + carve) | 34 (2.2%) | 62.4 us | 123.3 us | 47.9 us | 244.3 us |
|   large path (cudaMalloc, no reuse) | 0 (0.0%) | 0 ns | 0 ns | 0 ns | 0 ns |
| free (event record) | 1538 (100.0%) | 600 ns | 800 ns | 600 ns | 8.8 us |
```

That is invocation 1 of the three committed in [`docs/bench/final-run1.json`](bench/final-run1.json),
[`…run2`](bench/final-run2.json), [`…run3`](bench/final-run3.json); the table in the README is the median across the
three. An earlier session, same method, produced 4.18 / 10.52 / 2.64 ms — the difference is run-to-run spread, and the
committed JSONs are the authoritative artifacts.

Read it as: 97.3 % of this workload's allocations are served by a recycled block
at ~400 ns; the other 2.7 % are the 34 chunk expansions the trace's peak forces,
at ~62 µs each. The driver modes have no such split, so their median is the
median over a bimodal mix of small and large `cudaMalloc`s.

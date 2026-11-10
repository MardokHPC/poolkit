# M4 measurements

Measured numbers for the three experiments the plan still lacked
(DEVELOPMENT_PLAN.md §7), plus the trace-replay internal-waste comparison. Every
number below was produced by `tools/measure_m4.cu` (device build) or
`tests/test_trace.cpp` (host build), on the hardware and toolchain stated next to
it. Nothing is estimated; where a measurement could not be taken, that is said
explicitly.

## Environment

| field | value |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 SUPER, 8192 MiB |
| compute capability | sm_75 |
| driver (nvidia-smi) | 610.88 |
| driver (cudaDriverGetVersion) | 13030 (= CUDA 13.3) |
| runtime / nvcc | CUDA 12.6.20 (`v12.6`) |
| host compiler | MSVC 19.44 (VS 2022 BuildTools) |
| free memory at measurement time | 7158 MiB |

Methodology for every timed number: **1 warmup run + 5 timed runs**, each run of
`N` samples, `std::chrono::steady_clock` around each sample, reported as the
median of the per-run medians, the median of the per-run p99s, the smallest min
and the largest max (plan §Metodologia di misura: "1 warmup + 5 run, mediana +
min/max"). One process at a time; the GPU was otherwise idle.

Caveat that applies to every sub-microsecond number: `steady_clock` on this
machine has ~100 ns resolution, so `min = 0 ns` and a `median = 100 ns` both
mean "too fast to resolve", not "free". The p99/max still separate the two
query states, which is what matters.

---

## 1. Real `cudaMalloc` granularity

Method (per the corrected plan note — a single 1 MiB allocation shows a 0-byte
delta because the driver had already reserved the memory, so the granularity is
only visible across a cumulative, growing sequence):

- **1a.** one run, allocating 1, 2, 4, 8, 16, 32, 64, 128, 256 MiB in sequence,
  recording the `cudaMemGetInfo` free-memory delta after each step;
- **1b.** 1000 allocations of 4 MiB, recording the first and the steady per-block
  delta and the cumulative consumption.

### 1a. Increasing blocks

| requested | free delta |
|---|---:|
| 1 MiB | **2 MiB** |
| 2 MiB | 2 MiB |
| 4 MiB | 4 MiB |
| 8 MiB | 8 MiB |
| 16 MiB | 16 MiB |
| 32 MiB | 32 MiB |
| 64 MiB | 64 MiB |
| 128 MiB | 128 MiB |
| 256 MiB | 256 MiB |

### 1b. 1000 × 4 MiB

| quantity | value |
|---|---:|
| first allocation delta | 4 MiB |
| steady per-block delta | 4 MiB |
| cumulative consumption (1000 blocks) | 4000 MiB |

### Judgement

The **granularity is 2 MiB**: a 1 MiB request consumes 2 MiB, and everything at
or above 2 MiB is consumed exactly. `chunk_bytes = 2 MiB` — the v1.0 default —
**is exactly the observed granularity**, so the default needs no change, and the
risk "granularity ≠ 2 MiB" (plan §Registro dei rischi) did not materialise. The
slab-carving decision (§4.2 of `docs/cost-model.md`, whether to round chunks up
to whole blocks) therefore stands as shipped: a 2 MiB chunk is the driver's own
unit, and rounding it up to `c·ceil(C/c)` would *buy* extra driver memory at the
same 2 MiB granularity, not free tail.

---

## 2. `cudaEventQuery` cost

100 000 queries on an already-completed event and 100 000 on a pending event
(kept pending by a `clock64()` spin kernel — see the note below), `steady_clock`
around each query, 1 warmup + 5 runs.

| state | median | p99 | min | max |
|---|---:|---:|---:|---:|
| completed event | 100 ns | 100 ns | 0 ns | 24.2 µs |
| pending event | 100 ns | 200 ns | 0 ns | 23.8 µs |

The p99 gap (100 → 200 ns) is the only real signal: querying a pending event is
about twice the cost of a completed one, but both are at the clock-resolution
floor.

### Judgement

`cudaEventQuery` is ~100–200 ns, i.e. negligible against the pool's own free cost
(~600 ns) and a `cudaMalloc` (~1.2 ms, §3 — that figure is the *maximum* of the
growth path, not a typical allocation: see the note at the end of §3). The plan's
criterion — "switch to epoch reuse if the per-free query exceeds ~5% of the
allocate cost" — is **not** triggered: the query is ~10% of an allocate+free
*pair* and a rounding error next to a chunk allocation. Per-block deferred reuse
is justified; the epoch fallback (one event per N frees) stays a documented
alternative, not a requirement.

---

## 3. Pool `allocate`/`free` cost: hit vs growth

Same methodology (1 warmup + 5 runs, 10 000 ops per run).

| path | median | p99 | min | max |
|---|---:|---:|---:|---:|
| hit: `allocate`+`free` (one recycled block) | 1000 ns | 1100 ns | 900 ns | 13.5 µs |
| hit: `free` alone (event record) | 600 ns | 800 ns | 500 ns | 44.1 µs |
| growth: `allocate` only (10 000 blocks, never freed) | 100 ns | 400 ns | 100 ns | **1.2 ms** |

The growth median (100 ns) is the slab-carving path — taking a never-used slot.
The growth max (1.2 ms) is the `cudaMalloc` of a fresh 2 MiB chunk, which
happens once every ~2048 blocks of the 1024 B class and dominates the tail. (The
tool allocates 1024 B, and the shipped table contains the power of two, so
`index_for(1024) = 1024`; 2 MiB / 1024 B = 2048 blocks per chunk. An earlier
version of this paragraph said the 1280 B class, which is Table B's *geometric*
class for that request, not the class the shipped union table — and therefore
this tool — uses.)

### Judgement

The pool's steady-state hit path (reuse + event record) is **~1 µs**, against
**~1.2 ms** for the growth path that calls `cudaMalloc` — three orders of
magnitude. That gap is exactly what the pool exists to amortise, and it is the
number the M6 benchmark (`README` table) should headline: an allocation served
from a recycled block costs ~1000× less than a fresh driver allocation. The
`free` cost is dominated by `cudaEventCreateWithFlags`/`cudaEventRecord`, not by
the query (§2).

> **Retraction (M6b, `docs/measurements-m6.md` §3).** The 1.2 ms above is the
> **maximum** of 10 000 growth-path samples, not the cost of a `cudaMalloc`: it is
> the tail where the driver expands its heap, and it includes the process's first
> chunk. Measured again with a distribution, the median is **~2.3 µs** for a
> `cudaMalloc` on the trace's own sizes and **~61 µs** for a fresh 2 MiB chunk, so
> the honest ratios are **~6×** and **~152×** on the median — not ~1000×, which is
> the ratio against the single worst event recorded here. The numbers in this
> document are left as they were measured on the date in the header; the retracted
> headline is the one in `README.md`, and it now carries the medians.

---

## 4. Trace replay: internal waste vs the model

`traces/inference_transformer.jsonl` (1538 allocs, 1538 frees; weights
~1 MiB and long-lived, activations ~2–8 KiB and short-lived; seed 42) replayed
twice on `FakeBackend` — counters bit-identical, as asserted by
`test_trace.cpp` — and the size histogram integrated through
`SizeClassTable::expected_internal_waste()`.

| quantity | value |
|---|---:|
| internal waste on the trace histogram (capacity-normalised) | **23.915 %** |
| model prediction for log-uniform requests (integrated) | 18.028 % |
| deviation (absolute / relative) | +5.89 pp / **+32.6 %** |

The deviation is **outside** the plan's ±20 % relative criterion. That is not a
surprise and not a bug: the trace is deliberately *not* log-uniform, and the
weights (~1 MiB = 1 048 576 B) sit just above class 19 (920 576 B) and land in
class 20 (1 380 864 B), wasting ~32 % per weight. The capacity-normalised mean is
dominated by those large weights, so it reads high. The point of the trace is
precisely to surface this: a real bimodal workload on `g = 1.5` wastes noticeably
more than the log-uniform placeholder predicts, and the plan's own rule says that
is when `g` (or the class set) gets revisited — see the open items in the M4
report.

Two normalisations, stated rather than conflated: the trace number above is
**capacity-normalised** (`sum(c−s)/sum(c)`), matching the model's 18.028 %.
`PoolSnapshot::internal_fragmentation` is **request-normalised**
(`sum(c−s)/sum(s)`, ≈ 21.64 % for log-uniform). The snapshot reports 0 % at the
end of a full replay because the trace frees its weights before the end, so the
live set is empty — `internal_fragmentation` measures the *live* set at a moment,
while the histogram integral measures the *workload*. The comparison has to use
the histogram (what the cost model predicts), not the end-of-trace snapshot.

**Follow-up (M5).** The answer the rule asks for is §9 of
[`cost-model.md`](cost-model.md#9-why-powers-of-two-m5): the class set gained the
powers of two, not `g`. Measured with the same method
(`tools/measure_classes.cpp`, also a ctest entry), the same trace now reads
**15.234 %** at `k = 34` — −15.5 % relative to the model, i.e. inside ±20 % —
with expected tail 0.962 % → 0.543 % and worst-case retention 46 → 68 MiB.

---

## Implementation notes (things the measurements forced)

- **`__nanosleep` is a no-op here.** On this driver (610.88, sm_75),
  `__nanosleep(1000 ms)` returned in ~0.9 ms. The spin kernels in
  `tests/device_kernels.cu` and `tools/measure_m4.cu` busy-wait on `clock64()`
  (~1.46 GHz on this card) instead, which keeps a stream reliably busy.
- **`min = 0 ns`** in §2/§3 is the `steady_clock` resolution floor, not a real
  zero.
- The device build (VS 2022 BuildTools, MSVC 19.44 + CUDA 12.6, sm_75) and the
  host build (VS 18 Community, MSVC 19.51) both compile clean under `/W4 /WX`.

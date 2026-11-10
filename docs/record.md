# The recorder: a trace from a real program

`poolkit_record` turns a CUDA program's actual `cudaMalloc`/`cudaFree` traffic into
the JSONL that [`trace-format.md`](trace-format.md) defines, so the rest of the
toolchain (`poolkit_advise`, `poolkit_gate`, `poolkit_bench`) can work on a real
workload instead of a generator. It is a thin, opinionated wrapper around the
CUPTI activity API that the [M5 spike](spike-cupti.md) proved works on this stack
(Windows + CUDA 12.6 + sm_75, no privileges).

```bash
poolkit_record -o trace.jsonl [--min-size 4096] [--max-records 2000000]
               [--drop-internal] --workload <lib.dll|lib.so> [-- args...]
```

| flag | default | meaning |
|---|---|---|
| `-o`, `--out <f>` | required | trace file to write |
| `--min-size <bytes>` | `4096` | drop allocations smaller than this (and their frees, coherently) |
| `--max-records <n>` | `2000000` | stop recording after n records; the file is still valid |
| `--drop-internal` | off | **not implemented**, see §5 |
| `--workload <lib>` | — | shared library exporting `poolkit_record_workload` |
| `-- [args...]` | — | arguments for that entry point; without `--workload`, a command to spawn |
| `-h`, `--help` | | usage |

Exit codes: `0` trace written; `2` bad arguments / CUPTI unavailable / library
that cannot be loaded; `4` the workload ran but produced no records; `77` no CUDA
device (so the test suite skips instead of failing); **otherwise the workload's
own exit code**, with the trace flushed and written before returning it.

---

## 1. Why the workload is a library, not a child process

Which route works, in one table — the rest of this section and §9 explain why:

| what you want to record | route | Windows | Linux |
|---|---|---|---|
| a workload you can build as a shared library | `--workload <lib.dll\|lib.so>` | **works**, and `ctest` covers it (`poolkit_record_target_record`) | the same source builds; no Linux driver run exists here |
| a program you did not write, and cannot rebuild | injection through `CUDA_INJECTION64_PATH` (§9) | **works**, and `ctest` covers it (`poolkit_inject_record_target`) | the `.so` compiles from the same source, **never run** on a Linux driver (§9.3) |
| an arbitrary command, wrapped | `poolkit_record -- <program>` | **records nothing**: 0 records, exit `4` | same |

There is no route that records "any binary, anywhere". CUPTI's activity API
instruments **the process that calls `cuptiSubscribe`**: it does not see another
process's CUDA calls. That is not a poolkit restriction, it
is why NVIDIA's own tools inject a library (and on Windows a Detours shim) into
the target. The original plan for this milestone said "spawn the child with
`CreateProcess`/`_spawnvp` and record it", which cannot work with this API. The
recorder still accepts that form and still spawns the child — so the answer to
"why doesn't this wrap any binary" is a measurement, not an opinion:

```text
$ poolkit_record -o child.jsonl -- poolkit_record_target.exe
record_target: 1 cycles, 4 allocations x 4194304 + 65536 + 1048576 B tracked
poolkit_record: 0 records: CUPTI instruments this process only, and the spawned
child's CUDA calls are invisible here. Build the workload as a shared library and
pass it with --workload (docs/record.md section 1), or record the program
unmodified with poolkit_inject.
```

Exit code `4`, no file. The workloads that *can* be recorded are shared libraries
exporting one C symbol:

```cpp
extern "C" int poolkit_record_workload(int argc, char** argv);
// return 0 on success; anything else becomes poolkit_record's exit code.
```

The recorder loads it with `LoadLibrary`/`dlopen` **after** CUPTI is subscribed,
calls it, synchronises the device, flushes the activity buffers, and unloads it.
`tests/device/record_target.cu` is built twice from one source for exactly this
reason: as an executable (a normal CUDA program) and as a library (the recorder's
workload).

Recording an **unmodified** application is what `poolkit_inject` is for (§9): on
Windows, a DLL that CUDA loads into the target when `CUDA_INJECTION64_PATH` is set,
exporting `InitializeInjection`. It reuses this same recording core, so a trace
produced by injection and one produced by `poolkit_record` have the same dialect.
It is verified on Windows only — §9.3 states what is and is not known about Linux.

## 2. What is recorded, field by field

One `CUPTI_ACTIVITY_KIND_MEMORY2` record becomes one line:

| trace field | source | note |
|---|---|---|
| `op` | `memoryOperationType` | `ALLOCATION` → `"alloc"`, `RELEASE` → `"free"`; any other operation type has no representation and is skipped |
| `size` | `bytes` | the size the application asked for, as CUPTI reports it; `0` on a free, as the schema requires |
| `ptr` | `address` | **the raw 64-bit driver address written as a decimal integer**. The replay treats it as an opaque identity, so precision is irrelevant to it; the real address is kept because it can be matched against `compute-sanitizer`/`cuda-gdb` output |
| `stream` | — | **always `0`**: `CUpti_ActivityMemory3` does not carry the stream that caused the operation. Writing a guessed stream would be worse than writing the truth, and the host replay treats the field as opaque |
| `ordinal` | arrival order | assigned when the record is accepted, strictly increasing, which is what the parser enforces |

Verified on the two allocators the plan asks about:

* `cudaMalloc`/`cudaFree` → 3 allocs + 3 frees for `record_target`;
* `cudaMallocAsync`/`cudaFreeAsync` → the same 8 MEMORY2 records, same mapping.
  The driver's pool allocator reports one `ALLOCATION`/`RELEASE` pair per
  application call, **no special mapping was needed**, and the sizes are the
  requested ones (the pool's internal suballocation does not appear).

Timestamps are *not* written: the schema has no field for them, and inventing one
would break every reader. `timestamp` and `correlationId` are in the CUPTI record
if a future schema version wants them.

## 3. Filters, and why the trace stays valid

The parser refuses an orphan free and a duplicate live `ptr`
([`trace-format.md` §1](trace-format.md#1-one-json-object-per-line)), so every
filter in the recorder is applied to *both* sides of an allocation:

* **`--min-size`** (default 4096 B) drops small allocations — an inference stack
  does thousands of tiny `cudaMalloc`s for workspace and they would dominate the
  file without describing the workload. The dropped address is remembered, and
  its `free` is dropped too. `record_target` exercises exactly this path: three
  tracked sizes are kept, one 1024 B allocation is filtered with its free.
* **frees of blocks allocated before recording started** (runtime internals,
  or something the process did before `cuptiSubscribe`) have no matching alloc;
  they are counted as `orphan frees` and skipped, never written.
* **`--max-records`** truncates the *end* of the trace. A truncated trace is a
  workload that simply never frees its last blocks, which the format allows (live
  blocks at the end are legal), so the file remains parseable and replayable.
  The tool prints how many records it refused.
* **`--drop-internal` is not implemented** and says so on stderr. The heuristic
  a recorder can apply — "never freed" or "a size the runtime likes" — also
  deletes a workload's long-lived weights and its grow-only caches, i.e. exactly
  the allocations whose retention the tool is supposed to measure. A filter that
  silently removes measurements is worse than no filter. If runtime noise shows
  up, use `--min-size` and read the `filtered` line.

A useful sanity check after a recording: the `filtered:` line. `orphan frees` > 0
means the process allocated before the subscription (usually the CUDA context);
`duplicate allocs` > 0 would mean CUPTI delivered the same address while it was
still live, and the second one is dropped because the replay cannot represent it.

## 4. What it cannot record

| not recorded | why | consequence |
|---|---|---|
| another process's CUDA calls | CUPTI is in-process (§1) | the workload must be a library, or the program recorded with injection (§9) — which is built and verified on **Windows only** (§9.3); `poolkit_record -- <command>` is not a workaround, it records 0 records and exits `4` |
| allocations made before `cuptiSubscribe` | CUPTI only reports what happens while enabled | the context's own memory shows up as orphan frees at best; that is the `filtered` count |
| the stream of an operation | not in `CUpti_ActivityMemory3` | `stream` is `0`; a replay cannot reproduce stream-ordered reuse from this trace |
| timestamps, correlation ids | not in the trace schema | no ordering beyond the arrival order; no kernel correlation |
| device/context id | not in the trace schema | a multi-GPU process is recorded as one flat sequence |
| allocator internals of `cudaMallocAsync` | MEMORY2 reports the application call, not the pool's suballocations | good for the trace, but it means the trace cannot tell you how much the driver's pool kept |

## 5. Overhead

Measured on **Windows** (RTX 2060 SUPER, sm_75, driver API 13.3 / display driver 610.88, CUDA 12.6) against
`tests/device/record_target`, same binary with and without the recorder, median of
7–10 runs each (`docs/measurements-m6.md` §1 has the table):

* **fixed cost ≈ 15–17 ms per process**: CUPTI init, the flush, and writing the
  file. This is also the cost of recording a 4-allocation program, which is why
  the absolute numbers on tiny workloads look dominated by it.
* **marginal cost ≈ 2.6–4.8 µs per recorded operation** — the range is two
  measurement sessions on the same machine, 3000 written lines apart — from 4000
  memory operations (3000 of them written) costing 24–30 ms more than the same
  workload unrecorded.

That is the number that decides whether recording a real inference run is
affordable, and it was not measured by the M5 spike. For a 60 s run with a few
thousand allocations it is noise; for a microbenchmark it is not.

Both figures are Windows figures, measured once: no recorder overhead has been
timed on a Linux driver here (§9.3).

## 6. Determinism

The *file* is not byte-identical between two runs, and cannot be: addresses come
from the driver and change with the process's allocation history. What is stable
for the same program on the same input is

* the **number of operations** and their **order** (given the same code path),
* hence the size histogram, and therefore everything `poolkit_advise` computes
  from it.

Two things can still change the operation sequence legitimately: the first run of
a process initialises the context lazily (extra internal allocations, usually
below `--min-size` or unfreed), and a program whose control flow depends on a
clock will allocate differently. Record twice and diff the sizes when in doubt:

```bash
poolkit_record -o a.jsonl --workload ./libworkload.so
poolkit_record -o b.jsonl --workload ./libworkload.so
diff <(cut -d, -f2 a.jsonl) <(cut -d, -f2 b.jsonl)   # sizes only
```

## 7. A real recording

```text
$ poolkit_record -o record_target.jsonl \
      --workload build-dev/poolkit_record_target_workload.dll
record_target: 1 cycles, 4 allocations x 4194304 + 65536 + 1048576 B tracked
poolkit_record: NVIDIA GeForce RTX 2060 SUPER, sm_75, in-process workload
  MEMORY2 records seen 8, kept 6 (3 allocs, 3 frees)
  filtered: 1 allocs below 4096 B, 0 frees without an alloc, 0 duplicate allocs
  elapsed 135.3 ms
  wrote record_target.jsonl (403 B)

$ cat record_target.jsonl
{"op":"alloc","size":4194304,"ptr":30165434368,"stream":0,"ordinal":0}
{"op":"alloc","size":65536,"ptr":30169628672,"stream":0,"ordinal":1}
{"op":"alloc","size":1048576,"ptr":30169694208,"stream":0,"ordinal":2}
{"op":"free","size":0,"ptr":30165434368,"stream":0,"ordinal":3}
{"op":"free","size":0,"ptr":30169628672,"stream":0,"ordinal":4}
{"op":"free","size":0,"ptr":30169694208,"stream":0,"ordinal":5}
```

`ctest` runs this end to end: `poolkit_record_target_record` produces the trace,
`poolkit_record_trace_replays` validates it with the real parser and the real
replay (`tests/device/record_check.cpp`): three allocations of the expected sizes,
the 1024 B one absent, every free paired, replay peak > 0. Both skip with exit 77
on a machine with no device.

## 8. Troubleshooting

| symptom | cause | what to do |
|---|---|---|
| `CUPTI error in cuptiSubscribe` | the CUPTI runtime library is not next to the executable and not on `PATH`/`LD_LIBRARY_PATH` | on Windows the build copies `cupti64_*.dll` next to `poolkit_record`; elsewhere add `<CUDA>/extras/CUPTI/lib64` |
| exit 77 | no CUDA device | expected on CI without a GPU |
| exit 4, 0 records | child command (§1), or the workload never allocated above `--min-size` | use `--workload`, or lower `--min-size` |
| `does not export 'poolkit_record_workload'` | the library was built without `__declspec(dllexport)` / `-fvisibility=default` | see `tests/device/record_target.cu`'s export macro |
| `WARNING: CUPTI dropped N of its own records` | activity arrived faster than the 8 MiB buffers could be consumed | the trace is incomplete; record a shorter run |
| trace parses but the peak looks impossible | the workload allocates before the recorder subscribes, or the run was truncated | compare `allocs` with the live-block count, and check the `filtered:` line |

---

## 9. `poolkit_inject`: recording a program you did not write

`poolkit_record` needs a workload library it can load itself. That covers code you
compile; it does not cover `python train.py`, a game, or anything else already built.
For those there is `poolkit_inject` — **on Windows, and only on Windows in this
release** (§9.3): a DLL that CUDA loads **into the target process** at driver
initialisation:

```bat
set CUDA_INJECTION64_PATH=<build>\poolkit_inject.dll
set POOLKIT_RECORD_OUT=trace.jsonl
<run the program as it is>
```

`POOLKIT_RECORD_MIN_SIZE` (default 4096), `POOLKIT_RECORD_MAX_RECORDS` (default
2000000), `POOLKIT_RECORD_BUFFER_KB` (default 32) and `POOLKIT_RECORD_FLUSH_MS`
(default 100) are optional. `POOLKIT_RECORD_DEBUG=1` traces the injection path to
stderr, one line per step.

With no `POOLKIT_RECORD_OUT` the DLL is **inert**: it returns from
`InitializeInjection` without touching CUPTI, so exporting `CUDA_INJECTION64_PATH`
and forgetting the output path costs nothing (measured: the target runs and exits
with its own code). The target's exit code is preserved either way.

### 9.1 What CUDA provides, and what it costs

When `CUDA_INJECTION64_PATH` names a library, the driver loads it during its own
initialisation and calls the exported `InitializeInjection`; this is documented by
CUPTI (`extras/CUPTI/samples/cupti_trace_injection/README.txt`) and it is the
mechanism NVIDIA's own tools use. From inside that call the recorder does exactly
what `poolkit_record` does: `cuptiSubscribe`, `cuptiActivityRegisterCallbacks`,
`cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2)`. The export is
`extern "C" __declspec(dllexport) int InitializeInjection(void*)`; CUDA 12.6's own
sample declares it `int InitializeInjection(void)` and returns 1, and on the x64
ABIs poolkit targets the two forms agree.

The part that had to be *measured* is the exit. The CUPTI sample flushes from an
`atexit` handler and warns in a comment that this is unsafe on Windows — it hooks
`RtlExitUserProcess` with Detours for exactly this reason. poolkit takes no
Detours dependency, so the first implementation tried the obvious thing and it
failed in a specific way:

```text
poolkit_inject: [debug] finish_recording: entered
poolkit_inject: [debug] stop: cuptiActivityFlushAll(1)      <- never returns
```

Both `cuptiActivityFlushAll(1)` and `cuptiActivityFlushAll(0)` block forever when
called from that handler on this stack (Windows, CUDA 12.6, driver API 13.3 / display driver 610.88), and the
resulting process is not merely hung: `taskkill /F` reports "no running instance"
while the process stays in `tasklist` — a thread stuck in the driver cannot be
terminated. A recorder that wedges the program it was asked to observe is worse
than one that loses its last buffers, so:

* `finish_at_exit()` makes **no CUPTI call at all**. It flushes and closes the
  trace file and prints the counters of what CUPTI already delivered;
* the trace materialises because CUPTI delivers a buffer when it **fills**, and
  the shim asks for small buffers (32 KiB ≈ 570 records) instead of 8 MiB. The
  independent `cuptiActivityFlushPeriod` path was tried first and delivered
  nothing at all with 8 MiB buffers on this stack, which is why it is only a
  second chance and not the mechanism.

Consequences, stated plainly rather than discovered by a user:

* the **last partial buffer is lost**: up to ~570 memory operations, and with the
  default `--min-size` filter the file simply ends with live blocks, which the
  format allows;
* a program that performs fewer than ~570 `cudaMalloc`/`cudaFree` operations and
  exits may produce an empty trace. Lower `POOLKIT_RECORD_BUFFER_KB` (8 KiB works,
  at the price of more records dropped under a high allocation rate) or record a
  longer run;
* a process that calls `ExitProcess`/`TerminateProcess` directly, or is killed,
  runs no `atexit` handler: what CUPTI already flushed is on disk (the writer is
  incremental, one line per record), nothing else happens;
* only MEMORY2 is enabled — no kernels, no memcpys, no streams, same as §2.

### 9.2 A real injected recording (Windows)

`tests/device/record_target.cu` is the workload: it is a normal CUDA executable,
and nothing in it knows that this run is recorded.

```text
$ set CUDA_INJECTION64_PATH=<build>\poolkit_inject.dll
$ set POOLKIT_RECORD_OUT=inject_target.jsonl
$ poolkit_record_target.exe 400
   stderr: poolkit_inject: recording MEMORY2 into 'inject_target.jsonl'
                           (min-size 4096 B, max-records 2000000, flush every 100 ms)
   stdout: record_target: 400 cycles, 4 allocations x 4194304 + 65536 + 1048576 B tracked
   stdout:   MEMORY2 records seen 2816, kept 2112 (1056 allocs, 1056 frees)
             filtered: 352 allocs below 4096 B, 0 frees without an alloc, 0 duplicate allocs
             elapsed 343.8 ms
   stderr: poolkit_inject: recording stopped
   stderr: poolkit_inject: wrote inject_target.jsonl (147082 B)

$ poolkit_record_check inject_target.jsonl --injected
record_check: OK 2112 ops, 1056 allocs (4194304 B, 65536 B, 1048576 B), 1056 frees,
              peak 8388608 B, 3 blocks
```

`ctest` runs exactly this (`poolkit_inject_record_target`, then
`poolkit_inject_trace_replays`), so the claim above is a test and not a transcript
that could rot. The second check is the relaxed one: an injected recorder
subscribes *before* the runtime creates its context, so the driver's own startup
allocations may legitimately be in the file — the check requires that the file
parses, replays, contains the workload's three sizes and no allocation below the
min-size, instead of demanding an exact operation count that would be a test of
the driver.

### 9.3 Linux: not verified

**Status: the source is platform-neutral, the binary has never been run on Linux.**
The injection entry point and `CUDA_INJECTION64_PATH` are documented for Linux
too (the CUPTI sample builds a `.so`), and nothing in `inject.cu` is
Windows-specific apart from the export macro. It is **not built or tested there**
(`POOLKIT_BUILD_INJECT` is applied on Windows only): nobody has run it on a Linux
driver here, and a target that no one exercised would be the first bug report of
the release. `-DPOOLKIT_BUILD_INJECT=ON -DPOOLKIT_WITH_CUDA=ON` builds it for
whoever wants to be the first.

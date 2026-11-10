# Trace format

A trace is the input every measurement in this repository shares: `poolkit_advise`
turns it into a recommendation, `poolkit_gate` compares two runs of the same one,
and `tests/test_trace.cpp` replays it on a CPU with no GPU. This document is the
contract — the parser in `include/poolkit/trace.hpp` rejects anything that is not
on this page, and it rejects it with a line number.

M6b ships the recorder ([`record.md`](record.md)): `poolkit_record` turns a real
CUDA program's `cudaMalloc`/`cudaFree` traffic into this JSONL through CUPTI
`MEMORY2`, with `ptr` set to the driver address as a decimal integer. The
committed [`traces/inference_transformer.jsonl`](../traces/inference_transformer.jsonl)
and the generator in `traces/generators/` remain what the host-only suite runs on
(a machine without a GPU has no trace to record).

---

## 1. One JSON object per line

JSONL: UTF-8, one object per line, `\n` separators, blank lines ignored. No
trailing commas, no comments, no arrays, no nested objects.

```json
{"op":"alloc","size":4096,"ptr":3,"stream":0,"ordinal":7}
{"op":"free","size":0,"ptr":3,"stream":0,"ordinal":12}
```

| field | type | presence | meaning |
|---|---|---|---|
| `op` | string | required | `"alloc"` or `"free"` |
| `size` | integer | required | bytes requested; `0` on a free |
| `ptr` | integer | required | trace-side block identifier, assigned by the recorder |
| `stream` | integer | required | logical stream id, opaque to the host replay |
| `ordinal` | integer | required | global order, strictly increasing |

Rules the parser enforces, each with the error message it produces:

* **every field is required.** There is no implicit default: a missing `stream`
  is an invalid trace, not stream 0. `trace line N: missing field 'stream'`.
* **`ordinal` is strictly increasing**, not merely non-decreasing. Two operations
  with the same ordinal have no defined order, and a replay with no defined order
  is not reproducible: `ordinal 3 is not greater than the previous 3`.
* **an `alloc` of size 0 is invalid.** It has no size class, and giving it one
  (256 B) would hide a caller bug behind padding.
* **unknown fields are invalid.** A typo (`"sizes"`) must not silently become
  "field missing, use the default" — there are no defaults.
* **integers only.** `"size": 4096.0` is invalid: bytes are counted, not measured.
* **the file must not be empty.**

## 2. Replay rules

`replay_trace(pool, trace)` drives a real `poolkit::Pool` through the operations
in file order (which the parser has already required to be ordinal order):

1. `alloc` calls `pool.allocate(size, stream)` and records the returned handle
   under `ptr`.
2. `free` looks up `ptr`, calls `pool.free(...)` and forgets the handle.
3. A `free` of a `ptr` that is not live — a double free, a free without an
   alloc, a ptr from another trace — throws `TraceError` with the ordinal.
4. An `alloc` that reuses a `ptr` still live throws too: two live handles with
   the same identity cannot be represented, so the trace is wrong, not the replay.
5. At the end of the trace the snapshot is taken **before** any handle is
   released, so `bytes_resident`, `tail_bytes` and the per-class statistics
   describe the state the workload actually left behind. Handles still live at
   the end (long-lived weights, typically) are the caller's to release; the
   replay does not invent frees for them.

The `stream` id is opaque to the host replay: `trace_stream()` casts it to the
`StreamHandle` type, which is `void*` in a host-only build and `cudaStream_t`
under CUDA. `FakeBackend` and `SimBackend` ignore it. It exists because a free
records an event on the stream that last used the block, and dropping the field
from the format would make the recorded traces unusable on a real device.

### Peak accounting

`TraceReplayResult` carries two numbers the snapshot does not:

| field | definition |
|---|---|
| `peak_bytes` | max over operations of *(chunks resident + bytes of live large-path blocks)* |
| `peak_live_blocks` | max number of handles alive at the same time |

The large path matters: an allocation above `max_class_bytes` goes straight to
the backend and is **not** part of `bytes_resident` (`PoolSnapshot` tracks it
separately in `bytes_large_live`). It is still memory the job occupies, so a peak
that ignored it would report a configuration as cheaper precisely because it
stopped pooling the expensive requests. `peak_bytes` includes it, and the
`covered` column of `poolkit_advise` says how much of the traffic that is.

## 3. Determinism

Replaying the same trace on the same backend yields the same counters, byte for
byte — `tests/test_trace.cpp` replays twice and compares every field of the
snapshot. That property is what makes a replay a measurement instead of a
benchmark: there is no clock, no scheduler, no allocator address lottery. Every
branch in `Pool` depends only on the operations read and the pool's own state.

The exception is not an exception: `SimBackend` and `FakeBackend` return different
*addresses* for the same chunk (real vs fake) and identical *counters*.
`tests/test_tools.cpp` pins the equality of peak, resident bytes and chunk count
between the two.

## 4. What can go wrong

| symptom | cause | what the tool says |
|---|---|---|
| `cannot open trace file` | wrong path, wrong working directory | `poolkit_advise: poolkit: cannot open trace file '...'` (exit 2) |
| `ordinal N is not greater than the previous M` | recorder emitted a batch out of order, or two traces concatenated | the line number of the second operation |
| `free of inactive ptr P` | recorder dropped an alloc line, or the trace was truncated mid-stream | the ordinal of the free |
| `alloc reuses ptr P` | recorder reused an identifier before the free was written | the ordinal of the second alloc |
| `an alloc with size 0 has no size class` | a zero-byte `cudaMalloc` was recorded | the line number |
| trace replays but the peak looks impossible | the trace is truncated: no frees | not an error — compare `allocs` and the live-block count in the report |

A trace that parses is not automatically a *representative* trace: a replay of
3 000 operations from a script is a model of a workload, not the workload. That
limit is stated in [`advise.md`](advise.md) §Limits.

## 5. Minimal example

```jsonl
{"op":"alloc","size":100,"ptr":0,"stream":0,"ordinal":0}
{"op":"alloc","size":300,"ptr":1,"stream":0,"ordinal":1}
{"op":"free","size":0,"ptr":0,"stream":0,"ordinal":2}
{"op":"alloc","size":200,"ptr":2,"stream":0,"ordinal":3}
{"op":"free","size":0,"ptr":1,"stream":0,"ordinal":4}
{"op":"free","size":0,"ptr":2,"stream":0,"ordinal":5}
```

Five operations, two blocks live at the peak. The three allocations land in two
classes of the default table: `100` → 256 and `200` → 256, while `300` → 512
(capacities are rounded up to a multiple of `min_class_bytes = 256`). Save it as
`small.jsonl` and:

```bash
$ ./build/poolkit_advise --trace small.jsonl --top 3
```

`tests/test_tools.cpp` uses exactly this shape for its hand-checked waste value
(`392 / 5888 = 6.6576 %` on the default table), so the arithmetic in the docs is
reproduced by the suite.

## 6. Writing traces by hand

Nothing stops you: the format is small enough that a shell heredoc is a valid
recorder. Two habits make the result useful:

* **`ptr` unique per live allocation.** It is the only identity the replay has;
  the committed traces use a monotone counter.
* **frees that match the allocs.** A trace where 1 538 blocks are allocated and
  3 freed is not wrong, it is a different workload (long-lived weights), and the
  advisor will report a peak as large as that implies.

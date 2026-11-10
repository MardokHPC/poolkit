# Metrics: what the pool reports, and what it does not

Five numbers, three of them called "fragmentation" for three different reasons.
`PoolSnapshot` (include/poolkit/metrics.hpp) is the whole surface; each field below is either
an integer that can be checked by adding up what you allocated, or a ratio defined here.

Every value in this document is produced by `poolkit_test_metrics` and `poolkit_test_pool_host`,
on the states described below. Nothing is estimated.

---

## 1. The fields

| field | definition | unit |
|---|---|---|
| `bytes_requested_live` | Σ `size()` over live handles, large path included | bytes |
| `bytes_live_in_pool` | the same sum, pool-backed blocks only | bytes |
| `bytes_capacity_live` | Σ `capacity()` over live pool-backed blocks | bytes |
| `bytes_resident` | Σ chunks obtained from the driver, free ones included | bytes |
| `bytes_large_live` | live large-path bytes (above `max_class_bytes`) | bytes |
| `bytes_free_in_pool` | ready + pending + never-handed-out slots | bytes |
| `quarantined_bytes` | capacity of blocks lost to a failed event record | bytes |
| `tail_bytes` | Σ over resident chunks of `chunk − blocks·capacity` | bytes |
| `high_water_mark_bytes` | max of `bytes_resident` over the pool's life | bytes |
| `release_all_refused` | `release_all()` calls rejected because handles were live | count |

The identity that ties them together, checked byte for byte in `test_pool_host.cpp`:

```
bytes_resident = bytes_capacity_live + bytes_free_in_pool
                 + quarantined_bytes + tail_bytes
```

`bytes_free_in_pool` is split into three, because they are not interchangeable:
`bytes_free_ready` (freed and reusable now), `bytes_free_pending` (freed, waiting for the
event recorded on the free), `bytes_unused_slots` (in a slab, never handed out).

## 2. The three fragmentation components

### 2.1 `internal_fragmentation` — the size class rounded the request up

```
internal_fragmentation = (Σ capacity_live − Σ size_live) / Σ size_live
                       = (bytes_capacity_live − bytes_live_in_pool) / bytes_live_in_pool
```

0 when nothing is live. Request-normalised, not capacity-normalised: the number answers "how
many extra bytes did I pay per byte I asked for". A pool holding nothing but 100-byte requests,
each parked in a 256-byte class, reports **1.56** — it paid 2.56 bytes for every byte requested.

The large path is excluded from both numerator and denominator: a large allocation is asked
from the driver at its exact size, so it has no rounding to report and would only dilute the
ratio.

### 2.2 `tail_fragmentation` — the slab carve stranded bytes

```
tail_fragmentation = Σ_chunks (chunk_bytes − blocks·class_size) / Σ_chunks chunk_bytes
                   = tail_bytes / bytes_resident
```

This is the part of a chunk that no block of that class can ever use: `2 MiB / c` leaves a
remainder whenever `c` does not divide the chunk. It is a property of the *carve rule*, not of
the workload, and it only goes away when the class divides the chunk (or when the class is
larger than the chunk and gets a dedicated one).

### 2.3 `resident_slack` / `free_ratio` — memory paid and not used

```
resident_slack = free_ratio = bytes_free_in_pool / bytes_resident
```

Both names, one number: `resident_slack` reads well next to the other two, `free_ratio` reads
well in a benchmark table. 0 when nothing is resident. This is the metric that deferred reuse
makes *large on purpose*: blocks whose event has not fired are counted here, and
`release_all()` is the way to give them back — once those events have completed, because it
asks them and never waits for a stream.

### 2.4 `quarantined_bytes` — memory the pool will never give back

Not a fragmentation component: a leak, reported as one.

If the backend cannot record the event on a `free`, the pool has no way to know whether the
stream is still writing into that block. The only two options are:

| choice | immediate | three days later |
|---|---|---|
| hand the block out again (**fail-open**) | one lost allocation, invisible | the next caller writes into memory a kernel still owns — silent data corruption |
| never hand it out again (**fail-closed**, shipped) | one block parked forever, visible in `quarantined_bytes` | nothing |

A driver hiccup is not an authorisation to reuse memory that may be in flight. So a
quarantined block is terminal: it is not in the ready list, it is not in the pending list, the
rescan never promotes it, and the chunk that holds it is kept by `release_all()` as well —
there is no event left to ask. It counts in `free_errors` too, so the leak has a cause
attached.

The same reasoning governs `release_all()`: called with a live handle it frees **nothing** and
returns 0, incrementing `release_all_refused`. Returning memory the caller can still write to
is corruption, not a trim.

## 3. `external_fragmentation(request)` — the one that cannot be a single number

```
c(request) = smallest class with capacity >= request       (SizeClassTable::index_for)
usable_free(request) = (ready_blocks + unused_blocks) · capacity   of the class c(request)
external_fragmentation(request) = 1 − usable_free(request) / bytes_free_in_pool
```

- 0 means every free byte could serve this request;
- 1 means none of it could: the free memory is in the wrong size class;
- 0 when nothing is free at all (there is nothing stranded either).

It is **one class, not every class at least as large as the request**: `Pool::allocate`
maps a request to exactly `c(request)` and never borrows a block from a bigger class, so
free slots there are free memory the pool will not use for this request. Counting them as
usable would report a number the allocator cannot honour — and it would understate the
fragmentation, which is the one thing this metric exists to avoid. Concretely, with two
free ready blocks in the 512 B class (`bytes_free_in_pool = 1024`) and nothing in the
256 B class:

| request | class the pool uses | `usable_free` | `external_fragmentation` |
|---|---|---:|---:|
| 256 B | 256 B (empty) | 0 | **1.0** — free, but in the wrong class |
| 512 B | 512 B | 1024 | 0.0 |
| 1024 B | none: above `max_class_bytes`, large path | 0 | 1.0 by the arithmetic, **not defined** as a fragmentation: there is no class to be wrong for |

A request above `max_class_bytes` never touches the pool table — it goes straight to the
driver — so the metric says nothing about it. `usable_free` returns 0 because no class
serves that request, and the 1.0 that follows is a division artefact, not a measurement:
on the large path, external fragmentation is undefined, not maximal.

Three deliberate choices, all visible:

* **`ready + unused`, never `pending`.** A freed block whose event has not completed exists,
  but it cannot be handed out yet. Counting it as usable would make the metric optimistic about
  exactly the window this pool is designed around.
* **Quarantined memory is not free.** A quarantined block is neither usable nor recoverable, so
  it appears in `quarantined_bytes` and not in `bytes_free_in_pool` — and therefore not here
  either. Stranded is not the same as lost.
* **It is a function, not a field.** The same snapshot answers 0.0 for a 256-byte request and
  1.0 for a 512-byte one over the same free memory. Any allocator that reports a single
  "fragmentation %" is choosing one request size on your behalf and not saying which.

## 4. Worked example, computed by hand and by the test

Config: `chunk_bytes = 1000`, `min_class_bytes = 256`, `max_class_bytes = 512`, `growth = 2`.
Two classes, `{256, 512}`. One live request per class (100 B and 300 B).

| quantity | arithmetic | value |
|---|---|---|
| chunks | class 256: `1000/256 = 3` blocks, tail `1000 − 768 = 232`<br>class 512: `1000/512 = 1` block, tail `1000 − 512 = 488` | 2000 B resident |
| `bytes_capacity_live` | `256 + 512` | 768 B |
| `bytes_live_in_pool` | `100 + 300` | 400 B |
| `internal_fragmentation` | `(768 − 400) / 400` | **0.92** |
| `tail_bytes` | `232 + 488` | 720 B |
| `tail_fragmentation` | `720 / 2000` | **0.36** |
| `bytes_free_in_pool` | class 256 has 2 unused slots, class 512 none: `2 · 256` | 512 B |
| `resident_slack` | `512 / 2000` | **0.256** |
| identity | `768 + 512 + 720` | 2000 ✓ |

Then free the 100 B block and ask `external_fragmentation`. The 256 B class now holds two
never-used slots and one block waiting for its event, so `bytes_free_in_pool = 2·256 + 256 =
768` and the 512 B class has no free slot at all (its single block is live):

* request 256 B: `usable_free = 2 · 256 = 512` of 768 → `1 − 512/768` = **0.333**. The other
  256 B is free but *pending*, not usable;
* request 512 B: `usable_free = 0` → **1.0**. Every free byte is in the wrong class.

Same pool, two requests, two different answers — and neither is "the fragmentation".

## 5. Hit rate and timings

```
hit_rate = reuse_count / (reuse_count + growth_count)
```

`reuse_count` counts allocations served by a block that had been freed before (including a
pending block promoted by the rescan, tracked separately in `recovered_count`);
`growth_count` counts allocations that consumed a slot never handed out before, whether the
slot was already in a slab or came with a new chunk. The per-class `ClassStats::hit_rate` is
the same ratio per class, and it is the number to look at when tuning a workload's class mix.

`alloc_avg_ns` / `alloc_max_ns` / `free_avg_ns` / `free_max_ns` are `std::chrono::steady_clock`
measurements of the pool's own work, in nanoseconds. They exclude whatever the caller does with
the block, and they include nothing the driver did asynchronously.

`Pool::poll()` is the explicit way to move `bytes_free_pending` into `bytes_free_ready`: one
`query` per pending block, no waiting, and it returns how many were promoted. Without it the
ready list only advances inside `allocate` — which is enough for a busy workload and useless
for looking at the numbers after a `cudaDeviceSynchronize`.

## 6. What is NOT measured

* **Device-side cost.** No `cudaEventElapsedTime`, no occupancy, no bandwidth: the timings are
  host-side nanoseconds. M4 adds the `cudaEventQuery` experiment separately.
* **Time spent in the driver.** `cudaMalloc` is inside `alloc_*_ns` when a chunk is created, but
  the split between "pool logic" and "driver call" is not reported — the `growth_count` and
  `chunk_count` fields are there to correlate it.
* **Peak per class.** Only the global `high_water_mark_bytes` exists; a per-class peak would
  need a second vector of maxima and has not been requested by any use case yet.
* **Anything about overlapping lifetimes.** Fragmentation here is a memory quantity, not a
  schedule. Two working sets that never coexist can show the same numbers as one that does.
* **Multi-device or multi-pool totals.** A `PoolSnapshot` describes one `Pool`.

# The advisor: which size classes does *this* trace want?

`poolkit_advise` answers one question with a measurement instead of a rule of
thumb: **given a trace of allocations, which size classes and which chunk
minimise the waste and the peak, and what does that cost in retention?**

It exists because the cost model in [`cost-model.md`](cost-model.md) can tell you
what a table costs *on average*, and nothing about what it costs *on your
workload*. `g = 1.5` was chosen on a frontier for a log-uniform placeholder; the
committed trace then showed 23.9 % of internal waste against the model's 18.0 %
(§9 of the same document). The advisor closes that gap: it integrates the
workload's own histogram, and then it replays the workload to see the peak.

---

## 1. Usage

```bash
poolkit_advise --trace traces/inference_transformer.jsonl
               [--chunk 2MiB] [--min-class 256] [--max-class 2MiB]
               [--growth 1.5] [--no-power-of-two] [--extra 1MiB,4MiB]
               [--json report.json] [--top 10]
```

| flag | default | meaning |
|---|---|---|
| `--trace <f>` | required | the trace to analyse ([format](trace-format.md)) |
| `--chunk <size>` | `2MiB` | chunk_bytes of the **reference** configuration |
| `--min-class <size>` | `256` | `min_class_bytes` of every swept table |
| `--max-class <size>` | `2MiB` | `max_class_bytes` of the **reference** configuration |
| `--growth <f>` | `1.5` | growth factor of the **reference** configuration |
| `--no-power-of-two` | off | reference table without the powers of two |
| `--extra <list>` | none | extra class capacities, comma separated |
| `--json <f>` | — | write the recommended configuration's report |
| `--top <n>` | `10` | rows in the table |

Sizes accept a plain byte count or a `K`/`KiB`/`M`/`MiB`/`G`/`GiB` suffix, always
powers of two. `KB` is deliberately not accepted: it means 1000 to some readers
and 1024 to others, and a chunk size that silently differs by 2.4 % is exactly
the kind of unit bug this toolchain exists to make visible.

Exit code: `0` on success — an advisor that finds nothing to improve still
succeeded — and `2` for invalid arguments or a trace it cannot read.

## 2. What it prints

Real output on the committed trace (abridged; `--top 5`):

```text
trace: `traces/inference_transformer.jsonl`
ops 3076, allocs 1538, distinct sizes 845, largest request 1098185 B
sha256 `31fc9efff2e99b2c7d1e75e89b26b1a21c322bf5069be1e6cd29a47ed9733b10`

sweep: 1457 configurations, 1456 distinct rows, 1448 replays

reference: g = 1.50, powers of two on, chunk 2.00 MiB, max class 2.00 MiB
k = 34, internal 15.234 %, tail 0.543 %, retention 68.00 MiB, peak 68.00 MiB, covered 100.000 %

| rank | growth | pow2 | chunk | max class | covered | k | internal | tail | retention k*C | peak sim | peak blocks |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.43 | no | 1.00 MiB | 1.00 MiB | 49.650 % | 23 | 6.555 % | 0.191 % | 23.00 MiB | 41.45 MiB | 39 |
| 2 | 1.39 | no | 1.00 MiB | 1.00 MiB | 49.650 % | 25 | 6.679 % | 0.116 % | 25.00 MiB | 41.45 MiB | 39 |
| 3 | 1.36 | no | 1.00 MiB | 1.00 MiB | 49.650 % | 26 | 6.849 % | 0.114 % | 26.00 MiB | 41.45 MiB | 39 |
| 4 | 1.52 | no | 1.00 MiB | 1.00 MiB | 49.650 % | 20 | 7.886 % | 0.156 % | 20.00 MiB | 41.45 MiB | 39 |
| 5 | 1.69 | no | 1.00 MiB | 1.00 MiB | 49.650 % | 16 | 8.471 % | 0.208 % | 16.00 MiB | 41.52 MiB | 39 |

## recommended

g = 1.70, powers of two off, chunk 1.00 MiB, max class 2.00 MiB -> k = 18 classes

peak simulated **43.63 MiB** with 39 blocks live at the peak; internal waste
8.715 %, tail 0.205 %, retention bound 18.00 MiB, covered 100.000 %

**why.** Lowest simulated peak among the configurations that keep max class at
2.00 MiB: 24.37 MiB below the reference (68.00 MiB); internal waste -6.518 pp vs
the reference; worst-case retention 50.00 MiB less than the reference, the price
of -16 classes.

**overall best, not recommended.** g = 1.43, powers of two off, chunk 1.00 MiB,
max class 1.00 MiB reaches 41.45 MiB (2.18 MiB less), but only 49.650 % of the
requested bytes would still be rounded by the table: the rest becomes one driver
allocation per block, never reused. This model prices bytes, not driver calls, so
the recommendation stays on your max class; `tools/bench` (M6b) is what measures
the difference.
```

## 3. The columns, and what each one is not

| column | definition | not |
|---|---|---|
| `k` | `SizeClassTable::count()` | not the classes the workload uses |
| `internal` | `Σ count(s)·(c(s)−s) / Σ count(s)·c(s)` over requests `s ≤ max class` | not normalised by requested bytes (that is `w/(1−w)`, larger) |
| `tail` | expected tail waste weighted by the trace's own class mix | not the replay's tail; this is the model, the replay's `tail_fragmentation` is the measurement |
| `retention k*C` | `k · chunk_bytes` | not the retained set: it is the worst case "one chunk per class" |
| `peak sim` | max over operations of *(resident chunks + live large-path bytes)* | not the end-of-replay resident set, which is smaller; not a GPU measurement |
| `peak blocks` | max handles alive at once | not the allocation count |
| `covered` | share of requested bytes that `max class` lets the table round | not a quality score |

`internal` is **capacity-normalised**, matching `docs/cost-model.md` §2 and
`tools/measure_classes.cpp`, so numbers in this document and in that one can be
compared. `covered` exists because a smaller `max class` makes `internal` look
better by sending the expensive requests somewhere the metric does not look: the
large path, one backend allocation per block, never reused.

`peak sim` is the reason to run the tool instead of reading the model. On the
committed trace the reference configuration's *measured* peak (68.00 MiB) happens
to equal its retention bound, because the workload touches all 34 classes and
each gets one 2 MiB chunk. On most traces it does not: the bound is a ceiling, and
the peak is what the job asks for.

## 4. The sweep

For every point of

```
include_power_of_two ∈ {false, true}
growth               ∈ [1.10, 2.00], step 0.01   (91 values)
chunk                ∈ {1, 2, 4, 8} MiB           (4 values)
max_class            ∈ {1, 2} MiB                 (2 values)
```

the tool computes the columns above, replaying the trace for the peak. That is
1 456 grid points plus the reference configuration. The work is deduplicated by
*class table and chunk*, which is what the replay actually depends on: 1 448
replays for the 1 457 configurations above, because some growth factors round to
identical tables — those are collapsed to the smallest `g` that produces the
table, and the row shows that `g`.

`min_class_bytes` and `extra_sizes` are not axes: they describe the hardware
alignment floor and the caller's known sizes. They apply to every swept table.

### Why these axes and not more

* `growth` is the knob the cost model already parameterises (`suggest_growth`);
  sweeping it here evaluates it against the workload instead of the placeholder.
* `chunk` is the retention unit: `k·C` grows linearly with it, and the tail moves
  with `C mod c`.
* `max_class` decides what the pool serves at all. It is in the sweep because the
  spec asks the question, and it is the axis with the biggest hidden cost — see
  §5.
* `min_class_bytes` is fixed at 256 B by the alignment promise (a weaker
  alignment can corrupt vectorised user code). It is a flag so that a port can
  measure a different floor, not so that the advisor can recommend one.

## 5. Why the recommendation stays on your `max class`

Ranking is by simulated peak, then internal + tail waste, then `k`, then a total
order on the configuration so the result never depends on iteration order. But
the *recommendation* is taken from the rows that keep the reference
`max_class_bytes`, and the overall best is reported separately when it differs.

The reason is not squeamishness. Crossing `max class` does not tune the table, it
changes which allocations the pool serves: everything above it becomes one
`cudaMalloc` per block, no carving, no reuse, no event. That is cheap in bytes —
which is why it wins the peak ranking by 2.18 MiB on this trace — and expensive
in driver calls, which this model does not price. Recommending it on the strength
of the byte count alone would be exactly the kind of confident, wrong advice this
repository is written against. `tools/bench` (M6b) measures the call cost, and
until it does, the advisor says what it can see and shows the rest.

## 6. No host memory proportional to the trace

A trace that describes 8 GiB of device memory must not need 8 GiB of host RAM to
analyse, or the analysis becomes a machine-spec problem instead of a policy
question. `SimBackend` (`include/poolkit/sim_backend.hpp`) returns 256 B-aligned
fake addresses from a bumped arena and keeps 32 bytes of bookkeeping per chunk;
it allocates none of the bytes it promises, and its event path mirrors
`FakeBackend`'s default (an event is complete when recorded). The equality of the
two backends — same peak, same resident bytes, same chunk count, same reuse
counters — is a test, not a comment (`tests/test_tools.cpp`,
`ToolsSimBackend.SamePeakAsFakeBackend`).

The full sweep above takes ~0.8 s on the committed trace in a Release build.

## 7. The JSON report

`--json` writes the **recommended** configuration in schema 1, the document
`poolkit_gate` consumes. See [`gate.md`](gate.md) §4 for the schema and the
stability rules.

## 8. Limits

* **The trace is the model.** Every number is a property of the trace you passed.
  A trace recorded from a script is not the workload; a trace recorded from the
  application is (M6b). If the trace is not representative, the advice is precise
  about the wrong thing — this is the first thing to check before believing a
  recommendation.
* **The peak is a simulation of this pool, not a GPU measurement.** It includes
  the pool's deferred reuse: a freed block is reusable only when its event
  completes, which the simulation treats as "immediately after the free", the
  same policy `FakeBackend` gives. A real stream with queued work can hold blocks
  longer. The measurement of that difference is M6b's benchmark.
* **Waste is modelled on requests, not on writes.** A block that is allocated and
  touched for one byte still costs its class size. That is the right model for
  capacity planning, not for bandwidth.
* **`retention k·C` is a worst case, not a prediction.** Read it next to
  `peak sim`: a workload touching a subset of the classes pays for a subset of
  the chunks.
* **Bytes only.** The advisor does not price allocation time, so it cannot say
  whether fewer, larger chunks beat more, smaller ones. Bench (M6b) does.
* **One process, one device.** No multi-GPU, no per-stream pools: the trace has a
  `stream` field, and v1.0's pool treats every stream the same way.

# CI memory gate

`poolkit_gate` turns two memory reports into a pass/fail decision, so a change
that makes the job need 10 % more GPU memory fails CI before it reaches `main`.

It is the space twin of `cuda-timer`'s `perf_gate` — same four verdicts, same
exit codes, same rule that a comparison which is not valid is `suspect` and not
`regression`. One of those two tools gates the time, this one gates the space,
and a person should not have to remember two conventions for the same decision.

## 1. Usage

```bash
poolkit_gate --baseline base.json --current cur.json
             [--max-peak-increase 0.05] [--max-waste-increase 0.02]
             [--baseline-trace a.jsonl --current-trace b.jsonl]
             [--json verdict.json] [--quiet]
```

| flag | default | meaning |
|---|---|---|
| `--baseline <f>` | — | baseline report (`poolkit_advise --json`) |
| `--current <f>` | — | report of the change under test |
| `--max-peak-increase <f>` | `0.05` | max **relative** growth of the peak |
| `--max-waste-increase <f>` | `0.02` | max growth of the waste, as a fraction (2 percentage points) |
| `--baseline-trace <f>` | — | alternative to a report: replay the trace with the reference configuration |
| `--current-trace <f>` | — | the same, for the current side |
| `--json <f>` | — | write the verdict document to a file |
| `--quiet` | off | do not print the verdict document on stdout |

Give **either** two reports **or** two traces. Traces are replayed with the same
reference configuration on both sides (`g = 1.5`, powers of two, 2 MiB chunk,
2 MiB max class), so the only thing that can differ is the trace itself — which
makes `--baseline-trace` the quick check and reports the fine one.

## 2. Exit codes and verdicts

| Exit code | Verdict | Meaning |
|---|---|---|
| `0` | `pass` | every metric inside its threshold |
| `0` | `improvement` | at least one metric beyond its threshold, in the good direction |
| `1` | `regression` | at least one metric beyond its threshold, in the bad direction |
| `2` | `suspect` | **the comparison is not valid** — different trace, different configuration, unknown schema, unreadable report |
| `2` | error | bad arguments, missing file, cannot write `--json` |

Precedence: `regression` wins over `improvement`. "The peak grew 10 % but the
waste improved 3 pp" is not a pass: the job may no longer fit.

`suspect` is a verdict, not an error path. It is the answer to "I cannot tell
whether this is better", and the honest response is to fix the comparison, not to
lower a threshold until the gate says `pass`. This is the lesson cuda-timer
already paid for with environment sidecars.

### What the gate decides on

| metric | kind | threshold |
|---|---|---|
| `peak_resident_bytes` | bytes | relative, `--max-peak-increase` |
| `internal_waste` | fraction | absolute, `--max-waste-increase` (pp) |
| `tail_waste` | fraction | absolute, `--max-waste-increase` (pp) |
| `retention_bound_bytes` (`k·C`) | bytes | relative, `--max-peak-increase` |
| `classes` (`k`) | count | reported only |

Waste is already a fraction of a fraction; comparing it relatively would make a
change from 0.2 % to 0.3 % look like a 50 % regression. It is compared in
percentage points.

`k` is reported and does not decide. A table with more classes and a smaller
chunk retains the same bytes, and calling that a regression would be wrong; the
cost shows up in `retention_bound_bytes` and in the waste, which are the things
that actually cost memory.

## 3. When the answer is `suspect`

| situation | reason string starts with |
|---|---|
| different trace file (SHA-256 differs) | `different trace: sha256 …` |
| same hash, different op/alloc counts (impossible unless the report is wrong) | `different trace: 3076/1538 ops/allocs vs …` |
| different configuration (growth, chunk, min/max class, powers of two, extra sizes) | `different configuration: growth 1.7 vs 1.6` |
| unknown or future `schema` | `poolkit: report schema 3 is not supported by this build (reads schema 1 and 2)` |
| a schema-2 report (bench) compared with a schema-1 report (analysis) | `schema 1 vs 2` |
| report is not an advise report | `not advise reports ('…')` |
| baseline peak of 0 | `baseline peak is 0: a relative change is undefined` |
| malformed JSON, missing field, unknown field, non-finite number | `poolkit: report JSON line N: …` |

Trace identity is the **file hash** when both reports carry one. If only one
does, the gate falls back to `ops`/`allocs` and says so in `notes`; if neither
does, it falls back and says so too. A report with `ops = 0` has no identity at
all and is `suspect`.

The configuration must be identical because two different configurations are two
different measurements. "g = 1.10 needs more chunks than g = 2.00" is a fact
about the grid, not a regression in your change.

## 4. The report schema (schema 1, and schema 2 from M6b)

`poolkit_advise --json` writes **schema 1**, and `poolkit_gate` reads it. It is
versioned and stable: a field is not renamed or removed inside schema 1, only
added with a schema bump. **Schema 2** is schema 1 plus a required `bench`
section, written by `poolkit_bench` ([`bench.md`](bench.md) §6); this reader
accepts both, and the gate still refuses to compare a bench report with an
analysis report (`not advise reports`), because those are different measurements.

```json
{
  "schema": 1,
  "tool": "poolkit_advise",
  "trace": {"path": "traces/inference_transformer.jsonl", "ops": 3076, "allocs": 1538,
            "sha256": "31fc9efff2e99b2c7d1e75e89b26b1a21c322bf5069be1e6cd29a47ed9733b10"},
  "config": {"growth": 1.7, "chunk_bytes": 1048576, "min_class_bytes": 256,
             "max_class_bytes": 2097152, "power_of_two": false, "extra_sizes": []},
  "classes": 18,
  "waste": {"internal": 0.087154, "tail": 0.00205},
  "peak": {"resident_bytes": 45754368, "blocks": 39},
  "retention_bound_bytes": 18874368
}
```

Guarantees:

* **deterministic.** The same trace and flags produce the same bytes, so a report
  can be committed and diffed. Object keys are emitted in a fixed order.
* **classic locale everywhere.** Doubles are written with `.`, never `,`, even if
  the host locale says otherwise; a locale with a decimal comma once corrupted a
  CSV silently in cuda-timer. Reading uses the classic locale too.
* **finite numbers only.** Writing a `NaN` or an infinity throws; reading one
  fails. `null` is not accepted for a required field.
* **strict.** Unknown fields, duplicate keys and a different `schema` are
  rejected with a line number instead of being ignored.
* **the trace path is the argument as given**, not a canonical path: the identity
  is the hash.

The verdict document is a different shape (`"tool": "poolkit_gate"`):

```json
{
  "schema": 1,
  "tool": "poolkit_gate",
  "verdict": "regression",
  "reason": "peak_resident_bytes grew 10.000 % > 5.000 %",
  "trace": {"sha256": "31fc9eff…", "ops": 3076, "allocs": 1538},
  "config": {"growth": 1.7, "chunk_bytes": 1048576, "min_class_bytes": 256,
             "max_class_bytes": 2097152, "power_of_two": false, "extra_sizes": []},
  "thresholds": {"max_peak_increase": 0.05, "max_waste_increase": 0.02},
  "metrics": [
    {"name": "peak_resident_bytes", "baseline": 45754368, "current": 50329805,
     "delta": 4575437, "relative": 0.1},
    {"name": "internal_waste", "baseline": 0.087154, "current": 0.087154,
     "delta": 0, "relative": 0},
    {"name": "tail_waste", "baseline": 0.00205, "current": 0.00205, "delta": 0, "relative": 0},
    {"name": "classes", "baseline": 18, "current": 18, "delta": 0, "relative": 0},
    {"name": "retention_bound_bytes", "baseline": 18874368, "current": 18874368,
     "delta": 0, "relative": 0}
  ],
  "notes": []
}
```

`relative` is `null` when the baseline is zero: the change is undefined, and
writing `0` would let a consumer mistake "cannot tell" for "no change".

`--quiet` suppresses the document on stdout but **not** the one-line verdict on
stderr:

```text
poolkit_gate: regression — peak_resident_bytes grew 10.000 % > 5.000 %
```

A CI log that says nothing is a CI log nobody reads.

## 5. Workflow

1. Record or generate the trace that represents the workload (M6b's
   `poolkit_record`; until then `traces/`).
2. Produce the baseline report and commit it next to the trace:

   ```bash
   ./build/poolkit_advise --trace traces/inference_transformer.jsonl \
                          --json bench/memory-baseline.json
   ```

3. In CI, produce the current report from the same trace, with the code under
   test:

   ```bash
   ./build/poolkit_advise --trace traces/inference_transformer.jsonl \
                          --json current.json
   ./build/poolkit_gate --baseline bench/memory-baseline.json --current current.json \
                        --json gate.json --quiet
   ```

4. Let the exit code decide: `1` fails the job, `2` fails it too and means "fix
   the comparison", not "ignore the gate".

## 6. GitHub Actions (self-hosted GPU runner)

The tools are host-only — no `nvcc`, no device — but the trace they read must
describe the real workload, so the useful runner is the one that recorded it.
Hosted `ubuntu-latest` works for the gate step itself.

```yaml
name: memory gate
"on":
  push:        { branches: [main] }
  pull_request: { branches: [main] }

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  memory:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Configure
        run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
                   -DPOOLKIT_BUILD_TOOLS=ON -DPOOLKIT_BUILD_TESTS=OFF
      - name: Build the tools
        run: cmake --build build --target poolkit_advise poolkit_gate
      - name: Advise the current build
        run: ./build/poolkit_advise --trace traces/inference_transformer.jsonl
                                    --json current.json
      - name: Memory gate
        run: ./build/poolkit_gate --baseline bench/memory-baseline.json
                                  --current current.json --json gate.json --quiet
      - name: Keep the verdict
        if: always()
        uses: actions/upload-artifact@v4
        with: { name: memory-verdict, path: gate.json }
```

`poolkit_test_tools` in the host suite is what keeps the gate honest: it asserts
the four exit codes and the suspect cases, so a refactor that quietly turns
`suspect` into `pass` fails the build.

## 7. What makes a comparison valid

* **The same trace.** The gate enforces the hash; keep the baseline trace in the
  repository next to the baseline report, or the day it changes nobody will know
  which trace produced the report.
* **The same configuration.** Change one knob at a time; the gate will refuse to
  compare across configurations, and that refusal is the point.
* **The same definition of the metrics.** That is what `schema` is for.
* **A representative trace.** The gate cannot tell that the baseline trace was
  recorded on a toy workload. See [`advise.md`](advise.md) §Limits.

## 8. When the workload itself changes

A change that alters *which* allocations the program makes changes the trace, so
the gate answers `suspect` — and it is right: comparing a 3 076-op trace with a
3 500-op one measures the feature, not the allocator. The workflow is to re-record
the trace and regenerate the baseline report in the same commit, which is a
reviewable act; a gate that let the comparison through would hide it.

The two questions are different and have different tools:

| question | tool |
|---|---|
| did this change make the same workload need more memory? | `poolkit_gate` |
| which classes and chunk does this new workload want? | `poolkit_advise` |

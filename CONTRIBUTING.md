# Contributing

poolkit is a library whose only real product is a *measurement*: a size class
table, a fragmentation number, a trace. So the rules below are mostly about
keeping numbers and tests honest, not about style.

- [Two toolchains on one Windows machine](#two-toolchains-on-one-windows-machine)
- [The helper scripts](#the-helper-scripts)
- [Running the tests](#running-the-tests)
- [The rules that are not negotiable](#the-rules-that-are-not-negotiable)
- [Regenerating every number in the docs](#regenerating-every-number-in-the-docs)
- [Traces: committed vs generated](#traces-committed-vs-generated)
- [Releasing](#releasing)

---

## Two toolchains on one Windows machine

The project is built with two different installations on purpose, and it matters
which one you use:

| configuration | toolchain | why |
|---|---|---|
| **host-only** (`POOLKIT_WITH_CUDA=OFF`, the default) | **Visual Studio 18** Community (it announces itself as *Visual Studio 2026 Developer Command Prompt*, 18.10): `VC\Auxiliary\Build\vcvars64.bat`, MSVC 14.51, Ninja | this is what a user with no GPU and no CUDA toolkit gets. It must configure, compile warning-free and pass every host test without `nvcc` ever running. If it needs the toolkit, the property the whole test strategy rests on is gone. |
| **device** (`POOLKIT_WITH_CUDA=ON`) | **Visual Studio 2022 BuildTools** + **CUDA 12.6** (`nvcc` from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`) | CUDA 12.6 is the toolkit the device numbers in `docs/` were taken with. MSVC 14.44 is newer than CUDA 12.6 knows, so `NVCC_PREPEND_FLAGS=--allow-unsupported-compiler` is required — the host code is still compiled with `/W4 /WX` through `-Xcompiler`, so nothing is being let through. |

`nvcc` is also on `PATH` from CUDA 12.1 on this machine. The build therefore
**pins `CMAKE_CUDA_COMPILER`**; do not let CMake pick a toolkit by accident, and do
not configure the device tree from a prompt that cannot see `cl.exe` (silently
using a different MSVC is how the `CUDA::cupti` target can end up pointing at a
different toolkit than `nvcc`).

Linux (CI configuration): GCC from the distribution, CUDA 11.8 or 12.1, Ninja,
`cmake >= 3.25` for the `CUDA::cupti` target.

## The helper scripts

Two `.bat` files at the repository root, both `configure → build → ctest`:

```bat
build-host.bat            :: host-only tree in build-host/, VS 18
build-host.bat clean      :: same, after deleting the tree (a from-scratch run)
build-device.bat          :: device tree in build-dev/, VS 2022 + CUDA 12.6
build-device.bat clean
```

They call the right `vcvars64.bat` themselves, so they can be started from any
`cmd.exe` — including one that already has the *other* toolchain in its
environment, which is the point.

Doing it by hand, if you need a different directory or extra flags — note that a
**Developer Prompt** (`x64 Native Tools Command Prompt`) is required, or keep the
`call vcvars64.bat` line as written: a bare `ctest --test-dir build-host` from a plain
`cmd.exe` fails `poolkit_consumer_install` because `rc.exe` is not on `PATH`. The `.bat`
scripts set the environment themselves, which is why they work from anywhere:

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build-host -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DPOOLKIT_BUILD_TESTS=ON -DPOOLKIT_BUILD_TOOLS=ON -DPOOLKIT_WITH_CUDA=OFF
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set NVCC_PREPEND_FLAGS=--allow-unsupported-compiler
cmake -S . -B build-dev -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" ^
      -DCMAKE_CUDA_ARCHITECTURES=75 -DPOOLKIT_BUILD_TESTS=ON -DPOOLKIT_WITH_CUDA=ON
cmake --build build-dev
ctest --test-dir build-dev --output-on-failure
```

GoogleTest is fetched by CMake when tests are enabled, pinned to a tag; the first
configure needs network access.

## Running the tests

```bash
ctest --test-dir build-host --output-on-failure     # everything that needs no GPU
ctest --test-dir build-dev  --output-on-failure     # the same, plus the device suites
ctest --test-dir build-dev  -R poolkit_test_device  # one suite
```

A `[[nodiscard]]` call inside `EXPECT_THROW` needs an explicit `(void)` cast: the
macro discards the value, and GCC turns that into `-Werror=unused-result`. MSVC does
not diagnose it, so this is a mistake the local host build cannot catch — write
`EXPECT_THROW((void)pool.allocate(50000), poolkit::PoolError)`; the same holds for
`ASSERT_THROW`, `EXPECT_NO_THROW` and `ASSERT_NO_THROW`.

**`77` means "this environment cannot run the test"**, and it is registered as
`SKIP_RETURN_CODE` on every entry, so a machine without a card reports `SKIPPED`
instead of a green that hides a test that never ran. The device suites — and
`poolkit_example_basic_usage`, and the two recorder tests — are the ones that skip.

`poolkit_bench` and `poolkit_measure_m4` are *tools*, not tests: they need a quiet
card and wall time, so they are built but not registered. Run them by hand; the
numbers they print go in a document with the date and the config next to them.

## The rules that are not negotiable

1. **Warnings are errors.** `/W4 /WX` on MSVC, `-Wall -Wextra -Wpedantic -Werror`
   on GCC/Clang, and the same host flags reach the `.cu` files through
   `-Xcompiler` (plus both CUPTI libraries: `poolkit_record`, `poolkit_inject`).
   If a vendor header is noisy, silence it around the include — not in the file,
   and never by dropping a flag.
2. **No weakened test.** A test that fails is fixed by fixing the code, or by
   changing what the *documented contract* is — never by loosening the assertion so
   the new behaviour passes. If a test cannot run here (no GPU, no CUPTI, no
   compiler), it returns `77` and is registered as a skip; that is visible at
   configure/test time and is honest. Deleting an assertion is not.
3. **No invented numbers.** Every figure in `README.md` and `docs/` comes out of a
   tool in this repository, on a named stack (GPU, driver, CUDA, config), and each
   section says how to regenerate it. If a number cannot be reproduced, it is
   removed, not rounded.
4. **The core headers stay CUDA-free.** `backend.hpp`, `pool.hpp`, `slab.hpp`,
   `metrics.hpp`, `size_class*.hpp`, `trace.hpp` must compile with no toolkit
   installed. `POOLKIT_HAS_CUDA_RUNTIME` is the only switch, and it lives in
   `config.hpp`.
5. **A trace that parses must replay.** Anything that writes JSONL (`poolkit_record`,
   `poolkit_inject`, the generators in `traces/generators/`) has to produce a file
   the parser accepts and the replay can drive: `ordinal` strictly increasing, no
   orphan free, no duplicate live `ptr`, no unknown fields.
6. **`README.md` and `DEVELOPMENT_PLAN.md` are the maintainer's.** The plan is not
   published (see `.gitignore`).
7. **No secrets in a commit.** No tokens, no machine-specific paths, no
   `CMakeCache.txt`. Check `git status` and `git diff` before you commit.

## Regenerating every number in the docs

All commands are relative to the repository root, with the binaries from
`build-host/` (host tools) or `build-dev/` (device tools).

| document | tool | command |
|---|---|---|
| `docs/cost-model-sweep.md` | `poolkit_sweep_growth` | `poolkit_sweep_growth docs/cost-model-sweep.md` (stdout when the path is omitted); the shipped minimum is pinned by `ctest -R poolkit_sweep_growth` |
| `docs/cost-model.md` §9 (before/after of the class union) | `poolkit_measure_classes` | `poolkit_measure_classes traces/inference_transformer.jsonl`, pinned by `ctest -R poolkit_measure_classes` |
| `docs/advise.md` (the recommendation block) | `poolkit_advise` | `poolkit_advise --trace traces/inference_transformer.jsonl --top 5`, pinned by `ctest -R poolkit_advise_committed_trace` |
| `docs/gate.md` (the four verdicts) | `poolkit_gate` | the five `poolkit_gate_cli_*` ctest entries are the contract: pass, regression, improvement, suspect trace, suspect config |
| `docs/bench.md` §5, `docs/measurements-m6.md` §2 | `poolkit_bench` (device) | `poolkit_bench --trace traces/inference_transformer.jsonl --repeat 5 --warmup 1 --json bench.json` |
| `docs/measurements-m4.md` (granularity, `cudaEventQuery` cost, hit vs growth) | `poolkit_measure_m4` (device) | run it; it is not a test (quiet card, wall time) |
| `docs/record.md` §5 (recorder overhead) and §9.2 (injected recording) | `poolkit_record`, `poolkit_inject` | see below |
| `docs/trace-format.md` | `include/poolkit/trace.hpp` | the parser *is* the contract; `tests/test_trace.cpp` is its test |

Recording, the two ways:

```bat
:: a workload library, loaded in-process by the recorder
build-dev\poolkit_record.exe -o record_target.jsonl --workload build-dev\poolkit_record_target_workload.dll
build-dev\poolkit_record_check.exe record_target.jsonl

:: an unmodified program, through the injection shim
set CUDA_INJECTION64_PATH=%CD%\build-dev\poolkit_inject.dll
set POOLKIT_RECORD_OUT=inject_target.jsonl
build-dev\poolkit_record_target.exe 400
build-dev\poolkit_record_check.exe inject_target.jsonl --injected
```

Both are `ctest` entries (`poolkit_record_target_record` →
`poolkit_record_trace_replays`, `poolkit_inject_record_target` →
`poolkit_inject_trace_replays`), so a change that breaks the recorder breaks the
build. `docs/record.md` §9.1 explains why the injected trace can lose its last
~570 operations and why the exit path calls no CUPTI function at all: do not
"fix" that by adding a flush to the exit handler, the process stops being
killable (the transcript is in the document).

When you quote a measurement: one process at a time, one mode at a time, 1 warmup
+ 5 runs, medians and p99 (never a mean), and the config (GPU, driver, CUDA, `g`,
chunk, `max class`) printed next to the number. `docs/bench.md` and
`docs/measurements-m6.md` are the examples to copy.

## Traces: committed vs generated

`traces/inference_transformer.jsonl` and its generator
`traces/generators/inference_transformer.py` are committed: they are the input the
host-only CI runs on, since a machine without a card has no trace to record.

Everything recorded by a test or a manual run (`*.jsonl` in a build tree, in the
repository root, or the ones ctest writes next to the binaries) is generated and
ignored. Do not commit a recording: it is machine- and run-specific, and the
`ordinal`/`ptr` fields make it unreviewable.

## Releasing

1. `POOLKIT_VERSION_MAJOR/MINOR/PATCH` and `POOLKIT_VERSION_STRING` in
   `include/poolkit/config.hpp`, and `project(poolkit VERSION ...)` in
   `CMakeLists.txt`, must agree with the top entry of `CHANGELOG.md`. They are three
   places and a typo in one of them is a support ticket.
2. `CHANGELOG.md` gets a `## [x.y.z] - YYYY-MM-DD` section with `Added` /
   `Changed` / `Fixed`; the `[Unreleased]` links at the bottom point at the tag.
3. `ctest` green in both trees, from scratch (`build-host.bat clean`,
   `build-device.bat clean`): the installed-package consumer test and the two
   recorder tests only mean something when nothing stale is in the tree.
4. Tag `vX.Y.Z` and publish the release. **The tag is created by the maintainer**,
   never by a build script or by CI: a tag is a statement about what was verified.

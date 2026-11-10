# Sanitizer report — v1.0.0

Transcript of the `compute-sanitizer` runs behind the claim in the README ("memcheck on the deferred-reuse tests:
0 errors"). Machine: RTX 2060 SUPER (sm_75), driver 610.88, CUDA 12.6.20, Windows 11.
Tool: **NVIDIA Compute Sanitizer 2024.3.0.0 (build 34551201)**.

Reproduce from a Developer Prompt, after `build-device.bat`:

```bat
call "%CUDA_PATH%\bin\compute-sanitizer.bat" --tool memcheck --error-exitcode 1 ^
     build-dev\poolkit_test_device.exe --gtest_filter=DevicePool.*
call "%CUDA_PATH%\bin\compute-sanitizer.bat" --tool racecheck --error-exitcode 1 ^
     build-dev\poolkit_test_device.exe --gtest_filter=DevicePool.*
call "%CUDA_PATH%\bin\compute-sanitizer.bat" --tool memcheck --error-exitcode 1 ^
     build-dev\poolkit_example_basic_usage.exe
```

`--gtest_filter` is deliberate: the full suite under `racecheck` is slow enough to be killed mid-analysis on this
machine, and an interrupted run reports nothing useful. The filter covers the four tests that exercise real device
memory and the deferred-reuse path.

## memcheck — `DevicePool.*`

```
[==========] 4 tests from 1 test suite ran. (420 ms total)
[  PASSED  ] 4 tests.
========= ERROR SUMMARY: 0 errors
```

| test | what it touches |
|---|---|
| `RealAllocationReturnsAlignedValidPointers` | `cudaMalloc` chunks, pointer arithmetic, 256 B alignment |
| `DeferredReuseWaitsForTheRealEvent` | kernel writes into a block, `free`, event, reuse, `cudaMemcpy` back |
| `AllocateAndFreeDoNotBlockOnTheDevice` | 10 000 allocate/free with no synchronisation |
| `ReleaseAllRefusesWhileAHandleIsAlive` | `release_all` refusal path |

## racecheck — `DevicePool.*`

```
[==========] 4 tests from 1 test suite ran. (418 ms total)
[  PASSED  ] 4 tests.
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

## memcheck — `examples/basic_usage.cu`

```
after three allocations:
  resident           4194304 B in 2 chunks
  live               8654848 B requested, 266240 B capacity
  large path         8388608 B in 1 blocks (never pooled, never reused)
  fragmentation      internal 0.00 %, tail 0.00 %, resident slack 93.65 %
  reuse/growth       0 / 2 (hit rate 0.0 %)
  usable free for 4 KiB: 2093056 B
poll promoted 2 blocks to reusable
release_all returned 4194304 B, resident is now 0 B
========= ERROR SUMMARY: 0 errors
```

## What this does and does not prove

**Proves:** no invalid device reads/writes, no out-of-bounds, no leaked device allocations at kernel level in the
covered tests; no shared-memory hazards in the kernels those tests launch.

**Does not prove:** the absence of host-side lifetime bugs. `racecheck` reasons about shared memory inside kernels;
`memcheck` flags a device access to freed memory only if a kernel actually performs it. The real guard for our own
failure modes — a block handed out while its work is in flight, a chunk freed with a handle alive, a slab published
after a failed insertion — is the host test suite with `FakeBackend`, plus the deterministic stress test that checks
an interval map at every one of 10 000 operations. Sanitizers cannot replace that, and this document does not claim
they do.

**Not covered:** the CUPTI recorder and the injection shim (they are instrumented by CUPTI itself; running them under
compute-sanitizer would mean two instrumentation layers on the same process).

## Known environment noise

Two wedged processes from a failed injection experiment (`poolkit_example_basic_usage.exe`, `poolkit_record_target.exe`)
were still resident on the machine during these runs. They are idle and pinned in a driver call; `taskkill /F` reports
"no running instance" while `tasklist` still lists them, and only a reboot clears them. They do not allocate, so they do
not affect the result, but a reader reproducing on a clean machine should expect the same output rather than a
bit-identical timing.

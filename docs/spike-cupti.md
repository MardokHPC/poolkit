# Spike: CUPTI as a trace recorder (M5)

Question: can poolkit ship a recorder that sees every driver allocation with **size, address
and timestamp**, so `traces/*.jsonl` stops being a generator and becomes a measurement? This is
the throwaway experiment that answers it. It is not part of the library and is off by default
(`-DPOOLKIT_BUILD_SPIKE=ON`, requires `POOLKIT_WITH_CUDA=ON`).

**Answer: yes.** On Windows + CUDA 12.6 + sm_75, `CUPTI_ACTIVITY_KIND_MEMORY2` delivers one
record per `cudaMalloc`/`cudaFree` with the three fields the trace format needs.

## Environment

| field | value |
|---|---|
| GPU | RTX 2060 SUPER, sm_75 |
| driver | 610.88 |
| CUDA toolkit / nvcc | 12.6.20 |
| CUPTI | 2024.3.0 (`extras\CUPTI\lib64\cupti64_2024.3.0.dll`), header `CUPTI_API_VERSION` = 24 |
| CMake / host compiler | 4.4.3 / MSVC 19.44 (VS 2022 BuildTools) |
| privileges | none: plain user process, no admin, no profiling permission prompt |

## What it links

```cmake
add_executable(poolkit_spike_cupti_memory tools/spike/cupti_memory_spike.cu)
target_link_libraries(poolkit_spike_cupti_memory PRIVATE CUDA::cupti CUDA::cudart)
```

`CUDA::cupti` is the imported target from `find_package(CUDAToolkit)`; it adds
`<root>/extras/CUPTI/include` and links `extras/CUPTI/lib64/cupti.lib`. At run time the DLL is
**not** copied next to the executable, so the directory has to be on `PATH`. One vendor-header
wart: `cupti_activity.h` triggers MSVC `C4324` inside the header itself, so the spike pushes
`#pragma warning(disable : 4324)` around the `#include` and nowhere else (the build keeps
`/W4 /WX`).

Source: `tools/spike/cupti_memory_spike.cu`, **167 lines**, of which the CUPTI part is ~55. API
surface used: `cuptiSubscribe`, `cuptiActivityRegisterCallbacks`,
`cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2)`, `cuptiActivityFlushAll`,
`cuptiActivityGetNextRecord` → `CUpti_ActivityMemory3`
(`memoryOperationType`, `address`, `bytes`, `timestamp`, `deviceId`, `contextId`).

## Workload and output

The program subscribes before the first CUDA call, then: `cudaMalloc(1 MiB)` → kernel writing
256 floats → `cudaFree` → `cudaMalloc(4 KiB)` → `cudaFree` → flush. Actual output:

```
=== CUPTI MEMORY2 spike ===
CUPTI header version: 24
MEMORY2 op=alloc address=0x706000000 bytes=1048576  timestamp_ns=456650879872300 device=0 context=1
MEMORY2 op=free  address=0x706000000 bytes=1048576  timestamp_ns=456650881267700 device=0 context=1
MEMORY2 op=alloc address=0x706000000 bytes=4096     timestamp_ns=456650881517300 device=0 context=1
MEMORY2 op=free  address=0x706000000 bytes=4096     timestamp_ns=456650881522200 device=0 context=1
records=4 allocations=2 releases=2 allocated=1052672 released=1052672
SPIKE OK: MEMORY2 delivers size, address and timestamp per operation
```

Exit code 0, gated on: ≥ 2 allocations and ≥ 2 releases, and the byte totals equal
`1 MiB + 4 KiB` both ways. A false positive would have to invent four records with the exact
addresses and sizes the program used.

## Caveats a real recorder has to plan for

1. **Records are batched.** They arrive when a buffer fills or on `cuptiActivityFlushAll`; a
   recorder cannot assume one callback per operation. Buffers are handed to the client, so the
   sink must either copy what it needs or own the buffer.
2. **`timestamp` is the device timeline in ns**, not `steady_clock`. Correlating with host time
   needs `cuptiGetTimestamp` / the runtime-API activity kinds; `correlationId` is already in the
   record for that.
3. **It sees every allocation in the process**, driver internals included. Selecting "the pool's
   chunks" is a filter on the caller's own wrapper, or on address ranges it recorded itself.
4. **Overhead is not measured here.** Enabling MEMORY2 alone in an idle process is cheap, but a
   real inference run is the number that matters, and this spike did not produce it.
5. `CUPTI_ACTIVITY_KIND_MEMORY2` is the modern kind; `CUPTI_ACTIVITY_KIND_MEMORY` is the
   one-record-per-block legacy variant.

## Consequence for M6

The recorder is feasible without extra privileges, and the schema it needs (`address`, `bytes`,
alloc/free, timestamp) is exactly what `CUpti_ActivityMemory3` carries. The recommended shape is
therefore: wrap `cudaMalloc`/`cudaFree` in a counting backend for the pool's own chunk traffic
(cheap, always available), and use CUPTI as the independent verification channel — which is what
a trace recorder has to be if the trace is to be trusted.

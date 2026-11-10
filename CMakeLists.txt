cmake_minimum_required(VERSION 3.18)

project(poolkit
    VERSION 1.0.0
    DESCRIPTION "Header-only GPU memory pool with size classes derived from a cost model"
    HOMEPAGE_URL "https://github.com/MardokHPC/poolkit"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(POOLKIT_BUILD_TESTS "Build the host test suite (fetches GoogleTest)" OFF)
option(POOLKIT_BUILD_TOOLS "Build the host-only command line tools" ON)
option(POOLKIT_WITH_CUDA "Enable the CUDA backend glue (needs nvcc)" OFF)
# Throwaway experiment from M5: can a trace recorder be built on CUPTI? Off by
# default, not installed, not part of any test. See tools/spike/.
option(POOLKIT_BUILD_SPIKE "Build the CUPTI activity spike in tools/spike" OFF)

# The CUPTI recorder (poolkit_record) and the injection shim (poolkit_inject) are
# only built when the toolkit exports the CUDA::cupti target (CMake >= 3.25).
# Older CMake would therefore skip them *silently*, and a "green" CI job would
# quietly have stopped testing the recorder. The CI job that installs CUDA passes
# -DPOOLKIT_REQUIRE_CUPTI=ON so that the absence is a configure error instead.
option(POOLKIT_REQUIRE_CUPTI "Fail the configure if CUPTI (CUDA::cupti) is missing" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POOLKIT_WITH_CUDA)
    # The CUDA language is opt-in: the host-only build (the default, and the CI
    # job that proves the core needs no GPU) never sees nvcc. With CUDA on, the
    # build needs a CUDA-capable compiler in PATH (nvcc from the CUDA toolkit)
    # and a matching MSVC host compiler.
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # sm_75 is the reference hardware (RTX 2060 SUPER); overridable with
    # -DCMAKE_CUDA_ARCHITECTURES=...
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
elseif(POOLKIT_REQUIRE_CUPTI)
    message(FATAL_ERROR
        "POOLKIT_REQUIRE_CUPTI=ON requires POOLKIT_WITH_CUDA=ON: CUPTI is part of the "
        "CUDA toolkit, and asking for it without the toolkit cannot succeed.")
endif()

# ---------------------------------------------------------------------------
# Warnings: fatal everywhere, on the CXX language only
#
# One genex per flag, not `$<...:-Wall -Wextra>`: a generator expression result
# is not split on spaces, so the one-string form silently passes "-Wall -Wextra"
# as a single malformed option with some generators.
# ---------------------------------------------------------------------------
function(poolkit_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:-Wall>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:-Wextra>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:-Wpedantic>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:-Werror>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    )
endfunction()

# Fatal warnings for CUDA sources: the host code inside a .cu and the library
# headers it includes are compiled by the *host* compiler behind nvcc, so the
# flags have to reach it through -Xcompiler (plan §Registro dei rischi: "stessi
# flag sui .cu via -Xcompiler"). Which flags those are depends on that compiler,
# not on the platform: /W4 /WX is MSVC, -Wall -Wextra -Werror is GCC/Clang —
# without -Wpedantic, which trips over nvcc's own generated `#line` directives.
# Passing /W4 to g++ is what used to break the `cuda` CI job on Linux, where
# nvcc's host compiler is GCC.
# nvcc's own warnings are not promoted to errors here on purpose: the host flags
# are the ones that gate the headers the library ships.
function(poolkit_cuda_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # One -Xcompiler with a comma-separated list: nvcc forwards each element
        # to the host compiler as a separate argument.
        #
        # -Wpedantic is deliberately absent. It is on for the host C++ build, but
        # nvcc's own generated code emits `#line` directives that GCC flags as
        # "-Wpedantic: style of line directive is a GCC extension", and with
        # -Werror that turns every .cu into a failed build. cudafe++ output is not
        # ours to fix; the flags here gate the headers, and -Wall -Wextra -Werror
        # still do that on the host code inside the .cu.
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
        )
    else()
        message(WARNING
            "poolkit: unknown CUDA host compiler '${CMAKE_CXX_COMPILER_ID}'; "
            "building the .cu files without fatal host warnings")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# INTERFACE target (header-only)
#
#   add_subdirectory(poolkit) / FetchContent -> poolkit::poolkit
#   find_package(poolkit CONFIG)             -> poolkit::poolkit
# ---------------------------------------------------------------------------
add_library(poolkit INTERFACE)
add_library(poolkit::poolkit ALIAS poolkit)

target_include_directories(poolkit INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(poolkit INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# CUDA glue (only with POOLKIT_WITH_CUDA=ON)
#
# poolkit_cuda is the one translation unit that needs nvcc: the CudaBackend
# methods declared in pool.cuh. It links the CUDA runtime transitively so a
# consumer (the device test, the measurement tool) only has to link poolkit_cuda.
# ---------------------------------------------------------------------------
if(POOLKIT_WITH_CUDA)
    add_library(poolkit_cuda STATIC src/pool.cu)
    target_link_libraries(poolkit_cuda PUBLIC poolkit::poolkit CUDA::cudart)
    set_target_properties(poolkit_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    poolkit_cuda_warnings(poolkit_cuda)
endif()

# ---------------------------------------------------------------------------
# Tools (host-only, no CUDA, no link dependencies)
#
# poolkit_sweep_growth regenerates docs/cost-model-sweep.md. It links the same
# headers the library ships, so the published numbers cannot drift from the
# code that produces them.
# ---------------------------------------------------------------------------
if(POOLKIT_BUILD_TOOLS)
    add_executable(poolkit_sweep_growth tools/sweep_growth.cpp)
    target_link_libraries(poolkit_sweep_growth PRIVATE poolkit::poolkit)
    poolkit_warnings(poolkit_sweep_growth)

    # M5: before/after of the "popular sizes" union on the committed trace.
    # Links the trace reader too (still host-only: trace.hpp includes pool.hpp,
    # never a CUDA header).
    add_executable(poolkit_measure_classes tools/measure_classes.cpp)
    target_link_libraries(poolkit_measure_classes PRIVATE poolkit::poolkit)
    poolkit_warnings(poolkit_measure_classes)

    # M6a: the two CLIs that are the product. Host-only, no CUDA header, no
    # dependency beyond the headers the library ships. `tools/*.hpp` is where the
    # work lives (advise.hpp, gate.hpp, report_json.hpp); these two translation
    # units are argument parsing.
    add_executable(poolkit_advise tools/advise.cpp)
    target_link_libraries(poolkit_advise PRIVATE poolkit::poolkit)
    poolkit_warnings(poolkit_advise)

    add_executable(poolkit_gate tools/gate.cpp)
    target_link_libraries(poolkit_gate PRIVATE poolkit::poolkit)
    poolkit_warnings(poolkit_gate)

    # Pinned to the trace file, so the numbers quoted in README/docs are
    # reproduced by `ctest` and not only by whoever wrote the document.
    add_test(NAME poolkit_measure_classes
        COMMAND poolkit_measure_classes ${PROJECT_SOURCE_DIR}/traces/inference_transformer.jsonl)
    set_tests_properties(poolkit_measure_classes PROPERTIES
        PASS_REGULAR_EXPRESSION "powers of two \\(shipped\\)")

    # The advisor on the committed trace: a full sweep plus a real replay, with
    # the recommendation block the docs quote.
    add_test(NAME poolkit_advise_committed_trace
        COMMAND poolkit_advise --trace ${PROJECT_SOURCE_DIR}/traces/inference_transformer.jsonl
                --top 5)
    set_tests_properties(poolkit_advise_committed_trace PROPERTIES
        PASS_REGULAR_EXPRESSION "## recommended")

    # Running it is a test: it exercises every class table from g = 1.10 to 2.00
    # and the slab arithmetic for each class. The regex pins the published
    # minimum, so a change in the cost model cannot slip through unnoticed.
    add_test(NAME poolkit_sweep_growth COMMAND poolkit_sweep_growth)
    set_tests_properties(poolkit_sweep_growth PROPERTIES
        PASS_REGULAR_EXPRESSION "minimum total waste: g = 1.10")

    # Gate exit codes at the CLI boundary, not only in the unit tests: the four
    # verdicts are the contract, and a wrapper that forgot to return the code
    # would still pass a "did it print regression" check.
    set(POOLKIT_TOOLS_TEST_DIR ${PROJECT_SOURCE_DIR}/tests/tools)
    function(poolkit_expect_exit name tool expected)
        add_test(NAME ${name} COMMAND ${CMAKE_COMMAND}
            -DTOOL=$<TARGET_FILE:${tool}>
            -DEXPECTED=${expected}
            "-DARGS=${ARGN}"
            -P ${POOLKIT_TOOLS_TEST_DIR}/run_expect.cmake)
    endfunction()

    poolkit_expect_exit(poolkit_gate_cli_pass poolkit_gate 0
        --baseline ${POOLKIT_TOOLS_TEST_DIR}/data/report_baseline.json
        --current ${POOLKIT_TOOLS_TEST_DIR}/data/report_baseline.json)
    poolkit_expect_exit(poolkit_gate_cli_regression poolkit_gate 1
        --baseline ${POOLKIT_TOOLS_TEST_DIR}/data/report_baseline.json
        --current ${POOLKIT_TOOLS_TEST_DIR}/data/report_peak_regression.json)
    poolkit_expect_exit(poolkit_gate_cli_improvement poolkit_gate 0
        --baseline ${POOLKIT_TOOLS_TEST_DIR}/data/report_baseline.json
        --current ${POOLKIT_TOOLS_TEST_DIR}/data/report_peak_improvement.json)
    poolkit_expect_exit(poolkit_gate_cli_suspect_trace poolkit_gate 2
        --baseline ${POOLKIT_TOOLS_TEST_DIR}/data/report_baseline.json
        --current ${POOLKIT_TOOLS_TEST_DIR}/data/report_other_trace.json)
    poolkit_expect_exit(poolkit_gate_cli_suspect_config poolkit_gate 2
        --baseline ${POOLKIT_TOOLS_TEST_DIR}/data/report_baseline.json
        --current ${POOLKIT_TOOLS_TEST_DIR}/data/report_other_config.json)
endif()

# ---------------------------------------------------------------------------
# M4 measurement tool (device, only with POOLKIT_WITH_CUDA=ON)
#
# Produces the numbers that fill docs/measurements-m4.md: real cudaMalloc
# granularity, cudaEventQuery cost, pool allocate/free hit vs growth. Not a
# test (it needs a quiet GPU and wall time), so it is built but not registered.
# ---------------------------------------------------------------------------
if(POOLKIT_WITH_CUDA)
    add_executable(poolkit_measure_m4 tools/measure_m4.cu)
    target_link_libraries(poolkit_measure_m4 PRIVATE poolkit_cuda)
    poolkit_cuda_warnings(poolkit_measure_m4)

    # M6b: the GPU benchmark. Replays the same trace through poolkit, cudaMalloc
    # and cudaMallocAsync, with the five measurement rules of the plan baked into
    # the code (1 warmup + 5 runs, medians, config in the header). Like the M4
    # tool it is built but not registered as a test: it needs a quiet card.
    add_executable(poolkit_bench tools/bench.cu)
    target_link_libraries(poolkit_bench PRIVATE poolkit_cuda)
    target_include_directories(poolkit_bench PRIVATE ${PROJECT_SOURCE_DIR}/tools)
    poolkit_cuda_warnings(poolkit_bench)

    # M6b2: the example the README shows. Built in the device configuration and
    # run by ctest, so a signature change that breaks the documented usage breaks
    # the build (and not a reader's afternoon). No GTest, no clock, no assertions
    # beyond "it ran": the API's correctness is test_device's job.
    add_executable(poolkit_example_basic_usage examples/basic_usage.cu)
    target_link_libraries(poolkit_example_basic_usage PRIVATE poolkit_cuda)
    poolkit_cuda_warnings(poolkit_example_basic_usage)
    add_test(NAME poolkit_example_basic_usage COMMAND poolkit_example_basic_usage)
    set_tests_properties(poolkit_example_basic_usage PROPERTIES
        SKIP_RETURN_CODE 77
        PASS_REGULAR_EXPRESSION "release_all returned")

    # M6b: the CUPTI recorder. CUPTI instruments the calling process, so the
    # workload is a shared library loaded in-process; record_target.cu is built
    # twice (executable to prove it is a normal CUDA program, library for the
    # recorder) and record_check.cpp validates the produced trace with the real
    # parser and the real replay.
    if(TARGET CUDA::cupti)
        # cupti64_*.dll lives in extras/CUPTI/lib64, which is not on PATH by
        # default. Copying it next to a tool makes it runnable out of the build
        # tree without the caller editing PATH; on Linux the imported target's
        # directory is already in the build RPATH.
        # $<TARGET_FILE:CUDA::cupti> is cupti.lib (an import library), not the
        # runtime DLL, so the DLL is looked up next to it at configure time.
        #
        # This matters twice for the injection path: poolkit_inject.dll is loaded
        # by the CUDA driver inside the target process, whose DLL search starts in
        # the *executable's* directory. That directory is this one, so the CUPTI
        # runtime has to be here for the injected recording to start.
        function(poolkit_copy_cupti_runtime target)
            if(WIN32)
                get_target_property(POOLKIT_CUPTI_IMPLIB CUDA::cupti IMPORTED_IMPLIB)
                if(NOT POOLKIT_CUPTI_IMPLIB)
                    get_target_property(POOLKIT_CUPTI_IMPLIB CUDA::cupti IMPORTED_LOCATION)
                endif()
                if(POOLKIT_CUPTI_IMPLIB)
                    get_filename_component(POOLKIT_CUPTI_LIB_DIR "${POOLKIT_CUPTI_IMPLIB}" DIRECTORY)
                    file(GLOB POOLKIT_CUPTI_DLLS "${POOLKIT_CUPTI_LIB_DIR}/cupti64_*.dll")
                    foreach(POOLKIT_CUPTI_DLL IN LISTS POOLKIT_CUPTI_DLLS)
                        add_custom_command(TARGET ${target} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                "${POOLKIT_CUPTI_DLL}" $<TARGET_FILE_DIR:${target}>)
                    endforeach()
                endif()
            endif()
        endfunction()

        add_executable(poolkit_record tools/record.cu)
        target_link_libraries(poolkit_record PRIVATE CUDA::cupti CUDA::cudart)
        poolkit_cuda_warnings(poolkit_record)
        poolkit_copy_cupti_runtime(poolkit_record)

        add_executable(poolkit_record_target tests/device/record_target.cu)
        target_compile_definitions(poolkit_record_target PRIVATE
            POOLKIT_RECORD_TARGET_STANDALONE)
        target_link_libraries(poolkit_record_target PRIVATE CUDA::cudart)
        poolkit_cuda_warnings(poolkit_record_target)

        add_library(poolkit_record_target_workload SHARED tests/device/record_target.cu)
        target_compile_definitions(poolkit_record_target_workload PRIVATE
            POOLKIT_RECORD_TARGET_SHARED)
        target_link_libraries(poolkit_record_target_workload PRIVATE CUDA::cudart)
        poolkit_cuda_warnings(poolkit_record_target_workload)

        add_executable(poolkit_record_check tests/device/record_check.cpp)
        target_link_libraries(poolkit_record_check PRIVATE poolkit::poolkit)
        poolkit_warnings(poolkit_record_check)

        # --- M6b2: the injection shim ---------------------------------------
        # CUDA_INJECTION64_PATH: the driver loads this library into a *target*
        # process and calls InitializeInjection, so an unmodified CUDA program can
        # be recorded (tools/inject.cu, docs/record.md §9). It shares the recording
        # core with poolkit_record (tools/record_core.hpp), so the two cannot
        # produce different trace dialects.
        #
        # Built on Windows only: that is the platform where it was implemented and
        # measured. The Linux .so needs the same code (the injection entry point
        # and CUDA_INJECTION64_PATH are documented by CUPTI for both) but nobody
        # has run it, and shipping a build target nobody exercised is how a release
        # gets its first bug report. POOLKIT_BUILD_INJECT=ON builds it elsewhere
        # for whoever wants to be the first.
        option(POOLKIT_BUILD_INJECT "Build the CUDA_INJECTION64_PATH shim (Windows, M6b2)" ON)
        if(POOLKIT_BUILD_INJECT AND WIN32)
            add_library(poolkit_inject SHARED tools/inject.cu)
            # No CUDA::cudart on purpose: the shim must not initialise the runtime
            # before the application does, and it never calls the runtime API. Only
            # CUPTI, which the driver tolerates being used from its own init path.
            target_link_libraries(poolkit_inject PRIVATE CUDA::cupti)
            poolkit_cuda_warnings(poolkit_inject)
            poolkit_copy_cupti_runtime(poolkit_inject)
        elseif(NOT WIN32)
            message(STATUS "poolkit: poolkit_inject non compilato (solo Windows, M6b2)")
        endif()
    else()
        if(POOLKIT_REQUIRE_CUPTI)
            message(FATAL_ERROR
                "POOLKIT_REQUIRE_CUPTI=ON but the toolkit has no CUDA::cupti target. "
                "CMake >= 3.25 is needed for the imported CUPTI target; this is "
                "${CMAKE_VERSION}. The recorder and the injection shim would be skipped "
                "silently otherwise.")
        endif()
        message(STATUS "poolkit: CUPTI not found in the toolkit; poolkit_record non compilato")
    endif()
endif()

# ---------------------------------------------------------------------------
# CUPTI spike (M5): not a library feature, a two-hour experiment with an exit
# code. CUDA::cupti comes from FindCUDAToolkit; the import target is only
# created when the toolkit ships CUPTI (it does on 12.6, extras/CUPTI).
# ---------------------------------------------------------------------------
if(POOLKIT_BUILD_SPIKE)
    if(NOT POOLKIT_WITH_CUDA)
        message(FATAL_ERROR "POOLKIT_BUILD_SPIKE=ON requires POOLKIT_WITH_CUDA=ON")
    endif()
    if(NOT TARGET CUDA::cupti)
        message(FATAL_ERROR "POOLKIT_BUILD_SPIKE=ON but the toolkit has no CUPTI (CUDA::cupti)")
    endif()
    add_executable(poolkit_spike_cupti_memory tools/spike/cupti_memory_spike.cu)
    target_link_libraries(poolkit_spike_cupti_memory PRIVATE CUDA::cupti CUDA::cudart)
    poolkit_cuda_warnings(poolkit_spike_cupti_memory)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
include(CTest)
if(POOLKIT_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    # Pinned to a stable tag: a moving branch would make the suite's verdict
    # depend on somebody else's commit.
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        DOWNLOAD_EXTRACT_TIMESTAMP OFF
    )
    # Keep `cmake --install` of poolkit clean and skip gmock (unused).
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    function(poolkit_add_test name)
        add_executable(${name} ${ARGN} ${PROJECT_SOURCE_DIR}/tests/test_main.cpp)
        target_link_libraries(${name} PRIVATE poolkit::poolkit GTest::gtest)
        poolkit_warnings(${name})
        add_test(NAME ${name} COMMAND ${name})
        # Exit code 77 means "this environment cannot run the test" (device
        # suites in M4+). Kept on every test so a test can skip itself without
        # anyone editing the CMake later.
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    poolkit_add_test(poolkit_test_size_classes tests/test_size_classes.cpp)
    poolkit_add_test(poolkit_test_pool_host tests/test_pool_host.cpp)
    poolkit_add_test(poolkit_test_metrics tests/test_metrics.cpp)
    poolkit_add_test(poolkit_test_trace tests/test_trace.cpp)
    poolkit_add_test(poolkit_test_stress tests/test_stress.cpp)
    # The trace test reads the committed trace relative to the source tree. The
    # path is normalised to forward slashes so the string literal carries no
    # backslash escapes (which /W4 /WX would flag on MSVC).
    file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" POOLKIT_SOURCE_DIR_FORWARD)
    target_compile_definitions(poolkit_test_trace PRIVATE
        POOLKIT_SOURCE_DIR="${POOLKIT_SOURCE_DIR_FORWARD}")

    # M6a: the tools' logic lives in tools/*.hpp precisely so it can be tested
    # here, by name, with the real exit codes — not only by grepping stdout of a
    # binary from a CMake script.
    poolkit_add_test(poolkit_test_tools tests/test_tools.cpp)
    target_include_directories(poolkit_test_tools PRIVATE ${PROJECT_SOURCE_DIR}/tools)
    target_compile_definitions(poolkit_test_tools PRIVATE
        POOLKIT_SOURCE_DIR="${POOLKIT_SOURCE_DIR_FORWARD}")

    # Out-of-tree consumer: installs into a scratch prefix and builds a separate
    # project against the exported package. This is the only test that proves
    # `find_package(poolkit CONFIG)` actually works on an installed tree.
    # The consumer is configured and built in a child process, so it needs a
    # C++ compiler reachable from the environment this ctest runs in (on Windows:
    # a Developer Prompt / vcvars64 shell, as in CI). Without one the test is not
    # registered at all — a disabled test is visible at configure time, a silently
    # green one is not.
    find_program(POOLKIT_HOST_CXX NAMES cl.exe g++ clang++ HINTS ENV PATH)
    # MSVC cannot link a console program without the resource compiler, and on a
    # machine where the Windows SDK is only installed for one Visual Studio
    # instance `cl` is on PATH while `rc` is not. Requiring both keeps this test
    # from failing for a reason that has nothing to do with the package.
    if(WIN32)
        find_program(POOLKIT_HOST_RC NAMES rc.exe rc HINTS ENV PATH)
    else()
        set(POOLKIT_HOST_RC "not-needed")
    endif()
    if(POOLKIT_HOST_CXX AND POOLKIT_HOST_RC)
        add_test(NAME poolkit_consumer_install
            COMMAND ${CMAKE_COMMAND}
                -DPOOLKIT_SOURCE_DIR=${PROJECT_SOURCE_DIR}
                -DPOOLKIT_BUILD_DIR=${PROJECT_BINARY_DIR}
                -DWORK_DIR=${PROJECT_BINARY_DIR}/consumer-check
                "-DGENERATOR=${CMAKE_GENERATOR}"
                "-DGENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
                "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DBUILD_TYPE=$<IF:$<BOOL:${CMAKE_BUILD_TYPE}>,${CMAKE_BUILD_TYPE},Release>"
                -P ${PROJECT_SOURCE_DIR}/tests/consumer/run_install_consumer.cmake
        )
        set_tests_properties(poolkit_consumer_install PROPERTIES TIMEOUT 600)
    else()
        message(STATUS "poolkit: poolkit_consumer_install non registrato "
                       "(nessun compilatore C++ nel PATH: eseguire da Developer Prompt)")
    endif()

    # --- device suite (only with POOLKIT_WITH_CUDA=ON) ----------------------
    # test_device.cpp / test_main_device.cpp are plain C++ (GoogleTest never
    # sees nvcc); device_kernels.cu is the only nvcc-compiled file and carries
    # the spin kernel the deferred-reuse test needs. POOLKIT_USE_CUDA_RUNTIME
    # flips StreamHandle to cudaStream_t in the core headers for the C++
    # translation units; the .cu gets the same from __CUDACC__.
    if(POOLKIT_WITH_CUDA)
        add_executable(poolkit_test_device
            tests/test_device.cpp
            tests/test_main_device.cpp
            tests/device_kernels.cu
        )
        target_link_libraries(poolkit_test_device PRIVATE poolkit_cuda GTest::gtest)
        target_compile_definitions(poolkit_test_device PRIVATE POOLKIT_USE_CUDA_RUNTIME)
        poolkit_warnings(poolkit_test_device)
        poolkit_cuda_warnings(poolkit_test_device)
        add_test(NAME poolkit_test_device COMMAND poolkit_test_device)
        # 77 = "no device here": SKIPPED, neither red nor a false green.
        set_tests_properties(poolkit_test_device PROPERTIES SKIP_RETURN_CODE 77)

        # --- M6b: the CUPTI recorder, end to end ----------------------------
        # Test 1 produces the trace from the in-process workload; test 2 checks
        # it with the real parser and the real replay (sizes, no orphan frees,
        # peak > 0). The fixture makes 2 depend on 1, so a machine without a
        # device skips both (77) instead of failing on a missing file.
        if(TARGET poolkit_record)
            set(POOLKIT_RECORD_TRACE ${PROJECT_BINARY_DIR}/record_target.jsonl)
            add_test(NAME poolkit_record_target_record
                COMMAND poolkit_record -o ${POOLKIT_RECORD_TRACE}
                        --workload $<TARGET_FILE:poolkit_record_target_workload>)
            set_tests_properties(poolkit_record_target_record PROPERTIES
                SKIP_RETURN_CODE 77
                FIXTURES_SETUP poolkit_record_trace
                PASS_REGULAR_EXPRESSION "wrote .*record_target\\.jsonl")
            add_test(NAME poolkit_record_trace_replays
                COMMAND poolkit_record_check ${POOLKIT_RECORD_TRACE})
            set_tests_properties(poolkit_record_trace_replays PROPERTIES
                SKIP_RETURN_CODE 77
                FIXTURES_REQUIRED poolkit_record_trace
                PASS_REGULAR_EXPRESSION "record_check: OK")

            # --- M6b2: the same program, unmodified, through the shim --------
            # record_target.exe is a normal CUDA program (it is the same file that
            # is also built as a workload library); nothing in this test tells it
            # that it is being recorded. The only two things that differ from a
            # plain run are the environment variables, which is the entire point of
            # CUDA_INJECTION64_PATH: no recompile, no wrapper, no --workload.
            #
            # The check is the relaxed one (--injected): an injected recorder
            # subscribes before the context exists, so the driver's own startup
            # allocations can legitimately be in the file. See record_check.cpp.
            if(TARGET poolkit_inject)
                set(POOLKIT_INJECT_TRACE ${PROJECT_BINARY_DIR}/inject_target.jsonl)
                add_test(NAME poolkit_inject_record_target
                    COMMAND ${CMAKE_COMMAND} -E env
                        "CUDA_INJECTION64_PATH=$<TARGET_FILE:poolkit_inject>"
                        "POOLKIT_RECORD_OUT=${POOLKIT_INJECT_TRACE}"
                        $<TARGET_FILE:poolkit_record_target> 400)
                set_tests_properties(poolkit_inject_record_target PROPERTIES
                    SKIP_RETURN_CODE 77
                    FIXTURES_SETUP poolkit_inject_trace
                    PASS_REGULAR_EXPRESSION "poolkit_inject: wrote .*inject_target\\.jsonl")
                add_test(NAME poolkit_inject_trace_replays
                    COMMAND poolkit_record_check ${POOLKIT_INJECT_TRACE} --injected)
                set_tests_properties(poolkit_inject_trace_replays PROPERTIES
                    SKIP_RETURN_CODE 77
                    FIXTURES_REQUIRED poolkit_inject_trace
                    PASS_REGULAR_EXPRESSION "record_check: OK")
            endif()
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Install + export for find_package(poolkit CONFIG)
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/poolkit
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.cuh"
)

install(TARGETS poolkit EXPORT poolkitTargets)

install(EXPORT poolkitTargets
    FILE poolkitTargets.cmake
    NAMESPACE poolkit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poolkit
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/poolkit-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/poolkit-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poolkit
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/poolkit-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/poolkit-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/poolkit-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/poolkit
)

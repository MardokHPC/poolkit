# Installs poolkit into a scratch prefix and builds tests/consumer against the
# exported package, then runs it. Run in script mode (-P) from ctest, or by hand:
#
#   cmake -DPOOLKIT_SOURCE_DIR=. -DPOOLKIT_BUILD_DIR=build -DWORK_DIR=build/consumer-check \
#         -DGENERATOR=Ninja -DBUILD_TYPE=Release -P tests/consumer/run_install_consumer.cmake
#
# The point is the negative proof: the consumer has no include path back into the
# source tree, so a broken install/export fails here and nowhere else.

if(NOT POOLKIT_SOURCE_DIR OR NOT POOLKIT_BUILD_DIR OR NOT WORK_DIR)
    message(FATAL_ERROR "poolkit: POOLKIT_SOURCE_DIR, POOLKIT_BUILD_DIR and WORK_DIR are required")
endif()
if(NOT BUILD_TYPE)
    set(BUILD_TYPE Release)
endif()

# Preflight: ctest can be run from a shell that never called vcvars64.bat, and
# then the child configure fails with a wall of CMake text saying the compiler is
# "broken". It is not: on Windows MSVC cannot link a console program without
# rc.exe, and outside a Developer Prompt that is what is missing. Say it in one
# line. The test still fails — an unusable toolchain must not turn the suite
# green — but it fails for the reason that is actually true.
if(WIN32)
    find_program(POOLKIT_PREFLIGHT_RC NAMES rc.exe rc HINTS ENV PATH)
    if(NOT POOLKIT_PREFLIGHT_RC)
        message(FATAL_ERROR "poolkit: no rc.exe; run this from a Developer Prompt or use build-host.bat")
    endif()
endif()
if(CXX_COMPILER)
    if(NOT EXISTS "${CXX_COMPILER}")
        message(FATAL_ERROR "poolkit: no compiler at '${CXX_COMPILER}'; run this from a Developer Prompt or use build-host.bat")
    endif()
else()
    find_program(POOLKIT_PREFLIGHT_CXX NAMES cl.exe g++ clang++ HINTS ENV PATH)
    if(NOT POOLKIT_PREFLIGHT_CXX)
        message(FATAL_ERROR "poolkit: no C++ compiler; run this from a Developer Prompt or use build-host.bat")
    endif()
endif()

set(prefix "${WORK_DIR}/prefix")
set(consumer_build "${WORK_DIR}/consumer-build")

file(REMOVE_RECURSE "${WORK_DIR}")

# Runs a command, keeps stdout/stderr in the CTest log, fails loudly.
function(poolkit_run)
    execute_process(
        COMMAND ${ARGV}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "poolkit: command failed (${result})\n--- stdout ---\n${out}\n--- stderr ---\n${err}")
    endif()
    if(out)
        message(STATUS "${out}")
    endif()
    if(err)
        message(STATUS "${err}")
    endif()
endfunction()

# 1. Install: header-only, so this needs no compiled artifacts.
poolkit_run("${CMAKE_COMMAND}" --install "${POOLKIT_BUILD_DIR}" --prefix "${prefix}" --config "${BUILD_TYPE}")

if(NOT EXISTS "${prefix}/include/poolkit/size_classes.hpp")
    message(FATAL_ERROR "poolkit: size_classes.hpp missing from ${prefix}/include/poolkit")
endif()

# 2. Configure the consumer with nothing but the prefix.
set(configure_args -S "${POOLKIT_SOURCE_DIR}/tests/consumer" -B "${consumer_build}"
    "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
# Reuse the compiler of the parent build: the child has to pick the same toolchain
# even when several are installed, and the path is exact rather than guessed.
if(CXX_COMPILER)
    list(APPEND configure_args "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
endif()
if(GENERATOR)
    list(APPEND configure_args -G "${GENERATOR}")
endif()
# Same architecture as the outer build (multi-config generators only).
if(GENERATOR_PLATFORM)
    list(APPEND configure_args -A "${GENERATOR_PLATFORM}")
endif()
poolkit_run("${CMAKE_COMMAND}" ${configure_args})

# 3. Build and run. Executable location depends on the generator, so try both.
poolkit_run("${CMAKE_COMMAND}" --build "${consumer_build}" --config "${BUILD_TYPE}")

set(candidates
    "${consumer_build}/poolkit_consumer"
    "${consumer_build}/poolkit_consumer.exe"
    "${consumer_build}/${BUILD_TYPE}/poolkit_consumer"
    "${consumer_build}/${BUILD_TYPE}/poolkit_consumer.exe"
)
set(consumer_exe "")
foreach(candidate IN LISTS candidates)
    if(EXISTS "${candidate}")
        set(consumer_exe "${candidate}")
        break()
    endif()
endforeach()
if(NOT consumer_exe)
    message(FATAL_ERROR "poolkit: consumer executable not found under ${consumer_build}")
endif()

poolkit_run("${consumer_exe}")

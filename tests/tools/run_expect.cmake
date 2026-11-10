# Runs one poolkit tool and asserts its exit code.
#
# The four gate verdicts are a contract on the exit code (0 pass/improvement,
# 1 regression, 2 suspect/error). A CTest that only greps stdout would still pass
# if a wrapper forgot to return the code, so this script checks the number.
#
# Usage:
#   cmake -DTOOL=<path> -DEXPECTED=<int> "-DARGS=<arg;arg;...>" [-DEXPECT_STDOUT=<regex>] -P run_expect.cmake

if(NOT DEFINED TOOL OR NOT DEFINED EXPECTED)
    message(FATAL_ERROR "run_expect.cmake: TOOL and EXPECTED are required")
endif()

execute_process(
    COMMAND ${TOOL} ${ARGS}
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    RESULT_VARIABLE code
)

if(NOT code EQUAL EXPECTED)
    message(FATAL_ERROR
        "expected exit ${EXPECTED}, got ${code}\n--- stdout ---\n${out}\n--- stderr ---\n${err}")
endif()

if(DEFINED EXPECT_STDOUT AND NOT out MATCHES "${EXPECT_STDOUT}")
    message(FATAL_ERROR
        "stdout did not match '${EXPECT_STDOUT}'\n--- stdout ---\n${out}")
endif()

message(STATUS "exit ${code} as expected")

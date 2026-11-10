#pragma once

#include <stdexcept>
#include <string>

#include <poolkit/config.hpp>

// CUDA headers are optional: a hard include would make the library uncompilable without the toolkit
// and kill the host-only test suite.
#if POOLKIT_HAS_CUDA_RUNTIME
#include <cuda_runtime.h>
#endif

namespace poolkit {

// cudaError_t is an enum whose underlying type is int, so the ABI and the values match and code()
// stays comparable with cudaSuccess after a cast.
#if POOLKIT_HAS_CUDA_RUNTIME
using error_code = cudaError_t;
#else
using error_code = int;
#endif

POOLKIT_NODISCARD constexpr error_code success_code() noexcept {
    return static_cast<error_code>(0);
}

// No retry policy and no nullptr return: a caller that wants to react to OOM writes a catch block it
// can see in the source.
class PoolError : public std::runtime_error {
public:
    explicit PoolError(const std::string& what_arg, error_code code = success_code())
        : std::runtime_error(what_arg), code_(code) {}

    POOLKIT_NODISCARD error_code code() const noexcept { return code_; }

    POOLKIT_NODISCARD bool has_code() const noexcept { return code_ != success_code(); }

private:
    error_code code_;
};

}  // namespace poolkit

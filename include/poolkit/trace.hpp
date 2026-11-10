#pragma once

// Trace reader and deterministic replay. JSONL, one operation per line (docs/trace-format.md):
// {"op":"alloc","size":4096,"ptr":3,"stream":0,"ordinal":7}. No CUDA header here, so parsing and
// replay run on a CPU — which is what makes "replay twice, get the same counters" a real test.

#include <poolkit/pool.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace poolkit {

// Malformed JSON, a missing field, a bad op, a free of an inactive ptr, or a non-increasing ordinal.
class TraceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class TraceOpKind : std::uint8_t { alloc, free };

struct TraceOp {
    TraceOpKind kind = TraceOpKind::alloc;
    std::size_t size = 0;      // bytes requested; 0 on a free
    std::uint64_t ptr = 0;     // trace-side block id
    std::uint64_t stream = 0;  // logical stream id
    std::uint64_t ordinal = 0; // global order
};

struct Trace {
    std::vector<TraceOp> ops;
};

namespace trace_detail {

inline void skip_ws(const std::string& s, std::size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) {
        ++i;
    }
}

[[noreturn]] inline void fail(std::size_t line, const std::string& message) {
    throw TraceError("poolkit: trace line " + std::to_string(line) + ": " + message);
}

inline int hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

inline void append_utf8(std::string& out, std::uint32_t code_point) {
    if (code_point <= 0x7F) {
        out += static_cast<char>(code_point);
    } else if (code_point <= 0x7FF) {
        out += static_cast<char>(0xC0 | (code_point >> 6));
        out += static_cast<char>(0x80 | (code_point & 0x3F));
    } else {
        out += static_cast<char>(0xE0 | (code_point >> 12));
        out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (code_point & 0x3F));
    }
}

inline std::string parse_string(const std::string& s, std::size_t& i, std::size_t line) {
    if (i >= s.size() || s[i] != '"') {
        fail(line, "expected a string");
    }
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') {
            ++i;
            if (i >= s.size()) {
                fail(line, "unterminated escape");
            }
            switch (s[i]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (i + 4 >= s.size()) {
                        fail(line, "truncated \\u escape");
                    }
                    std::uint32_t cp = 0;
                    for (int k = 1; k <= 4; ++k) {
                        const int digit = hex_value(s[i + k]);
                        if (digit < 0) {
                            fail(line, "invalid \\u escape");
                        }
                        cp = (cp << 4) | static_cast<std::uint32_t>(digit);
                    }
                    i += 4;
                    append_utf8(out, cp);
                    break;
                }
                default:
                    fail(line, std::string("invalid escape '\\") + s[i] + "'");
            }
            ++i;
        } else {
            out += s[i];
            ++i;
        }
    }
    if (i >= s.size()) {
        fail(line, "unterminated string");
    }
    ++i;  // closing quote
    return out;
}

inline std::uint64_t parse_uint(const std::string& s, std::size_t& i, std::size_t line) {
    if (i >= s.size() || s[i] < '0' || s[i] > '9') {
        fail(line, "expected a non-negative integer");
    }
    // Checked accumulation: `value * 10 + digit` wraps silently, and "size": 18446744073709551620
    // used to become 4 and be accepted. The bound is the exact condition for the next step to fit.
    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t value = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        const std::uint64_t digit = static_cast<std::uint64_t>(s[i] - '0');
        if (value > (kMax - digit) / 10) {
            fail(line, "integer is too large for a 64-bit field");
        }
        value = value * 10 + digit;
        ++i;
    }
    return value;
}

inline TraceOp parse_line(const std::string& line, std::size_t number) {
    std::size_t i = 0;
    skip_ws(line, i);
    if (i >= line.size() || line[i] != '{') {
        fail(number, "expected '{'");
    }
    ++i;
    skip_ws(line, i);

    bool has_op = false;
    bool has_size = false;
    bool has_ptr = false;
    bool has_stream = false;
    bool has_ordinal = false;
    std::string op_text;
    TraceOp op;

    for (;;) {
        skip_ws(line, i);
        if (i >= line.size() || line[i] != '"') {
            fail(number, "expected a string key");
        }
        const std::string key = parse_string(line, i, number);
        skip_ws(line, i);
        if (i >= line.size() || line[i] != ':') {
            fail(number, "expected ':'");
        }
        ++i;
        skip_ws(line, i);

        if (i < line.size() && line[i] == '"') {
            const std::string value = parse_string(line, i, number);
            if (key == "op") {
                op_text = value;
                has_op = true;
            } else {
                fail(number, "unexpected string field '" + key + "'");
            }
        } else {
            const std::uint64_t value = parse_uint(line, i, number);
            if (key == "size") {
                op.size = static_cast<std::size_t>(value);
                has_size = true;
            } else if (key == "ptr") {
                op.ptr = value;
                has_ptr = true;
            } else if (key == "stream") {
                op.stream = value;
                has_stream = true;
            } else if (key == "ordinal") {
                op.ordinal = value;
                has_ordinal = true;
            } else {
                fail(number, "unknown field '" + key + "'");
            }
        }

        skip_ws(line, i);
        if (i < line.size() && line[i] == ',') {
            ++i;
            continue;
        }
        if (i < line.size() && line[i] == '}') {
            ++i;
            break;
        }
        fail(number, "expected ',' or '}'");
    }

    skip_ws(line, i);
    if (i != line.size()) {
        fail(number, "trailing characters after '}'");
    }

    if (!has_op) {
        fail(number, "missing field 'op'");
    }
    if (!has_size) {
        fail(number, "missing field 'size'");
    }
    if (!has_ptr) {
        fail(number, "missing field 'ptr'");
    }
    if (!has_stream) {
        fail(number, "missing field 'stream'");
    }
    if (!has_ordinal) {
        fail(number, "missing field 'ordinal'");
    }

    if (op_text == "alloc") {
        op.kind = TraceOpKind::alloc;
    } else if (op_text == "free") {
        op.kind = TraceOpKind::free;
    } else {
        fail(number, "op must be \"alloc\" or \"free\", got '" + op_text + "'");
    }
    if (op.kind == TraceOpKind::alloc && op.size == 0) {
        fail(number, "an alloc with size 0 has no size class");
    }
    return op;
}

}  // namespace trace_detail

inline Trace parse_trace(std::istream& in) {
    Trace trace;
    std::string line;
    std::size_t number = 0;
    bool first = true;
    std::uint64_t previous = 0;
    while (std::getline(in, line)) {
        ++number;
        if (line.empty()) {
            continue;
        }
        const TraceOp op = trace_detail::parse_line(line, number);
        if (!first && op.ordinal <= previous) {
            trace_detail::fail(number, "ordinal " + std::to_string(op.ordinal) +
                                           " is not greater than the previous " +
                                           std::to_string(previous));
        }
        trace.ops.push_back(op);
        previous = op.ordinal;
        first = false;
    }
    if (trace.ops.empty()) {
        throw TraceError("poolkit: empty trace");
    }
    return trace;
}

inline Trace parse_trace(const std::string& text) {
    std::istringstream in(text);
    return parse_trace(in);
}

inline Trace parse_trace_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw TraceError("poolkit: cannot open trace file '" + path + "'");
    }
    return parse_trace(in);
}

struct TraceReplayResult {
    std::size_t alloc_count = 0;
    std::size_t free_count = 0;
    PoolSnapshot snapshot;
    // The large path is not in bytes_resident (it never enters the pool) but it *is* memory on the
    // card, so the peak adds it; on a trace that never leaves the pool this is the high-water mark.
    std::size_t peak_bytes = 0;
    std::size_t peak_live_blocks = 0;
};

inline StreamHandle trace_stream(std::uint64_t id) noexcept {
    return reinterpret_cast<StreamHandle>(static_cast<std::uintptr_t>(id));
}

// Throws on a free of an inactive ptr or an alloc that reuses a live one. Bit-deterministic: every
// branch depends only on the ops read and on the pool's own deterministic state.
inline TraceReplayResult replay_trace(Pool& pool, const Trace& trace) {
    std::unordered_map<std::uint64_t, DevicePtr> live;
    std::unordered_map<std::uint64_t, std::size_t> live_large;
    std::size_t large_live_bytes = 0;
    const std::size_t max_class_bytes = pool.config().max_class_bytes;
    TraceReplayResult result;

    const auto sample_peak = [&]() noexcept {
        const std::size_t current = pool.capacity() + large_live_bytes;
        if (current > result.peak_bytes) {
            result.peak_bytes = current;
        }
        if (live.size() > result.peak_live_blocks) {
            result.peak_live_blocks = live.size();
        }
    };

    for (const TraceOp& op : trace.ops) {
        if (op.kind == TraceOpKind::alloc) {
            DevicePtr ptr = pool.allocate(op.size, trace_stream(op.stream));
            const auto [it, inserted] = live.emplace(op.ptr, std::move(ptr));
            if (!inserted) {
                throw TraceError("poolkit: alloc reuses ptr " + std::to_string(op.ptr) +
                                 " at ordinal " + std::to_string(op.ordinal));
            }
            if (op.size > max_class_bytes) {
                live_large.emplace(op.ptr, op.size);
                large_live_bytes += op.size;
            }
            ++result.alloc_count;
        } else {
            const auto it = live.find(op.ptr);
            if (it == live.end()) {
                throw TraceError("poolkit: free of inactive ptr " + std::to_string(op.ptr) +
                                 " at ordinal " + std::to_string(op.ordinal));
            }
            pool.free(std::move(it->second), trace_stream(op.stream));
            live.erase(it);
            const auto large = live_large.find(op.ptr);
            if (large != live_large.end()) {
                large_live_bytes -= large->second;
                live_large.erase(large);
            }
            ++result.free_count;
        }
        sample_peak();
    }

    result.snapshot = pool.snapshot();
    return result;
}

}  // namespace poolkit

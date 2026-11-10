#pragma once

// The report document shared by poolkit_advise, poolkit_gate and poolkit_bench. It is a
// contract: versioned, written and read by the same code, and always serialised with the
// classic locale, so a host decimal comma cannot turn 1.5 into a pass.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poolkit {
namespace tools {

// Bump only with a reader that understands both versions; poolkit_gate answers "suspect",
// never "pass", to a schema it does not know. 1 is the analysis report; 2 adds a required "bench" section.
inline constexpr int kReportSchema = 1;
inline constexpr int kBenchReportSchema = 2;
inline constexpr int kBenchSectionSchema = 1;

class ReportError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A path is not an identity: the same trace at two paths must compare equal, and `ops` alone
// would miss a size change, which is exactly the kind of change that moves the peak.
class Sha256 {
public:
    Sha256() { reset(); }

    void update(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        total_bytes_ += static_cast<std::uint64_t>(size);
        for (std::size_t i = 0; i < size; ++i) {
            buffer_[buffer_length_++] = bytes[i];
            if (buffer_length_ == 64) {
                compress(buffer_);
                buffer_length_ = 0;
            }
        }
    }

    // Destructive on purpose: a digest that can be taken twice invites updating after reading it.
    std::string hex_digest() {
        const std::uint64_t bit_length = total_bytes_ * 8u;
        const unsigned char pad_start = 0x80;
        update_raw(&pad_start, 1);
        const unsigned char zero = 0;
        while (buffer_length_ != 56) {
            update_raw(&zero, 1);
        }
        unsigned char length_bytes[8];
        for (int i = 0; i < 8; ++i) {
            length_bytes[i] = static_cast<unsigned char>((bit_length >> (56 - 8 * i)) & 0xFFu);
        }
        update_raw(length_bytes, 8);
        if (buffer_length_ != 0) {
            throw ReportError("poolkit: internal error: SHA-256 buffer not aligned");
        }

        static const char* hex = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (std::uint32_t word : state_) {
            for (int shift = 28; shift >= 0; shift -= 4) {
                out += hex[(word >> shift) & 0xFu];
            }
        }
        return out;
    }

private:
    static std::uint32_t rotr(std::uint32_t value, int bits) noexcept {
        return (value >> bits) | (value << (32 - bits));
    }

    void reset() {
        state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                  0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        total_bytes_ = 0;
        buffer_length_ = 0;
    }

    void update_raw(const unsigned char* data, std::size_t size) {
        for (std::size_t i = 0; i < size; ++i) {
            buffer_[buffer_length_++] = data[i];
            if (buffer_length_ == 64) {
                compress(buffer_);
                buffer_length_ = 0;
            }
        }
    }

    void compress(const unsigned char* block) {
        static const std::uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
            0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
            0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
            0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
            0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

        std::uint32_t w[64] = {};
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<std::uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 =
                rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 =
                rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];

        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t temp1 = h + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::vector<std::uint32_t> state_;
    std::uint64_t total_bytes_ = 0;
    unsigned char buffer_[64] = {};
    std::size_t buffer_length_ = 0;
};

inline std::string sha256_hex(const std::string& bytes) {
    Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.hex_digest();
}

// Streamed in 64 KiB blocks: a trace that describes 8 GiB is not read into host memory to be identified.
inline std::string sha256_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw ReportError("poolkit: cannot open '" + path + "' to hash it");
    }
    Sha256 hash;
    std::vector<char> buffer(64 * 1024);
    for (;;) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = in.gcount();
        if (got > 0) {
            hash.update(buffer.data(), static_cast<std::size_t>(got));
        }
        if (!in) {
            break;
        }
    }
    return hash.hex_digest();
}

inline bool is_sha256_hex(const std::string& text) noexcept {
    if (text.size() != 64) {
        return false;
    }
    for (char c : text) {
        const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!hex) {
            return false;
        }
    }
    return true;
}

struct TraceIdentity {
    std::string path;   // the argument as given, not a canonical path
    std::size_t ops = 0;
    std::size_t allocs = 0;
    std::string sha256; // empty means "not computed"; the gate then falls back
};

struct ConfigIdentity {
    double growth = 1.5;
    std::size_t chunk_bytes = std::size_t{2} << 20;
    std::size_t min_class_bytes = 256;
    std::size_t max_class_bytes = std::size_t{2} << 20;
    bool power_of_two = true;
    std::vector<std::size_t> extra_sizes;
};

struct WasteFractions {
    double internal = 0.0;
    double tail = 0.0;
};

struct PeakBytes {
    std::size_t resident_bytes = 0;
    std::size_t blocks = 0;
};

// Measured, not modelled: every number comes from a clock read around a real operation.
// min <= median <= p99 <= max is enforced on read.

struct BenchDistribution {
    double median = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct BenchMode {
    std::string mode;  // "pool", "cudamalloc", "cudaasync"
    std::size_t ops = 0;
    std::size_t allocs = 0;
    std::size_t frees = 0;
    BenchDistribution wall_ns;
    std::size_t peak_bytes = 0;
    // true  -> a cudaMemGetInfo sample every sample_every_ops ops (a lower bound on the peak);
    // false -> the pool's exact high-water mark, sample_every_ops 0.
    bool peak_sampled = false;
    std::size_t sample_every_ops = 0;
    BenchDistribution alloc_ns;
    BenchDistribution free_ns;
    BenchDistribution hit_ns;
    BenchDistribution growth_ns;
    std::size_t hit_ops = 0;
    std::size_t growth_ops = 0;
    BenchDistribution slot_ns;   // new block, existing chunk: bookkeeping
    std::size_t slot_ops = 0;
    BenchDistribution chunk_ns;  // new block, fresh chunk: one cudaMalloc
    std::size_t chunk_ops = 0;
    BenchDistribution large_ns;  // above max_class: one cudaMalloc, never reused
    std::size_t large_ops = 0;
};

struct BenchEnvironment {
    std::string gpu;
    std::string compute;  // "sm_75"
    int driver_version = 0;   // cudaDriverGetVersion, e.g. 13030
    int runtime_version = 0;  // cudaRuntimeGetVersion, e.g. 12060
    std::string arch;         // "x86_64" / "aarch64"
    std::size_t chunk_bytes = 0;
    double growth = 0.0;
    std::size_t min_class_bytes = 0;
    std::size_t max_class_bytes = 0;
    int warmup = 0;
    int repeat = 0;
};

struct BenchSection {
    int schema = kBenchSectionSchema;
    BenchEnvironment env;
    std::vector<BenchMode> modes;
};

struct Report {
    int schema = kReportSchema;
    std::string tool = "poolkit_advise";
    TraceIdentity trace;
    ConfigIdentity config;
    std::size_t classes = 0;
    WasteFractions waste;
    PeakBytes peak;
    std::size_t retention_bound_bytes = 0;
    bool has_bench = false;
    BenchSection bench;
};

inline bool same_config(const ConfigIdentity& a, const ConfigIdentity& b) noexcept {
    return a.growth == b.growth && a.chunk_bytes == b.chunk_bytes &&
           a.min_class_bytes == b.min_class_bytes && a.max_class_bytes == b.max_class_bytes &&
           a.power_of_two == b.power_of_two && a.extra_sizes == b.extra_sizes;
}

inline std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char ch : text) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(c >> 4) & 0xFu];
                    out += hex[c & 0xFu];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Fixed 6 decimals with trailing zeros trimmed: "1.5", not "1.500000", so the document stays diffable.
inline std::string json_number(double value) {
    if (!std::isfinite(value)) {
        throw ReportError("poolkit: refusing to write a non-finite number to a report");
    }
    if (value == 0.0) {
        return "0";
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(6) << value;
    std::string text = out.str();
    if (text.find('.') != std::string::npos) {
        while (!text.empty() && text.back() == '0') {
            text.pop_back();
        }
        if (!text.empty() && text.back() == '.') {
            text.pop_back();
        }
    }
    return text;
}

inline void write_distribution(std::ostream& out, const BenchDistribution& d) {
    out << "{\"median\": " << json_number(d.median) << ", \"p99\": " << json_number(d.p99)
        << ", \"min\": " << json_number(d.min) << ", \"max\": " << json_number(d.max) << "}";
}

inline void write_bench_json(std::ostream& out, const BenchSection& bench) {
    out << "  \"bench\": {\n";
    out << "    \"schema\": " << bench.schema << ",\n";
    out << "    \"env\": {\"gpu\": \"" << json_escape(bench.env.gpu) << "\", \"compute\": \""
        << json_escape(bench.env.compute) << "\", \"driver\": " << bench.env.driver_version
        << ", \"runtime\": " << bench.env.runtime_version << ", \"arch\": \""
        << json_escape(bench.env.arch) << "\", \"chunk_bytes\": " << bench.env.chunk_bytes
        << ", \"growth\": " << json_number(bench.env.growth)
        << ", \"min_class_bytes\": " << bench.env.min_class_bytes
        << ", \"max_class_bytes\": " << bench.env.max_class_bytes
        << ", \"warmup\": " << bench.env.warmup << ", \"repeat\": " << bench.env.repeat
        << "},\n";
    out << "    \"modes\": [";
    for (std::size_t i = 0; i < bench.modes.size(); ++i) {
        const BenchMode& mode = bench.modes[i];
        out << (i == 0 ? "\n" : ",\n");
        out << "      {\"mode\": \"" << json_escape(mode.mode) << "\", \"ops\": " << mode.ops
            << ", \"allocs\": " << mode.allocs << ", \"frees\": " << mode.frees
            << ", \"peak_bytes\": " << mode.peak_bytes
            << ", \"peak_sampled\": " << (mode.peak_sampled ? "true" : "false")
            << ", \"sample_every_ops\": " << mode.sample_every_ops << ", \"wall_ns\": ";
        write_distribution(out, mode.wall_ns);
        out << ", \"alloc_ns\": ";
        write_distribution(out, mode.alloc_ns);
        out << ", \"free_ns\": ";
        write_distribution(out, mode.free_ns);
        if (mode.mode == "pool") {
            out << ", \"hit_ns\": ";
            write_distribution(out, mode.hit_ns);
            out << ", \"hit_ops\": " << mode.hit_ops;
            out << ", \"growth_ns\": ";
            write_distribution(out, mode.growth_ns);
            out << ", \"growth_ops\": " << mode.growth_ops;
            out << ", \"slot_ns\": ";
            write_distribution(out, mode.slot_ns);
            out << ", \"slot_ops\": " << mode.slot_ops;
            out << ", \"chunk_ns\": ";
            write_distribution(out, mode.chunk_ns);
            out << ", \"chunk_ops\": " << mode.chunk_ops;
            out << ", \"large_ns\": ";
            write_distribution(out, mode.large_ns);
            out << ", \"large_ops\": " << mode.large_ops;
        }
        out << "}";
    }
    out << "\n    ]\n";
    out << "  }\n";
}

inline std::string to_json(const Report& report) {
    // Never through the global locale: a stream that is not imbued picks up LC_NUMERIC.
    if (report.has_bench && report.schema != kBenchReportSchema) {
        throw ReportError("poolkit: a bench section requires schema " +
                          std::to_string(kBenchReportSchema) + ", report says " +
                          std::to_string(report.schema));
    }
    if (!report.has_bench && report.schema != kReportSchema) {
        throw ReportError("poolkit: schema " + std::to_string(report.schema) +
                          " requires a bench section; this report has none");
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\n";
    out << "  \"schema\": " << report.schema << ",\n";
    out << "  \"tool\": \"" << json_escape(report.tool) << "\",\n";
    out << "  \"trace\": {\"path\": \"" << json_escape(report.trace.path)
        << "\", \"ops\": " << report.trace.ops << ", \"allocs\": " << report.trace.allocs
        << ", \"sha256\": \"" << json_escape(report.trace.sha256) << "\"},\n";
    out << "  \"config\": {\"growth\": " << json_number(report.config.growth)
        << ", \"chunk_bytes\": " << report.config.chunk_bytes
        << ", \"min_class_bytes\": " << report.config.min_class_bytes
        << ", \"max_class_bytes\": " << report.config.max_class_bytes
        << ", \"power_of_two\": " << (report.config.power_of_two ? "true" : "false")
        << ", \"extra_sizes\": [";
    for (std::size_t i = 0; i < report.config.extra_sizes.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << report.config.extra_sizes[i];
    }
    out << "]},\n";
    out << "  \"classes\": " << report.classes << ",\n";
    out << "  \"waste\": {\"internal\": " << json_number(report.waste.internal)
        << ", \"tail\": " << json_number(report.waste.tail) << "},\n";
    out << "  \"peak\": {\"resident_bytes\": " << report.peak.resident_bytes
        << ", \"blocks\": " << report.peak.blocks << "},\n";
    out << "  \"retention_bound_bytes\": " << report.retention_bound_bytes;
    if (report.has_bench) {
        out << ",\n";
        write_bench_json(out, report.bench);
    } else {
        out << "\n";
    }
    out << "}\n";
    return out.str();
}

// A strict JSON subset: no `null` (a required field set to null is a missing field) and no
// NaN/Infinity, which would let a verdict be computed on garbage.

struct JsonValue {
    enum class Kind { boolean, number, string, array, object };

    Kind kind = Kind::number;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;

    const JsonValue* find(const std::string& key) const noexcept {
        for (const auto& entry : object) {
            if (entry.first == key) {
                return &entry.second;
            }
        }
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    JsonValue parse_document() {
        JsonValue value = parse_value(0);
        skip_whitespace();
        if (position_ != text_.size()) {
            fail("trailing characters after the document");
        }
        return value;
    }

private:
    static constexpr int max_depth = 32;

    [[noreturn]] void fail(const std::string& message) const {
        std::size_t line = 1;
        for (std::size_t i = 0; i < position_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
            }
        }
        throw ReportError("poolkit: report JSON line " + std::to_string(line) + ": " + message);
    }

    void skip_whitespace() noexcept {
        while (position_ < text_.size()) {
            const char c = text_[position_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++position_;
            } else {
                break;
            }
        }
    }

    bool consume(char expected) noexcept {
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected, const char* what) {
        if (!consume(expected)) {
            fail(std::string("expected ") + what);
        }
    }

    JsonValue parse_value(int depth) {
        if (depth > max_depth) {
            fail("nesting deeper than " + std::to_string(max_depth));
        }
        skip_whitespace();
        if (position_ >= text_.size()) {
            fail("unexpected end of document");
        }
        const char c = text_[position_];
        JsonValue value;
        if (c == '{') {
            value.kind = JsonValue::Kind::object;
            parse_object(value, depth);
            return value;
        }
        if (c == '[') {
            value.kind = JsonValue::Kind::array;
            parse_array(value, depth);
            return value;
        }
        if (c == '"') {
            value.kind = JsonValue::Kind::string;
            value.string = parse_string();
            return value;
        }
        if (text_.compare(position_, 4, "true") == 0) {
            position_ += 4;
            value.kind = JsonValue::Kind::boolean;
            value.boolean = true;
            return value;
        }
        if (text_.compare(position_, 5, "false") == 0) {
            position_ += 5;
            value.kind = JsonValue::Kind::boolean;
            value.boolean = false;
            return value;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            value.kind = JsonValue::Kind::number;
            value.number = parse_number();
            return value;
        }
        fail(std::string("unexpected character '") + c + "'");
    }

    void parse_object(JsonValue& value, int depth) {
        expect('{', "'{'");
        skip_whitespace();
        if (consume('}')) {
            return;
        }
        for (;;) {
            skip_whitespace();
            if (position_ >= text_.size() || text_[position_] != '"') {
                fail("expected a string key");
            }
            const std::string key = parse_string();
            if (value.find(key) != nullptr) {
                fail("duplicate key '" + key + "'");
            }
            skip_whitespace();
            expect(':', "':'");
            value.object.emplace_back(key, parse_value(depth + 1));
            skip_whitespace();
            if (consume(',')) {
                continue;
            }
            if (consume('}')) {
                return;
            }
            fail("expected ',' or '}'");
        }
    }

    void parse_array(JsonValue& value, int depth) {
        expect('[', "'['");
        skip_whitespace();
        if (consume(']')) {
            return;
        }
        for (;;) {
            value.array.push_back(parse_value(depth + 1));
            skip_whitespace();
            if (consume(',')) {
                continue;
            }
            if (consume(']')) {
                return;
            }
            fail("expected ',' or ']'");
        }
    }

    std::string parse_string() {
        expect('"', "'\"'");
        std::string out;
        for (;;) {
            if (position_ >= text_.size()) {
                fail("unterminated string");
            }
            const char c = text_[position_++];
            if (c == '"') {
                return out;
            }
            if (c == '\\') {
                if (position_ >= text_.size()) {
                    fail("unterminated escape");
                }
                const char escape = text_[position_++];
                switch (escape) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        if (position_ + 4 > text_.size()) {
                            fail("truncated \\u escape");
                        }
                        std::uint32_t code = 0;
                        for (int i = 0; i < 4; ++i) {
                            const char digit = text_[position_++];
                            std::uint32_t nibble = 0;
                            if (digit >= '0' && digit <= '9') {
                                nibble = static_cast<std::uint32_t>(digit - '0');
                            } else if (digit >= 'a' && digit <= 'f') {
                                nibble = static_cast<std::uint32_t>(digit - 'a' + 10);
                            } else if (digit >= 'A' && digit <= 'F') {
                                nibble = static_cast<std::uint32_t>(digit - 'A' + 10);
                            } else {
                                fail("invalid \\u escape");
                            }
                            code = (code << 4) | nibble;
                        }
                        // Only ASCII and the two-byte range: enough for paths,
                        // and a surrogate pair would need state this parser
                        // deliberately does not keep.
                        if (code <= 0x7Fu) {
                            out += static_cast<char>(code);
                        } else if (code <= 0x7FFu) {
                            out += static_cast<char>(0xC0u | (code >> 6));
                            out += static_cast<char>(0x80u | (code & 0x3Fu));
                        } else {
                            out += static_cast<char>(0xE0u | (code >> 12));
                            out += static_cast<char>(0x80u | ((code >> 6) & 0x3Fu));
                            out += static_cast<char>(0x80u | (code & 0x3Fu));
                        }
                        break;
                    }
                    default:
                        fail(std::string("invalid escape '\\") + escape + "'");
                }
                continue;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                fail("raw control character in a string");
            }
            out += c;
        }
    }

    double parse_number() {
        const std::size_t start = position_;
        if (position_ < text_.size() && text_[position_] == '-') {
            ++position_;
        }
        bool digits = false;
        while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
            ++position_;
            digits = true;
        }
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
                digits = true;
            }
        }
        if (!digits) {
            fail("expected a number");
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() &&
                (text_[position_] == '+' || text_[position_] == '-')) {
                ++position_;
            }
            bool exponent_digits = false;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
                exponent_digits = true;
            }
            if (!exponent_digits) {
                fail("expected digits in the exponent");
            }
        }
        const std::string token = text_.substr(start, position_ - start);
        // Classic locale, not strtod: LC_NUMERIC is process-global and a host
        // that sets it to a decimal comma would read "1.5" as "1".
        std::istringstream stream(token);
        stream.imbue(std::locale::classic());
        double value = 0.0;
        stream >> value;
        if (stream.fail() || !std::isfinite(value)) {
            fail("number '" + token + "' is not a finite double");
        }
        return value;
    }

    const std::string& text_;
    std::size_t position_ = 0;
};

inline const JsonValue& require_member(const JsonValue& object, const std::string& path,
                                       const std::string& key) {
    if (object.kind != JsonValue::Kind::object) {
        throw ReportError("poolkit: report field '" + path + "' is not an object");
    }
    const JsonValue* member = object.find(key);
    if (member == nullptr) {
        throw ReportError("poolkit: report is missing field '" + path + "." + key + "'");
    }
    return *member;
}

// Largest integer a JSON double identifies exactly: past 2^53 consecutive integers collapse onto
// one double. The old bound, 2^64-1, *is* 2^64 as a double and casting that is UB, so the honest
// option is to refuse what a double cannot represent.
inline constexpr double kMaxExactInteger = 9007199254740992.0;  // 2^53

inline std::size_t require_uint(const JsonValue& object, const std::string& path,
                                const std::string& key) {
    const JsonValue& value = require_member(object, path, key);
    if (value.kind != JsonValue::Kind::number || value.number < 0.0 ||
        value.number != std::floor(value.number) || value.number >= kMaxExactInteger) {
        throw ReportError("poolkit: report field '" + path + "." + key +
                          "' must be a non-negative integer below 2^53 "
                          "(9007199254740992, the largest exact JSON double)");
    }
    return static_cast<std::size_t>(value.number);
}

inline int require_int(const JsonValue& object, const std::string& path, const std::string& key) {
    const JsonValue& value = require_member(object, path, key);
    if (value.kind != JsonValue::Kind::number || value.number != std::floor(value.number) ||
        std::fabs(value.number) > 2147483647.0) {
        throw ReportError("poolkit: report field '" + path + "." + key + "' must be an integer");
    }
    return static_cast<int>(value.number);
}

inline double require_number(const JsonValue& object, const std::string& path,
                             const std::string& key) {
    const JsonValue& value = require_member(object, path, key);
    if (value.kind != JsonValue::Kind::number || !std::isfinite(value.number)) {
        throw ReportError("poolkit: report field '" + path + "." + key +
                          "' must be a finite number");
    }
    return value.number;
}

inline std::string require_string(const JsonValue& object, const std::string& path,
                                  const std::string& key) {
    const JsonValue& value = require_member(object, path, key);
    if (value.kind != JsonValue::Kind::string) {
        throw ReportError("poolkit: report field '" + path + "." + key + "' must be a string");
    }
    return value.string;
}

inline bool require_bool(const JsonValue& object, const std::string& path, const std::string& key) {
    const JsonValue& value = require_member(object, path, key);
    if (value.kind != JsonValue::Kind::boolean) {
        throw ReportError("poolkit: report field '" + path + "." + key + "' must be a boolean");
    }
    return value.boolean;
}

inline void reject_unknown_keys(const JsonValue& object, const std::string& path,
                                const std::vector<std::string>& allowed) {
    for (const auto& entry : object.object) {
        bool known = false;
        for (const std::string& key : allowed) {
            if (entry.first == key) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw ReportError("poolkit: report field '" + path + "." + entry.first +
                              "' is not part of schema " + std::to_string(kReportSchema) +
                              "/" + std::to_string(kBenchReportSchema));
        }
    }
}

inline BenchDistribution read_distribution(const JsonValue& object, const std::string& path,
                                           const std::string& key) {
    const JsonValue& value = require_member(object, path, key);
    reject_unknown_keys(value, path + "." + key, {"median", "p99", "min", "max"});
    BenchDistribution d;
    d.median = require_number(value, path + "." + key, "median");
    d.p99 = require_number(value, path + "." + key, "p99");
    d.min = require_number(value, path + "." + key, "min");
    d.max = require_number(value, path + "." + key, "max");
    if (d.min < 0.0 || d.min > d.median || d.median > d.p99 || d.p99 > d.max) {
        throw ReportError("poolkit: report field '" + path + "." + key +
                          "' is not an ordered distribution (min <= median <= p99 <= max)");
    }
    return d;
}

inline BenchMode read_bench_mode(const JsonValue& value, const std::string& path) {
    if (value.kind != JsonValue::Kind::object) {
        throw ReportError("poolkit: report field '" + path + "' must be an object");
    }
    BenchMode mode;
    mode.mode = require_string(value, path, "mode");
    if (mode.mode != "pool" && mode.mode != "cudamalloc" && mode.mode != "cudaasync") {
        throw ReportError("poolkit: report field '" + path + ".mode' is '" + mode.mode +
                          "', expected pool, cudamalloc or cudaasync");
    }
    std::vector<std::string> allowed = {"mode",    "ops",         "allocs",  "frees",
                                        "peak_bytes", "peak_sampled", "sample_every_ops",
                                        "wall_ns", "alloc_ns",    "free_ns"};
    if (mode.mode == "pool") {
        allowed.push_back("hit_ns");
        allowed.push_back("hit_ops");
        allowed.push_back("growth_ns");
        allowed.push_back("growth_ops");
        allowed.push_back("slot_ns");
        allowed.push_back("slot_ops");
        allowed.push_back("chunk_ns");
        allowed.push_back("chunk_ops");
        allowed.push_back("large_ns");
        allowed.push_back("large_ops");
    }
    reject_unknown_keys(value, path, allowed);

    mode.ops = require_uint(value, path, "ops");
    mode.allocs = require_uint(value, path, "allocs");
    mode.frees = require_uint(value, path, "frees");
    mode.peak_bytes = require_uint(value, path, "peak_bytes");
    mode.peak_sampled = require_bool(value, path, "peak_sampled");
    mode.sample_every_ops = require_uint(value, path, "sample_every_ops");
    mode.wall_ns = read_distribution(value, path, "wall_ns");
    mode.alloc_ns = read_distribution(value, path, "alloc_ns");
    mode.free_ns = read_distribution(value, path, "free_ns");
    if (mode.mode == "pool") {
        mode.hit_ns = read_distribution(value, path, "hit_ns");
        mode.hit_ops = require_uint(value, path, "hit_ops");
        mode.growth_ns = read_distribution(value, path, "growth_ns");
        mode.growth_ops = require_uint(value, path, "growth_ops");
        mode.slot_ns = read_distribution(value, path, "slot_ns");
        mode.slot_ops = require_uint(value, path, "slot_ops");
        mode.chunk_ns = read_distribution(value, path, "chunk_ns");
        mode.chunk_ops = require_uint(value, path, "chunk_ops");
        mode.large_ns = read_distribution(value, path, "large_ns");
        mode.large_ops = require_uint(value, path, "large_ops");
        if (mode.hit_ops + mode.growth_ops != mode.allocs) {
            throw ReportError("poolkit: report field '" + path +
                              "': hit_ops + growth_ops is not allocs");
        }
        if (mode.slot_ops + mode.chunk_ops + mode.large_ops != mode.growth_ops) {
            throw ReportError("poolkit: report field '" + path +
                              "': slot_ops + chunk_ops + large_ops is not growth_ops");
        }
    }
    // The sampling contract, checked instead of assumed.
    if (mode.peak_sampled && mode.sample_every_ops == 0) {
        throw ReportError("poolkit: report field '" + path +
                          "' samples its peak with sample_every_ops = 0");
    }
    if (!mode.peak_sampled && mode.sample_every_ops != 0) {
        throw ReportError("poolkit: report field '" + path +
                          "' claims an exact peak but sets sample_every_ops");
    }
    if (mode.ops != mode.allocs + mode.frees) {
        throw ReportError("poolkit: report field '" + path +
                          "' counts ops that are neither allocs nor frees");
    }
    return mode;
}

inline BenchSection read_bench_section(const JsonValue& value, const std::string& path) {
    if (value.kind != JsonValue::Kind::object) {
        throw ReportError("poolkit: report field '" + path + "' must be an object");
    }
    reject_unknown_keys(value, path, {"schema", "env", "modes"});
    BenchSection bench;
    bench.schema = require_int(value, path, "schema");
    if (bench.schema != kBenchSectionSchema) {
        throw ReportError("poolkit: report field '" + path + ".schema' is " +
                          std::to_string(bench.schema) + ", this build reads " +
                          std::to_string(kBenchSectionSchema));
    }

    const JsonValue& env = require_member(value, path, "env");
    reject_unknown_keys(env, path + ".env",
                        {"gpu", "compute", "driver", "runtime", "arch", "chunk_bytes", "growth",
                         "min_class_bytes", "max_class_bytes", "warmup", "repeat"});
    bench.env.gpu = require_string(env, path + ".env", "gpu");
    bench.env.compute = require_string(env, path + ".env", "compute");
    bench.env.driver_version = require_int(env, path + ".env", "driver");
    bench.env.runtime_version = require_int(env, path + ".env", "runtime");
    bench.env.arch = require_string(env, path + ".env", "arch");
    bench.env.chunk_bytes = require_uint(env, path + ".env", "chunk_bytes");
    bench.env.growth = require_number(env, path + ".env", "growth");
    bench.env.min_class_bytes = require_uint(env, path + ".env", "min_class_bytes");
    bench.env.max_class_bytes = require_uint(env, path + ".env", "max_class_bytes");
    bench.env.warmup = require_int(env, path + ".env", "warmup");
    bench.env.repeat = require_int(env, path + ".env", "repeat");

    const JsonValue& modes = require_member(value, path, "modes");
    if (modes.kind != JsonValue::Kind::array || modes.array.empty()) {
        throw ReportError("poolkit: report field '" + path + ".modes' must be a non-empty array");
    }
    for (std::size_t i = 0; i < modes.array.size(); ++i) {
        bench.modes.push_back(
            read_bench_mode(modes.array[i], path + ".modes[" + std::to_string(i) + "]"));
    }
    return bench;
}

inline Report from_json(const std::string& text) {
    const JsonValue root = JsonParser(text).parse_document();
    if (root.kind != JsonValue::Kind::object) {
        throw ReportError("poolkit: a report must be a JSON object");
    }

    const int schema = require_int(root, "$", "schema");
    if (schema != kReportSchema && schema != kBenchReportSchema) {
        throw ReportError("poolkit: report schema " + std::to_string(schema) +
                          " is not supported by this build (reads schema " +
                          std::to_string(kReportSchema) + " and " +
                          std::to_string(kBenchReportSchema) + ")");
    }
    std::vector<std::string> root_keys = {"schema", "tool", "trace", "config", "classes", "waste",
                                          "peak", "retention_bound_bytes"};
    if (schema == kBenchReportSchema) {
        root_keys.push_back("bench");
    }
    reject_unknown_keys(root, "$", root_keys);

    Report report;
    report.schema = schema;
    report.tool = require_string(root, "$", "tool");

    const JsonValue& trace = require_member(root, "$", "trace");
    reject_unknown_keys(trace, "trace", {"path", "ops", "allocs", "sha256"});
    report.trace.path = require_string(trace, "trace", "path");
    report.trace.ops = require_uint(trace, "trace", "ops");
    report.trace.allocs = require_uint(trace, "trace", "allocs");
    report.trace.sha256 = require_string(trace, "trace", "sha256");
    if (!report.trace.sha256.empty() && !is_sha256_hex(report.trace.sha256)) {
        throw ReportError("poolkit: report field 'trace.sha256' must be 64 lowercase hex "
                          "characters or an empty string");
    }

    const JsonValue& config = require_member(root, "$", "config");
    reject_unknown_keys(config, "config",
                        {"growth", "chunk_bytes", "min_class_bytes", "max_class_bytes",
                         "power_of_two", "extra_sizes"});
    report.config.growth = require_number(config, "config", "growth");
    report.config.chunk_bytes = require_uint(config, "config", "chunk_bytes");
    report.config.min_class_bytes = require_uint(config, "config", "min_class_bytes");
    report.config.max_class_bytes = require_uint(config, "config", "max_class_bytes");
    report.config.power_of_two = require_bool(config, "config", "power_of_two");
    const JsonValue& extra = require_member(config, "config", "extra_sizes");
    if (extra.kind != JsonValue::Kind::array) {
        throw ReportError("poolkit: report field 'config.extra_sizes' must be an array");
    }
    for (const JsonValue& item : extra.array) {
        if (item.kind != JsonValue::Kind::number || item.number < 0.0 ||
            item.number != std::floor(item.number) || item.number >= kMaxExactInteger) {
            throw ReportError(
                "poolkit: report field 'config.extra_sizes' must hold non-negative integers "
                "below 2^53 (9007199254740992, the largest exact JSON double)");
        }
        report.config.extra_sizes.push_back(static_cast<std::size_t>(item.number));
    }

    report.classes = require_uint(root, "$", "classes");

    const JsonValue& waste = require_member(root, "$", "waste");
    reject_unknown_keys(waste, "waste", {"internal", "tail"});
    report.waste.internal = require_number(waste, "waste", "internal");
    report.waste.tail = require_number(waste, "waste", "tail");

    const JsonValue& peak = require_member(root, "$", "peak");
    reject_unknown_keys(peak, "peak", {"resident_bytes", "blocks"});
    report.peak.resident_bytes = require_uint(peak, "peak", "resident_bytes");
    report.peak.blocks = require_uint(peak, "peak", "blocks");

    report.retention_bound_bytes = require_uint(root, "$", "retention_bound_bytes");

    if (schema == kBenchReportSchema) {
        report.has_bench = true;
        report.bench = read_bench_section(require_member(root, "$", "bench"), "bench");
    }

    if (report.config.chunk_bytes == 0 || report.config.min_class_bytes == 0 ||
        report.config.max_class_bytes < report.config.min_class_bytes ||
        report.config.growth <= 1.0 || report.classes == 0) {
        throw ReportError("poolkit: report config is internally inconsistent");
    }
    return report;
}

}  // namespace tools
}  // namespace poolkit

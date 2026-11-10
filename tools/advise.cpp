// poolkit_advise CLI: argument parsing only, the measuring lives in advise.hpp, so the exit codes the
// tests assert are the ones the binary returns. 0 on success, 2 on bad arguments or an unreadable trace.

#include "advise.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* program) {
    std::cout
        << "usage: " << program << " --trace <file.jsonl> [options]\n"
        << "\n"
        << "Sweeps growth x powers-of-two x chunk x max class over a trace, replays\n"
        << "every configuration to get the simulated peak, and prints the ranking.\n"
        << "\n"
        << "Options:\n"
        << "  --trace <f>            trace to analyse (required)\n"
        << "  --chunk <size>         reference chunk_bytes (default 2MiB)\n"
        << "  --min-class <size>     min_class_bytes (default 256)\n"
        << "  --max-class <size>     reference max_class_bytes (default 2MiB)\n"
        << "  --growth <f>           reference growth factor (default 1.5)\n"
        << "  --no-power-of-two      reference table without the powers of two\n"
        << "  --extra <list>         extra class capacities, comma separated\n"
        << "  --json <f>             write the recommended configuration's report\n"
        << "  --top <n>              rows in the table (default 10)\n"
        << "  --help, -h             show this help\n"
        << "\n"
        << "Sizes accept a plain byte count or a K/KiB/M/MiB/G/GiB suffix\n"
        << "(always powers of two: 1MiB = 1048576).\n"
        << "\n"
        << "Exit codes: 0 success, 2 invalid arguments or unreadable trace.\n";
}

bool parse_bytes(const std::string& text, std::size_t& out) {
    std::size_t position = 0;
    std::uint64_t value = 0;
    while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position]))) {
        const std::uint64_t digit = static_cast<std::uint64_t>(text[position] - '0');
        if (value > (18446744073709551615ull - digit) / 10ull) {
            return false;  // would wrap
        }
        value = value * 10ull + digit;
        ++position;
    }
    if (position == 0) {
        return false;
    }

    std::string suffix;
    suffix.reserve(text.size() - position);
    for (std::size_t i = position; i < text.size(); ++i) {
        suffix += static_cast<char>(std::tolower(static_cast<unsigned char>(text[i])));
    }

    // Powers of two only. "KB" is deliberately not accepted: it means 1000 to some readers and 1024 to
    // others, and a silently different chunk size is the kind of unit bug this repo measures for a living.
    std::uint64_t multiplier = 0;
    if (suffix.empty() || suffix == "b") {
        multiplier = 1;
    } else if (suffix == "k" || suffix == "kib") {
        multiplier = 1024;
    } else if (suffix == "m" || suffix == "mib") {
        multiplier = 1024ull * 1024ull;
    } else if (suffix == "g" || suffix == "gib") {
        multiplier = 1024ull * 1024ull * 1024ull;
    } else {
        return false;
    }

    if (value > 18446744073709551615ull / multiplier) {
        return false;
    }
    const std::uint64_t bytes = value * multiplier;
    if (bytes > static_cast<std::uint64_t>(static_cast<std::size_t>(-1))) {
        return false;
    }
    out = static_cast<std::size_t>(bytes);
    return true;
}

bool parse_double(const std::string& text, double& out) {
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    double value = 0.0;
    stream >> value;
    if (stream.fail()) {
        return false;
    }
    stream >> std::ws;
    if (!stream.eof()) {
        return false;
    }
    out = value;
    return true;
}

bool parse_count(const std::string& text, std::size_t& out) {
    if (text.empty()) {
        return false;
    }
    std::uint64_t value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(ch - '0');
        if (value > (18446744073709551615ull - digit) / 10ull) {
            return false;
        }
        value = value * 10ull + digit;
    }
    out = static_cast<std::size_t>(value);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    poolkit::tools::AdviseOptions options;
    bool has_trace = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto take = [&](std::string& target) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << arg << "\n";
                return false;
            }
            target = argv[++i];
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--trace") {
            if (!take(options.trace_path)) {
                return 2;
            }
            has_trace = true;
        } else if (arg == "--json") {
            if (!take(options.json_path)) {
                return 2;
            }
        } else if (arg == "--chunk" || arg == "--min-class" || arg == "--max-class") {
            std::string value;
            if (!take(value)) {
                return 2;
            }
            std::size_t bytes = 0;
            if (!parse_bytes(value, bytes) || bytes == 0) {
                std::cerr << "invalid " << arg << " '" << value
                          << "' (expected a positive byte count with an optional K/M/G suffix)\n";
                return 2;
            }
            if (arg == "--chunk") {
                options.chunk_bytes = bytes;
            } else if (arg == "--min-class") {
                options.min_class_bytes = bytes;
            } else {
                options.max_class_bytes = bytes;
            }
        } else if (arg == "--growth") {
            std::string value;
            if (!take(value)) {
                return 2;
            }
            double growth = 0.0;
            if (!parse_double(value, growth) || !(growth > 1.0) || growth > 16.0) {
                std::cerr << "invalid --growth '" << value << "' (expected a finite value > 1)\n";
                return 2;
            }
            options.growth = growth;
        } else if (arg == "--no-power-of-two") {
            options.power_of_two = false;
        } else if (arg == "--extra") {
            std::string value;
            if (!take(value)) {
                return 2;
            }
            std::size_t start = 0;
            for (;;) {
                const std::size_t comma = value.find(',', start);
                const std::string item = value.substr(start, comma - start);
                std::size_t bytes = 0;
                if (item.empty() || !parse_bytes(item, bytes) || bytes == 0) {
                    std::cerr << "invalid --extra item '" << item << "'\n";
                    return 2;
                }
                options.extra_sizes.push_back(bytes);
                if (comma == std::string::npos) {
                    break;
                }
                start = comma + 1;
            }
        } else if (arg == "--top") {
            std::string value;
            if (!take(value)) {
                return 2;
            }
            std::size_t count = 0;
            if (!parse_count(value, count) || count == 0) {
                std::cerr << "invalid --top '" << value << "' (expected a positive integer)\n";
                return 2;
            }
            options.top = count;
        } else {
            std::cerr << "unknown argument: " << arg << "\n\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    if (!has_trace) {
        std::cerr << "--trace is required\n\n";
        print_usage(argv[0]);
        return 2;
    }

    // Structural validation before the sweep: a bad config must fail in milliseconds, not after 1400 replays.
    if (options.max_class_bytes < options.min_class_bytes) {
        std::cerr << "--max-class must be >= --min-class\n";
        return 2;
    }
    if (options.chunk_bytes < options.min_class_bytes) {
        std::cerr << "--chunk must be >= --min-class\n";
        return 2;
    }
    for (const std::size_t extra : options.extra_sizes) {
        if (extra < options.min_class_bytes || extra > options.max_class_bytes) {
            std::cerr << "--extra " << extra << " B is outside [--min-class, --max-class]\n";
            return 2;
        }
    }

    return poolkit::tools::run_advise(options, std::cout, std::cerr);
}

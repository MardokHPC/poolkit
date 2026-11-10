// poolkit_gate CLI: argument parsing only, the decision logic lives in gate.hpp. Exit codes: 0
// pass/improvement, 1 regression, 2 suspect or error, the same contract as cuda-timer's perf_gate.

#include "gate.hpp"

#include <cmath>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cout
        << "usage: " << program
        << " --baseline <report.json> --current <report.json> [options]\n"
        << "   or: " << program
        << " --baseline-trace <a.jsonl> --current-trace <b.jsonl> [options]\n"
        << "\n"
        << "Compares two poolkit_advise reports (or two traces replayed with the\n"
        << "reference configuration) and decides whether memory regressed.\n"
        << "\n"
        << "Options:\n"
        << "  --max-peak-increase <f>   max relative peak growth (default 0.05)\n"
        << "  --max-waste-increase <f>  max absolute waste growth, as a fraction\n"
        << "                            (default 0.02 = 2 percentage points)\n"
        << "  --json <f>                write the JSON verdict to a file\n"
        << "  --quiet                   do not print the JSON verdict on stdout\n"
        << "  --help, -h                show this help\n"
        << "\n"
        << "The verdict is 'suspect' (exit 2) when the comparison is not valid:\n"
        << "different trace, different configuration, unknown report schema, or a\n"
        << "report this build cannot read.\n"
        << "\n"
        << "Exit codes: 0 pass/improvement, 1 regression, 2 suspect or error.\n";
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

}  // namespace

int main(int argc, char** argv) {
    poolkit::tools::GateOptions options;

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
        } else if (arg == "--baseline") {
            if (!take(options.baseline_report)) {
                return 2;
            }
        } else if (arg == "--current") {
            if (!take(options.current_report)) {
                return 2;
            }
        } else if (arg == "--baseline-trace") {
            if (!take(options.baseline_trace)) {
                return 2;
            }
        } else if (arg == "--current-trace") {
            if (!take(options.current_trace)) {
                return 2;
            }
        } else if (arg == "--json") {
            if (!take(options.json_path)) {
                return 2;
            }
        } else if (arg == "--max-peak-increase" || arg == "--max-waste-increase") {
            std::string value;
            if (!take(value)) {
                return 2;
            }
            double parsed = 0.0;
            if (!parse_double(value, parsed) || !std::isfinite(parsed) || parsed < 0.0) {
                std::cerr << "invalid " << arg << " '" << value
                          << "' (expected a finite value >= 0)\n";
                return 2;
            }
            if (arg == "--max-peak-increase") {
                options.max_peak_increase = parsed;
            } else {
                options.max_waste_increase = parsed;
            }
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    return poolkit::tools::run_gate(options, std::cout, std::cerr);
}

#pragma once

// The memory gate: did this change make the job need more memory, by how much, and in which
// component? Only the same trace, config and readable schema compare; anything else is suspect,
// never regression: a gate that guesses teaches people to ignore it. `k` is reported, not decisive.

#include "advise.hpp"
#include "report_json.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace poolkit {
namespace tools {

enum class Verdict { pass, improvement, regression, suspect };

inline const char* verdict_name(Verdict verdict) noexcept {
    switch (verdict) {
        case Verdict::pass: return "pass";
        case Verdict::improvement: return "improvement";
        case Verdict::regression: return "regression";
        case Verdict::suspect: return "suspect";
    }
    return "suspect";
}

inline int exit_code(Verdict verdict) noexcept {
    switch (verdict) {
        case Verdict::pass:
        case Verdict::improvement:
            return 0;
        case Verdict::regression:
            return 1;
        case Verdict::suspect:
            return 2;
    }
    return 2;
}

struct GateOptions {
    std::string baseline_report;
    std::string current_report;
    std::string baseline_trace;
    std::string current_trace;
    double max_peak_increase = 0.05;
    double max_waste_increase = 0.02;
    std::string json_path;
    bool quiet = false;
};

struct MetricComparison {
    std::string name;
    double baseline = 0.0;
    double current = 0.0;
    double delta = 0.0;
    double relative = 0.0;
    bool relative_defined = false;
    // bytes are judged relative, fraction absolute (percentage points), count is reported only.
    enum class Kind { bytes, fraction, count } kind = Kind::count;
    bool regression = false;
    bool improvement = false;
};

struct GateResult {
    Verdict verdict = Verdict::suspect;
    std::string reason;
    Report baseline;
    Report current;
    std::vector<MetricComparison> metrics;
    std::vector<std::string> notes;

    std::string to_json(const GateOptions& options) const;
};

namespace gate_detail {

inline std::string first_config_difference(const ConfigIdentity& a, const ConfigIdentity& b) {
    if (a.growth != b.growth) {
        return "growth " + json_number(a.growth) + " vs " + json_number(b.growth);
    }
    if (a.chunk_bytes != b.chunk_bytes) {
        return "chunk_bytes " + std::to_string(a.chunk_bytes) + " vs " +
               std::to_string(b.chunk_bytes);
    }
    if (a.min_class_bytes != b.min_class_bytes) {
        return "min_class_bytes " + std::to_string(a.min_class_bytes) + " vs " +
               std::to_string(b.min_class_bytes);
    }
    if (a.max_class_bytes != b.max_class_bytes) {
        return "max_class_bytes " + std::to_string(a.max_class_bytes) + " vs " +
               std::to_string(b.max_class_bytes);
    }
    if (a.power_of_two != b.power_of_two) {
        return std::string("power_of_two ") + (a.power_of_two ? "true" : "false") + " vs " +
               (b.power_of_two ? "true" : "false");
    }
    return "extra_sizes differ";
}

}  // namespace gate_detail

inline GateResult compare_reports(const Report& baseline, const Report& current,
                                  const GateOptions& options) {
    GateResult result;
    result.baseline = baseline;
    result.current = current;

    std::vector<std::string> problems;

    const auto supported_schema = [](int schema) {
        return schema == kReportSchema || schema == kBenchReportSchema;
    };
    if (!supported_schema(baseline.schema) || !supported_schema(current.schema) ||
        baseline.schema != current.schema) {
        problems.push_back("schema " + std::to_string(baseline.schema) + " vs " +
                           std::to_string(current.schema));
    }
    if (baseline.tool != "poolkit_advise" || current.tool != "poolkit_advise") {
        problems.push_back("not advise reports ('" + baseline.tool + "' vs '" + current.tool + "')");
    }

    const bool baseline_hashed = !baseline.trace.sha256.empty();
    const bool current_hashed = !current.trace.sha256.empty();
    if (baseline_hashed && current_hashed) {
        if (baseline.trace.sha256 != current.trace.sha256) {
            problems.push_back("different trace: sha256 " + baseline.trace.sha256.substr(0, 12) +
                               "... vs " + current.trace.sha256.substr(0, 12) + "...");
        }
    } else if (baseline_hashed != current_hashed) {
        result.notes.push_back("only one report carries a trace hash; identity checked on "
                               "ops/allocs only");
    } else if (!baseline_hashed) {
        result.notes.push_back("no trace hash in either report; identity checked on ops/allocs "
                               "only");
    }
    if (baseline.trace.ops != current.trace.ops ||
        baseline.trace.allocs != current.trace.allocs) {
        problems.push_back("different trace: " + std::to_string(baseline.trace.ops) + "/" +
                           std::to_string(baseline.trace.allocs) + " ops/allocs vs " +
                           std::to_string(current.trace.ops) + "/" +
                           std::to_string(current.trace.allocs));
    }
    if (baseline.trace.ops == 0) {
        problems.push_back("baseline report carries no trace identity (ops = 0)");
    }

    if (!same_config(baseline.config, current.config)) {
        problems.push_back("different configuration: " +
                           gate_detail::first_config_difference(baseline.config, current.config));
    }

    if (baseline.peak.resident_bytes == 0) {
        problems.push_back("baseline peak is 0: a relative change is undefined");
    }

    const auto add = [&](std::string name, double base, double cur,
                         MetricComparison::Kind kind) {
        MetricComparison metric;
        metric.name = std::move(name);
        metric.baseline = base;
        metric.current = cur;
        metric.delta = cur - base;
        metric.kind = kind;
        metric.relative_defined = base != 0.0;
        metric.relative = base != 0.0 ? (cur - base) / base : 0.0;

        if (kind == MetricComparison::Kind::bytes) {
            if (metric.relative_defined) {
                metric.regression = metric.relative > options.max_peak_increase;
                metric.improvement = metric.relative < -options.max_peak_increase;
            }
        } else if (kind == MetricComparison::Kind::fraction) {
            metric.regression = metric.delta > options.max_waste_increase;
            metric.improvement = metric.delta < -options.max_waste_increase;
        }
        result.metrics.push_back(std::move(metric));
    };

    add("peak_resident_bytes", static_cast<double>(baseline.peak.resident_bytes),
        static_cast<double>(current.peak.resident_bytes), MetricComparison::Kind::bytes);
    add("internal_waste", baseline.waste.internal, current.waste.internal,
        MetricComparison::Kind::fraction);
    add("tail_waste", baseline.waste.tail, current.waste.tail, MetricComparison::Kind::fraction);
    add("classes", static_cast<double>(baseline.classes), static_cast<double>(current.classes),
        MetricComparison::Kind::count);
    add("retention_bound_bytes", static_cast<double>(baseline.retention_bound_bytes),
        static_cast<double>(current.retention_bound_bytes), MetricComparison::Kind::bytes);

    if (!problems.empty()) {
        result.verdict = Verdict::suspect;
        std::ostringstream reason;
        for (std::size_t i = 0; i < problems.size(); ++i) {
            if (i != 0) {
                reason << "; ";
            }
            reason << problems[i];
        }
        result.reason = reason.str();
        return result;
    }

    // A regression anywhere decides, even if something else improved: "the peak got 10 % bigger but
    // the waste got 3 pp better" is not a pass, the job may no longer fit.
    for (const MetricComparison& metric : result.metrics) {
        if (metric.regression) {
            result.verdict = Verdict::regression;
            std::ostringstream reason;
            reason.imbue(std::locale::classic());
            reason << metric.name << " grew " << std::fixed << std::setprecision(3)
                   << (metric.kind == MetricComparison::Kind::bytes ? metric.relative * 100.0
                                                                   : metric.delta * 100.0)
                   << (metric.kind == MetricComparison::Kind::bytes ? " %" : " pp")
                   << " > "
                   << (metric.kind == MetricComparison::Kind::bytes
                           ? options.max_peak_increase * 100.0
                           : options.max_waste_increase * 100.0)
                   << (metric.kind == MetricComparison::Kind::bytes ? " %" : " pp");
            result.reason = reason.str();
            return result;
        }
    }
    for (const MetricComparison& metric : result.metrics) {
        if (metric.improvement) {
            result.verdict = Verdict::improvement;
            std::ostringstream reason;
            reason.imbue(std::locale::classic());
            reason << metric.name << " fell " << std::fixed << std::setprecision(3)
                   << (metric.kind == MetricComparison::Kind::bytes ? -metric.relative * 100.0
                                                                   : -metric.delta * 100.0)
                   << (metric.kind == MetricComparison::Kind::bytes ? " %" : " pp")
                   << " beyond the "
                   << (metric.kind == MetricComparison::Kind::bytes
                           ? options.max_peak_increase * 100.0
                           : options.max_waste_increase * 100.0)
                   << (metric.kind == MetricComparison::Kind::bytes ? " %" : " pp")
                   << " budget";
            result.reason = reason.str();
            return result;
        }
    }

    result.verdict = Verdict::pass;
    result.reason = "all metrics inside their thresholds";
    return result;
}

inline std::string GateResult::to_json(const GateOptions& options) const {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\n";
    out << "  \"schema\": " << kReportSchema << ",\n";
    out << "  \"tool\": \"poolkit_gate\",\n";
    out << "  \"verdict\": \"" << verdict_name(verdict) << "\",\n";
    out << "  \"reason\": \"" << json_escape(reason) << "\",\n";
    out << "  \"trace\": {\"sha256\": \"" << json_escape(baseline.trace.sha256)
        << "\", \"ops\": " << baseline.trace.ops << ", \"allocs\": " << baseline.trace.allocs
        << "},\n";
    out << "  \"config\": {\"growth\": " << json_number(baseline.config.growth)
        << ", \"chunk_bytes\": " << baseline.config.chunk_bytes
        << ", \"min_class_bytes\": " << baseline.config.min_class_bytes
        << ", \"max_class_bytes\": " << baseline.config.max_class_bytes
        << ", \"power_of_two\": " << (baseline.config.power_of_two ? "true" : "false")
        << ", \"extra_sizes\": [";
    for (std::size_t i = 0; i < baseline.config.extra_sizes.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << baseline.config.extra_sizes[i];
    }
    out << "]},\n";
    out << "  \"thresholds\": {\"max_peak_increase\": "
        << json_number(options.max_peak_increase) << ", \"max_waste_increase\": "
        << json_number(options.max_waste_increase) << "},\n";
    out << "  \"metrics\": [\n";
    for (std::size_t i = 0; i < metrics.size(); ++i) {
        const MetricComparison& metric = metrics[i];
        out << "    {\"name\": \"" << json_escape(metric.name) << "\", \"baseline\": "
            << json_number(metric.baseline) << ", \"current\": " << json_number(metric.current)
            << ", \"delta\": " << json_number(metric.delta) << ", \"relative\": ";
        if (metric.relative_defined) {
            out << json_number(metric.relative);
        } else {
            // null, not 0: writing 0 would let a consumer mistake "cannot tell" for "no change".
            out << "null";
        }
        out << "}" << (i + 1 == metrics.size() ? "" : ",") << "\n";
    }
    out << "  ],\n";
    out << "  \"notes\": [";
    for (std::size_t i = 0; i < notes.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << "\"" << json_escape(notes[i]) << "\"";
    }
    out << "]\n";
    out << "}\n";
    return out.str();
}

// Both sides come from traces with the same configuration, so a difference can only come from the trace.
inline Report report_from_trace(const std::string& path, std::string& error) {
    AdviseOptions options;
    options.trace_path = path;
    try {
        const Trace trace = parse_trace_file(path);
        AdviseResult result = advise(options, trace);
        result.trace.sha256 = sha256_file(path);
        Report report = report_for(result, result.reference);
        return report;
    } catch (const std::exception& exception) {
        error = exception.what();
        return Report{};
    }
}

inline int run_gate(const GateOptions& options, std::ostream& out, std::ostream& err) {
    const bool have_reports = !options.baseline_report.empty() || !options.current_report.empty();
    const bool have_traces = !options.baseline_trace.empty() || !options.current_trace.empty();

    if (have_reports && have_traces) {
        err << "poolkit_gate: give either two reports (--baseline/--current) or two traces "
               "(--baseline-trace/--current-trace), not both\n";
        return 2;
    }
    if (have_reports &&
        (options.baseline_report.empty() || options.current_report.empty())) {
        err << "poolkit_gate: --baseline and --current must be given together\n";
        return 2;
    }
    if (have_traces && (options.baseline_trace.empty() || options.current_trace.empty())) {
        err << "poolkit_gate: --baseline-trace and --current-trace must be given together\n";
        return 2;
    }
    if (!have_reports && !have_traces) {
        err << "poolkit_gate: no input: give two reports (--baseline/--current) or two traces\n";
        return 2;
    }

    Report baseline;
    Report current;
    GateResult result;

    // One place writes the document, so the "unreadable input" path also leaves a --json file behind.
    const auto emit = [&](const std::string& document) -> bool {
        if (!options.quiet) {
            out << document;
        }
        if (options.json_path.empty()) {
            return true;
        }
        std::ofstream file(options.json_path);
        if (!file) {
            err << "poolkit_gate: cannot write " << options.json_path << "\n";
            return false;
        }
        file << document;
        if (!file) {
            err << "poolkit_gate: failed while writing " << options.json_path << "\n";
            return false;
        }
        return true;
    };

    try {
        if (have_traces) {
            std::string error;
            baseline = report_from_trace(options.baseline_trace, error);
            if (!error.empty()) {
                err << "poolkit_gate: baseline trace: " << error << "\n";
                return 2;
            }
            current = report_from_trace(options.current_trace, error);
            if (!error.empty()) {
                err << "poolkit_gate: current trace: " << error << "\n";
                return 2;
            }
        } else {
            std::ifstream baseline_file(options.baseline_report);
            if (!baseline_file) {
                err << "poolkit_gate: cannot open " << options.baseline_report << "\n";
                return 2;
            }
            std::ostringstream buffer;
            buffer << baseline_file.rdbuf();
            baseline = from_json(buffer.str());

            std::ifstream current_file(options.current_report);
            if (!current_file) {
                err << "poolkit_gate: cannot open " << options.current_report << "\n";
                return 2;
            }
            std::ostringstream current_buffer;
            current_buffer << current_file.rdbuf();
            current = from_json(current_buffer.str());
        }
    } catch (const std::exception& exception) {
        // Unreadable input is not a verdict this tool can give: it is "I cannot tell", which is exit code 2.
        result.verdict = Verdict::suspect;
        result.reason = exception.what();
        if (!emit(result.to_json(options))) {
            return 2;
        }
        err << "poolkit_gate: suspect — " << result.reason << "\n";
        return 2;
    }

    result = compare_reports(baseline, current, options);

    if (!emit(result.to_json(options))) {
        return 2;
    }
    // The one-line verdict stays on stderr with --quiet: a CI log that says nothing is a log nobody reads.
    err << "poolkit_gate: " << verdict_name(result.verdict) << " — " << result.reason << "\n";
    return exit_code(result.verdict);
}

}  // namespace tools
}  // namespace poolkit

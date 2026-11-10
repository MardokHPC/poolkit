#pragma once

// The advisor: which size classes and which chunk should this trace use, and what does that
// choice cost? The columns are measured on this trace's own histogram; the peak comes from
// replaying the whole trace on a SimBackend, which is the reason the tool exists.

#include <poolkit/pool.hpp>
#include <poolkit/sim_backend.hpp>
#include <poolkit/size_classes.hpp>
#include <poolkit/trace.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "report_json.hpp"

namespace poolkit {
namespace tools {

inline std::string human_mib(std::size_t bytes) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(2)
        << static_cast<double>(bytes) / 1048576.0 << " MiB";
    return out.str();
}

inline std::string human_pct(double fraction) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(3) << fraction * 100.0 << " %";
    return out.str();
}

struct AdviseOptions {
    std::string trace_path = "traces/inference_transformer.jsonl";
    // The *reference*: what the user would pick by hand today. Measured too, because "better
    // than what you have" is the only useful form of advice.
    std::size_t chunk_bytes = std::size_t{2} << 20;
    std::size_t min_class_bytes = 256;
    std::size_t max_class_bytes = std::size_t{2} << 20;
    double growth = 1.5;
    bool power_of_two = true;
    std::vector<std::size_t> extra_sizes;
    std::size_t top = 10;
    std::string json_path;
};

struct Measurement {
    std::size_t classes = 0;
    double internal_waste = 0.0;
    double tail_waste = 0.0;
    std::size_t retention_bound_bytes = 0;
    std::size_t peak_bytes = 0;
    std::size_t peak_blocks = 0;
    double covered_bytes_fraction = 1.0;
};

struct Candidate {
    SizeClassTable::Config table_config;
    std::size_t chunk_bytes = 0;
    Measurement measurement;
};

// distinct_rows collapses growth factors that round to the same capacities; distinct_replays
// is the number of times the trace was driven through a Pool, which is the expensive part.
struct SweepSummary {
    std::size_t configurations = 0;
    std::size_t distinct_rows = 0;
    std::size_t distinct_replays = 0;
};

struct AdviseResult {
    TraceIdentity trace;
    std::vector<Candidate> ranked;   // best first, one entry per distinct table
    Candidate recommended;
    Candidate best_overall;
    Candidate reference;
    SweepSummary sweep;
    std::size_t alloc_requests = 0;
    std::size_t distinct_sizes = 0;
    std::size_t largest_request = 0;
};

namespace advise_detail {

using Histogram = std::map<std::size_t, std::size_t>;

inline Histogram build_histogram(const Trace& trace, std::size_t& allocations,
                                 std::size_t& largest) {
    Histogram histogram;
    allocations = 0;
    largest = 0;
    for (const TraceOp& op : trace.ops) {
        if (op.kind != TraceOpKind::alloc) {
            continue;
        }
        ++histogram[op.size];
        ++allocations;
        largest = std::max(largest, op.size);
    }
    return histogram;
}

inline std::string table_key(const SizeClassTable& table) {
    std::string key;
    key.reserve(table.count() * 8);
    for (std::size_t i = 0; i < table.count(); ++i) {
        key += std::to_string(table.class_size(i));
        key += ',';
    }
    return key;
}

inline Pool::Config pool_config_from(const SizeClassTable::Config& table_config,
                                     std::size_t chunk_bytes) {
    Pool::Config config;
    config.chunk_bytes = chunk_bytes;
    config.min_class_bytes = table_config.min_class_bytes;
    config.max_class_bytes = table_config.max_class_bytes;
    config.growth = table_config.growth;
    config.include_power_of_two = table_config.include_power_of_two;
    config.extra_sizes = table_config.extra_sizes;
    return config;
}

// Histogram integral, capacity-normalised: sum count(s)*(c(s)-s) / sum count(s)*c(s) over the
// requests at or below max_class. Above it they go to the large path, and this table does not round them.
inline double integrate_internal_waste(const SizeClassTable& table, const Histogram& histogram) {
    const std::size_t max_class = table.config().max_class_bytes;
    double waste = 0.0;
    double capacity = 0.0;
    for (const auto& entry : histogram) {
        if (entry.first > max_class) {
            continue;
        }
        const double requested = static_cast<double>(entry.first);
        const double counted = static_cast<double>(entry.second);
        const double class_bytes =
            static_cast<double>(table.class_size(table.index_for(entry.first)));
        waste += counted * (class_bytes - requested);
        capacity += counted * class_bytes;
    }
    return capacity > 0.0 ? waste / capacity : 0.0;
}

inline double integrate_tail_waste(const SizeClassTable& table, const Histogram& histogram,
                                   std::size_t chunk_bytes) {
    const std::size_t max_class = table.config().max_class_bytes;
    std::vector<std::pair<std::size_t, double>> weights;
    weights.reserve(table.count());
    std::vector<std::size_t> per_class(table.count(), 0);
    for (const auto& entry : histogram) {
        if (entry.first > max_class) {
            continue;
        }
        per_class[table.index_for(entry.first)] += entry.second;
    }
    for (std::size_t i = 0; i < table.count(); ++i) {
        weights.emplace_back(table.class_size(i), static_cast<double>(per_class[i]));
    }
    return expected_tail_waste(weights, chunk_bytes);
}

inline Measurement measure(const SizeClassTable& table, std::size_t chunk_bytes,
                           const Histogram& histogram, const Trace& trace) {
    Measurement result;
    result.classes = table.count();
    result.internal_waste = integrate_internal_waste(table, histogram);
    result.tail_waste = integrate_tail_waste(table, histogram, chunk_bytes);
    result.retention_bound_bytes = table.count() * chunk_bytes;

    const std::size_t max_class = table.config().max_class_bytes;
    double total_bytes = 0.0;
    double covered_bytes = 0.0;
    for (const auto& entry : histogram) {
        const double bytes = static_cast<double>(entry.first) * static_cast<double>(entry.second);
        total_bytes += bytes;
        if (entry.first <= max_class) {
            covered_bytes += bytes;
        }
    }
    result.covered_bytes_fraction = total_bytes > 0.0 ? covered_bytes / total_bytes : 1.0;

    SimBackend backend;
    Pool pool(backend, pool_config_from(table.config(), chunk_bytes));
    const TraceReplayResult replay = replay_trace(pool, trace);
    result.peak_bytes = replay.peak_bytes;
    result.peak_blocks = replay.peak_live_blocks;
    return result;
}

// Key over what the replay sees: two configs with the same capacities and the same chunk are
// one measurement, whatever growth produced them.
inline std::string measurement_key(const SizeClassTable& table, std::size_t chunk_bytes) {
    std::string key = table_key(table);
    key += '#';
    key += std::to_string(chunk_bytes);
    return key;
}

}  // namespace advise_detail

// Fixed sweep: powers of two on/off, growth 1.10..2.00 step 0.01, chunk 1/2/4/8 MiB, max class
// 1 or 2 MiB. min_class_bytes and extra_sizes are inputs of every table, not axes of the sweep.
inline AdviseResult advise(const AdviseOptions& options, const Trace& trace) {
    AdviseResult result;
    result.trace.path = options.trace_path;
    result.trace.ops = trace.ops.size();

    std::size_t largest = 0;
    const auto histogram = advise_detail::build_histogram(trace, result.alloc_requests, largest);
    result.largest_request = largest;
    result.trace.allocs = result.alloc_requests;
    result.distinct_sizes = histogram.size();

    const std::size_t chunks[] = {std::size_t{1} << 20, std::size_t{2} << 20,
                                  std::size_t{4} << 20, std::size_t{8} << 20};
    const std::size_t max_classes[] = {std::size_t{1} << 20, std::size_t{2} << 20};

    std::unordered_map<std::string, Measurement> cache;
    std::vector<Candidate> swept;

    const auto consider = [&](const SizeClassTable::Config& table_config, std::size_t chunk) {
        SizeClassTable table(table_config);  // throws on an impossible config
        std::string key = advise_detail::measurement_key(table, chunk);
        auto cached = cache.find(key);
        if (cached == cache.end()) {
            const Measurement measured = advise_detail::measure(table, chunk, histogram, trace);
            cached = cache.emplace(std::move(key), measured).first;
            ++result.sweep.distinct_replays;
        }
        Candidate candidate;
        candidate.table_config = table_config;
        candidate.chunk_bytes = chunk;
        candidate.measurement = cached->second;
        swept.push_back(std::move(candidate));
    };

    for (const bool power_of_two : {false, true}) {
        for (int step = 110; step <= 200; ++step) {
            SizeClassTable::Config table_config;
            table_config.min_class_bytes = options.min_class_bytes;
            table_config.growth = static_cast<double>(step) / 100.0;
            table_config.include_power_of_two = power_of_two;
            table_config.extra_sizes = options.extra_sizes;
            for (const std::size_t chunk : chunks) {
                for (const std::size_t max_class : max_classes) {
                    table_config.max_class_bytes = max_class;
                    consider(table_config, chunk);
                    ++result.sweep.configurations;
                }
            }
        }
    }

    // Always measured, even when the grid does not contain it: advice that cannot be compared
    // with what the user runs today is not advice.
    SizeClassTable::Config reference_config;
    reference_config.min_class_bytes = options.min_class_bytes;
    reference_config.max_class_bytes = options.max_class_bytes;
    reference_config.growth = options.growth;
    reference_config.include_power_of_two = options.power_of_two;
    reference_config.extra_sizes = options.extra_sizes;
    consider(reference_config, options.chunk_bytes);
    ++result.sweep.configurations;
    const Candidate reference_row = swept.back();

    // Collapse growth factors that round to the same table, keeping the smallest g: "the table is
    // what matters, and this is the least aggressive growth that gives it" is the useful statement.
    std::vector<Candidate> unique;
    unique.reserve(swept.size());
    std::unordered_map<std::string, std::size_t> seen;
    for (const Candidate& candidate : swept) {
        const SizeClassTable table(candidate.table_config);
        std::string key = advise_detail::table_key(table);
        key += '#';
        key += std::to_string(candidate.chunk_bytes);
        key += '#';
        key += candidate.table_config.include_power_of_two ? 'p' : 'g';
        const auto [it, inserted] = seen.emplace(std::move(key), unique.size());
        if (!inserted) {
            continue;  // identical table, identical chunk: identical row
        }
        unique.push_back(candidate);
    }
    result.sweep.distinct_rows = unique.size();

    std::sort(unique.begin(), unique.end(), [](const Candidate& a, const Candidate& b) {
        if (a.measurement.peak_bytes != b.measurement.peak_bytes) {
            return a.measurement.peak_bytes < b.measurement.peak_bytes;
        }
        const double waste_a = a.measurement.internal_waste + a.measurement.tail_waste;
        const double waste_b = b.measurement.internal_waste + b.measurement.tail_waste;
        if (waste_a != waste_b) {
            return waste_a < waste_b;
        }
        if (a.measurement.classes != b.measurement.classes) {
            return a.measurement.classes < b.measurement.classes;
        }
        if (a.table_config.growth != b.table_config.growth) {
            return a.table_config.growth > b.table_config.growth;
        }
        if (a.chunk_bytes != b.chunk_bytes) {
            return a.chunk_bytes < b.chunk_bytes;
        }
        if (a.table_config.max_class_bytes != b.table_config.max_class_bytes) {
            return a.table_config.max_class_bytes < b.table_config.max_class_bytes;
        }
        return a.table_config.include_power_of_two < b.table_config.include_power_of_two;
    });

    result.ranked = unique;
    result.best_overall = result.ranked.front();
    // The recommendation stays on the caller's max class: crossing it turns every larger request
    // into one driver call per block, never reused, which this byte-pricing model cannot see.
    result.recommended = reference_row;
    for (const Candidate& candidate : result.ranked) {
        if (candidate.table_config.max_class_bytes == reference_row.table_config.max_class_bytes) {
            result.recommended = candidate;
            break;
        }
    }
    result.reference = reference_row;
    return result;
}

inline Report report_for(const AdviseResult& result, const Candidate& candidate,
                         const std::string& tool_name = "poolkit_advise") {
    Report report;
    report.schema = kReportSchema;
    report.tool = tool_name;
    report.trace = result.trace;
    report.config.growth = candidate.table_config.growth;
    report.config.chunk_bytes = candidate.chunk_bytes;
    report.config.min_class_bytes = candidate.table_config.min_class_bytes;
    report.config.max_class_bytes = candidate.table_config.max_class_bytes;
    report.config.power_of_two = candidate.table_config.include_power_of_two;
    report.config.extra_sizes = candidate.table_config.extra_sizes;
    report.classes = candidate.measurement.classes;
    report.waste.internal = candidate.measurement.internal_waste;
    report.waste.tail = candidate.measurement.tail_waste;
    report.peak.resident_bytes = candidate.measurement.peak_bytes;
    report.peak.blocks = candidate.measurement.peak_blocks;
    report.retention_bound_bytes = candidate.measurement.retention_bound_bytes;
    return report;
}

inline std::string describe_config(const Candidate& candidate) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "g = " << std::fixed << std::setprecision(2) << candidate.table_config.growth
        << ", powers of two " << (candidate.table_config.include_power_of_two ? "on" : "off")
        << ", chunk " << human_mib(candidate.chunk_bytes) << ", max class "
        << human_mib(candidate.table_config.max_class_bytes);
    if (!candidate.table_config.extra_sizes.empty()) {
        out << ", extra sizes [";
        for (std::size_t i = 0; i < candidate.table_config.extra_sizes.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << candidate.table_config.extra_sizes[i] << " B";
        }
        out << ']';
    }
    return out.str();
}

inline std::string advise_markdown(const AdviseResult& result, std::size_t top) {
    std::ostringstream out;
    out.imbue(std::locale::classic());

    out << "poolkit_advise\n\n";
    out << "trace: `" << result.trace.path << "`  \n";
    out << "ops " << result.trace.ops << ", allocs " << result.trace.allocs << ", distinct sizes "
        << result.distinct_sizes << ", largest request " << result.largest_request << " B  \n";
    out << "sha256 `" << result.trace.sha256 << "`\n\n";
    out << "sweep: " << result.sweep.configurations << " configurations, "
        << result.sweep.distinct_rows << " distinct rows, "
        << result.sweep.distinct_replays << " replays\n\n";

    out << "reference: " << describe_config(result.reference) << "  \n";
    out << "k = " << result.reference.measurement.classes << ", internal "
        << human_pct(result.reference.measurement.internal_waste) << ", tail "
        << human_pct(result.reference.measurement.tail_waste) << ", retention "
        << human_mib(result.reference.measurement.retention_bound_bytes) << ", peak "
        << human_mib(result.reference.measurement.peak_bytes) << ", covered "
        << human_pct(result.reference.measurement.covered_bytes_fraction) << "\n\n";

    out << "| rank | growth | pow2 | chunk | max class | covered | k | internal | tail | "
           "retention k*C | peak sim | peak blocks |\n";
    out << "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    const std::size_t shown = std::min(top, result.ranked.size());
    for (std::size_t i = 0; i < shown; ++i) {
        const Candidate& candidate = result.ranked[i];
        out << "| " << (i + 1) << " | " << std::fixed << std::setprecision(2)
            << candidate.table_config.growth << " | "
            << (candidate.table_config.include_power_of_two ? "yes" : "no") << " | "
            << human_mib(candidate.chunk_bytes) << " | "
            << human_mib(candidate.table_config.max_class_bytes) << " | "
            << human_pct(candidate.measurement.covered_bytes_fraction) << " | "
            << candidate.measurement.classes << " | "
            << human_pct(candidate.measurement.internal_waste) << " | "
            << human_pct(candidate.measurement.tail_waste) << " | "
            << human_mib(candidate.measurement.retention_bound_bytes) << " | "
            << human_mib(candidate.measurement.peak_bytes) << " | "
            << candidate.measurement.peak_blocks << " |\n";
    }
    out << "\n" << shown << " of " << result.ranked.size()
        << " distinct rows, ordered by simulated peak. `covered` is the fraction of requested "
           "bytes this table rounds; the rest takes the large path.\n\n";

    const Candidate& best = result.recommended;
    const Candidate& base = result.reference;
    out << "## recommended\n\n";
    out << describe_config(best) << " -> k = " << best.measurement.classes << " classes\n\n";
    out << "peak simulated **" << human_mib(best.measurement.peak_bytes) << "** with "
        << best.measurement.peak_blocks << " blocks live at the peak; internal waste "
        << human_pct(best.measurement.internal_waste) << ", tail "
        << human_pct(best.measurement.tail_waste) << ", retention bound "
        << human_mib(best.measurement.retention_bound_bytes) << ", covered "
        << human_pct(best.measurement.covered_bytes_fraction) << "\n\n";
    out << "**why.** ";

    if (best.table_config.max_class_bytes == base.table_config.max_class_bytes) {
        out << "Lowest simulated peak among the configurations that keep max class at "
            << human_mib(base.table_config.max_class_bytes) << ": ";
    } else {
        out << "Lowest simulated peak of the sweep (it also changes max class, from "
            << human_mib(base.table_config.max_class_bytes) << " to "
            << human_mib(best.table_config.max_class_bytes) << "): ";
    }

    if (best.measurement.peak_bytes < base.measurement.peak_bytes) {
        out << human_mib(base.measurement.peak_bytes - best.measurement.peak_bytes)
            << " below the reference (" << human_mib(base.measurement.peak_bytes) << ")";
    } else if (best.measurement.peak_bytes == base.measurement.peak_bytes) {
        out << "the reference already reaches it ("
            << human_mib(base.measurement.peak_bytes) << "), and nothing in the sweep beats it";
    } else {
        out << "best found (" << human_mib(best.measurement.peak_bytes)
            << "), above the reference: the reference is outside the sweep grid or the grid "
               "misses its table";
    }

    const double internal_delta =
        (best.measurement.internal_waste - base.measurement.internal_waste) * 100.0;
    out << "; internal waste " << (internal_delta >= 0.0 ? "+" : "-") << std::fixed
        << std::setprecision(3) << (internal_delta >= 0.0 ? internal_delta : -internal_delta)
        << " pp vs the reference";

    const bool shoots_lower =
        best.measurement.retention_bound_bytes <= base.measurement.retention_bound_bytes;
    const std::size_t retention_delta =
        shoots_lower
            ? base.measurement.retention_bound_bytes - best.measurement.retention_bound_bytes
            : best.measurement.retention_bound_bytes - base.measurement.retention_bound_bytes;
    const long long class_delta = static_cast<long long>(best.measurement.classes) -
                                  static_cast<long long>(base.measurement.classes);
    out << "; worst-case retention " << human_mib(retention_delta) << (shoots_lower ? " less" : " more")
        << " than the reference, the price of " << (class_delta >= 0 ? "+" : "")
        << class_delta << " classes.\n";
    if (best.measurement.covered_bytes_fraction < 1.0) {
        out << "\n**caveat.** Only "
            << human_pct(best.measurement.covered_bytes_fraction)
            << " of the requested bytes are rounded by this table; the remaining "
            << human_pct(1.0 - best.measurement.covered_bytes_fraction)
            << " take the large path (one backend allocation per block, never reused), which is "
               "cheap in bytes and not in allocation time. Its internal waste is computed over "
               "the bytes it rounds, as the column says.\n";
    }
    if (result.best_overall.measurement.peak_bytes < best.measurement.peak_bytes) {
        out << "\n**overall best, not recommended.** " << describe_config(result.best_overall)
            << " reaches " << human_mib(result.best_overall.measurement.peak_bytes) << " ("
            << human_mib(best.measurement.peak_bytes -
                         result.best_overall.measurement.peak_bytes)
            << " less), but only "
            << human_pct(result.best_overall.measurement.covered_bytes_fraction)
            << " of the requested bytes would still be rounded by the table: the rest becomes "
               "one driver allocation per block, never reused. This model prices bytes, not "
               "driver calls, so the recommendation stays on your max class; tools/bench "
               "(M6b) is what measures the difference.\n";
    }
    return out.str();
}

// The whole CLI minus argument parsing: tests call this, main() calls this, so the exit code
// the test asserts is the exit code the binary returns.
inline int run_advise(const AdviseOptions& options, std::ostream& out, std::ostream& err,
                      AdviseResult* result_out = nullptr) {
    Trace trace;
    try {
        trace = parse_trace_file(options.trace_path);
    } catch (const std::exception& error) {
        err << "poolkit_advise: " << error.what() << "\n";
        return 2;
    }

    AdviseResult result;
    try {
        result = advise(options, trace);
        result.trace.sha256 = sha256_file(options.trace_path);
    } catch (const std::exception& error) {
        err << "poolkit_advise: " << error.what() << "\n";
        return 2;
    }

    out << advise_markdown(result, options.top);

    if (!options.json_path.empty()) {
        const Report report = report_for(result, result.recommended);
        std::ofstream file(options.json_path);
        if (!file) {
            err << "poolkit_advise: cannot write " << options.json_path << "\n";
            return 2;
        }
        file << to_json(report);
        if (!file) {
            err << "poolkit_advise: failed while writing " << options.json_path << "\n";
            return 2;
        }
    }
    if (result_out != nullptr) {
        *result_out = std::move(result);
    }
    return 0;
}

}  // namespace tools
}  // namespace poolkit

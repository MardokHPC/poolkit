// M5 measurement: the size class table before and after the "popular sizes" union (powers of two plus
// explicit extra sizes), same method as M4 so the two documents compare. Host-only: the numbers in
// the docs cannot drift from the code that ships.

#include <poolkit/pool.hpp>
#include <poolkit/size_classes.hpp>
#include <poolkit/slab.hpp>
#include <poolkit/trace.hpp>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMinClass = 256;
constexpr std::size_t kMaxClass = std::size_t{2} << 20;
constexpr std::size_t kChunkMin = std::size_t{2} << 20;
constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr double kGrowth = 1.5;

struct Variant {
    const char* name = "";
    bool include_power_of_two = false;
    bool extra_mib = false;
};

struct Measurement {
    std::size_t classes = 0;
    double internal = 0.0;
    double tail = 0.0;
    std::size_t retention = 0;
    std::size_t resident = 0;
    std::size_t high_water = 0;
    std::size_t replay_tail = 0;
    double replay_tail_frag = 0.0;
};

std::string fmt(double value, int decimals) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

std::string pct(double value) { return fmt(value * 100.0, 3) + "%"; }

std::string mib(std::size_t bytes) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(1) << static_cast<double>(bytes) / static_cast<double>(kMiB);
    return out.str() + " MiB";
}

poolkit::SizeClassTable make_table(const Variant& variant) {
    poolkit::SizeClassTable::Config cfg;
    cfg.min_class_bytes = kMinClass;
    cfg.max_class_bytes = kMaxClass;
    cfg.growth = kGrowth;
    cfg.include_power_of_two = variant.include_power_of_two;
    if (variant.extra_mib) {
        cfg.extra_sizes = {kMiB};
    }
    return poolkit::SizeClassTable(cfg);
}

poolkit::Pool::Config make_pool_config(const Variant& variant) {
    poolkit::Pool::Config cfg;
    cfg.chunk_bytes = kChunkMin;
    cfg.min_class_bytes = kMinClass;
    cfg.max_class_bytes = kMaxClass;
    cfg.growth = kGrowth;
    cfg.include_power_of_two = variant.include_power_of_two;
    if (variant.extra_mib) {
        cfg.extra_sizes = {kMiB};
    }
    return cfg;
}

Measurement measure(const poolkit::Trace& trace, const Variant& variant) {
    const poolkit::SizeClassTable table = make_table(variant);
    Measurement result;
    result.classes = table.count();

    std::vector<double> sizes;
    std::vector<std::size_t> per_class(table.count(), 0);
    for (const poolkit::TraceOp& op : trace.ops) {
        if (op.kind != poolkit::TraceOpKind::alloc || op.size > kMaxClass) {
            continue;
        }
        sizes.push_back(static_cast<double>(op.size));
        ++per_class[table.index_for(op.size)];
    }
    result.internal = table.expected_internal_waste(sizes);

    // Weighted by *this* workload's class mix, not one unit per class: a class nobody allocates from
    // retains nothing.
    std::vector<std::pair<std::size_t, double>> weights;
    weights.reserve(table.count());
    for (std::size_t i = 0; i < table.count(); ++i) {
        weights.emplace_back(table.class_size(i), static_cast<double>(per_class[i]));
    }
    result.tail = poolkit::expected_tail_waste(weights, kChunkMin);
    result.retention = table.count() * kChunkMin;

    // The M4 "measure instead of estimate" step: `resident` is what the pool still holds from the
    // driver, i.e. the retention bound with this workload's usage instead of the worst case.
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend, make_pool_config(variant));
    const poolkit::TraceReplayResult replay = poolkit::replay_trace(pool, trace);
    result.resident = replay.snapshot.bytes_resident;
    result.high_water = replay.snapshot.high_water_mark_bytes;
    result.replay_tail = replay.snapshot.tail_bytes;
    result.replay_tail_frag = replay.snapshot.tail_fragmentation;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: poolkit_measure_classes [trace.jsonl]\n";
        return 2;
    }
    const std::string path = argc == 2 ? argv[1] : "traces/inference_transformer.jsonl";

    poolkit::Trace trace;
    try {
        trace = poolkit::parse_trace_file(path);
    } catch (const poolkit::TraceError& error) {
        std::cerr << "poolkit_measure_classes: " << error.what() << "\n";
        return 1;
    }

    const Variant variants[] = {
        {"geometric only (M4)", false, false},
        {"powers of two (shipped)", true, false},
        {"geometric + 1 MiB", false, true},
        {"powers of two + 1 MiB", true, true},
    };

    std::cout << "trace: " << path << " (" << trace.ops.size() << " ops)\n\n";
    std::cout << "| variant | k | internal waste | expected tail | retention k*C | "
                 "replay resident | replay high water | replay tail | replay tail frag |\n";
    std::cout << "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const Variant& variant : variants) {
        const Measurement m = measure(trace, variant);
        std::cout << "| " << variant.name << " | " << m.classes << " | " << pct(m.internal)
                  << " | " << pct(m.tail) << " | " << mib(m.retention) << " | "
                  << mib(m.resident) << " | " << mib(m.high_water) << " | " << mib(m.replay_tail)
                  << " | " << pct(m.replay_tail_frag) << " |\n";
    }
    return 0;
}

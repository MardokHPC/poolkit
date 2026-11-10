// Sweep of the geometric growth factor against the two modelled waste components, using the headers
// the library ships (the published table in docs/cost-model-sweep.md is this program's output):
// tables A-E, plus F for the union with the powers of two.

#include <poolkit/size_classes.hpp>
#include <poolkit/slab.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
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
constexpr int kGrowthFirst = 110;  // 1.10
constexpr int kGrowthLast = 200;   // 2.00
constexpr int kGrowthStep = 1;     // 0.01
constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kIntegralSamples = 20000;

struct Row {
    double growth = 0.0;
    std::size_t classes = 0;
    double internal = 0.0;
    double tail = 0.0;

    POOLKIT_NODISCARD double total() const noexcept { return internal + tail; }
};

poolkit::SizeClassTable make_table(double growth) {
    poolkit::SizeClassTable::Config cfg;
    cfg.min_class_bytes = kMinClass;
    cfg.max_class_bytes = kMaxClass;
    cfg.growth = growth;
    // Tables A-E model the geometric progression the closed form describes; Table F measures the union.
    cfg.include_power_of_two = false;
    return poolkit::SizeClassTable(cfg);
}

poolkit::SizeClassTable make_union_table(double growth) {
    poolkit::SizeClassTable::Config cfg;
    cfg.min_class_bytes = kMinClass;
    cfg.max_class_bytes = kMaxClass;
    cfg.growth = growth;
    return poolkit::SizeClassTable(cfg);
}

std::string fmt(double value, int decimals) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

std::string pct(double value) { return fmt(value * 100.0, 3) + "%"; }

std::string mib(double bytes) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(1) << bytes / static_cast<double>(kMiB);
    return out.str();
}

std::vector<std::pair<std::size_t, double>> uniform_weights(const poolkit::SizeClassTable& table) {
    std::vector<std::pair<std::size_t, double>> weights;
    weights.reserve(table.count());
    for (std::size_t i = 0; i < table.count(); ++i) {
        weights.emplace_back(table.class_size(i), 1.0);
    }
    return weights;
}

// Log-uniform samples strictly inside the interval, so no sample lands on a class boundary.
std::vector<double> log_uniform_samples(std::size_t steps) {
    std::vector<double> samples;
    samples.reserve(steps);
    const double lo = static_cast<double>(kMinClass);
    const double hi = static_cast<double>(kMaxClass);
    const double log_ratio = std::log(hi / lo);
    for (std::size_t i = 0; i < steps; ++i) {
        const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(steps);
        samples.push_back(lo * std::exp(log_ratio * t));
    }
    return samples;
}

struct FrontierPoint {
    std::size_t live_bytes = 0;
    double growth = 0.0;
    std::size_t classes = 0;
    double internal = 0.0;
    double retention = 0.0;
    double total = 0.0;
};

std::vector<Row> run_sweep() {
    std::vector<Row> rows;
    rows.reserve(static_cast<std::size_t>((kGrowthLast - kGrowthFirst) / kGrowthStep + 1));
    for (int step = kGrowthFirst; step <= kGrowthLast; step += kGrowthStep) {
        const double growth = static_cast<double>(step) / 100.0;
        const auto table = make_table(growth);
        Row row;
        row.growth = growth;
        row.classes = table.count();
        row.internal = table.expected_internal_waste_log_uniform();
        row.tail = poolkit::expected_tail_waste(uniform_weights(table), kChunkMin);
        rows.push_back(row);
    }
    return rows;
}

// Ties keep the smallest growth, i.e. more classes: saying "g = 1.1" when 1.2 is equally good
// would hide the real trade-off instead of exposing it.
const Row& minimum_row(const std::vector<Row>& rows) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < rows.size(); ++i) {
        if (rows[i].total() < rows[best].total()) {
            best = i;
        }
    }
    return rows[best];
}

// The objective that does have an interior optimum: rounding waste scales with the live set,
// retention with the class count. suggest_growth() is the shipped form of this table.
FrontierPoint frontier_point(std::size_t live_bytes) {
    FrontierPoint point;
    point.live_bytes = live_bytes;
    point.growth = poolkit::suggest_growth(static_cast<double>(live_bytes), kChunkMin, kMinClass,
                                           kMaxClass);
    const auto table = make_table(point.growth);
    point.classes = table.count();
    point.internal =
        table.expected_internal_waste_log_uniform() * static_cast<double>(live_bytes);
    point.retention = static_cast<double>(point.classes * kChunkMin);
    point.total = point.internal + point.retention;
    return point;
}

std::string markdown(const std::vector<Row>& rows, const Row& best) {
    std::ostringstream out;
    out.imbue(std::locale::classic());

    out << "<!-- Generated by tools/sweep_growth.cpp; do not edit by hand. "
           "Regenerate with: cmake --build build && ./build/poolkit_sweep_growth "
           "docs/cost-model-sweep.md -->\n\n";
    out << "# Size class sweep\n\n";
    out << "Waste of the geometric table as a function of the growth factor, for the v1.0\n";
    out << "configuration. Derivation and interpretation: [cost-model.md](cost-model.md).\n\n";
    out << "| parameter | value |\n|---|---|\n";
    out << "| `s_min` (`min_class_bytes`) | " << kMinClass << " B |\n";
    out << "| `s_max` (`max_class_bytes`) | " << kMaxClass << " B (2 MiB) |\n";
    out << "| `chunk_min` (`chunk_bytes`) | " << kChunkMin << " B (2 MiB) |\n";
    out << "| growth range | " << fmt(static_cast<double>(kGrowthFirst) / 100.0, 2) << " to "
        << fmt(static_cast<double>(kGrowthLast) / 100.0, 2) << ", step "
        << fmt(static_cast<double>(kGrowthStep) / 100.0, 2) << " |\n";
    out << "| internal waste | closed form for log-uniform requests, normalised by assigned capacity |\n";
    out << "| tail waste | mean of `tail_i / chunk_i` over classes, one unit of resident weight each |\n\n";
    out << "Class counts come from the **rounded** table (`c_{i+1} = round_up(ceil(c_i·g), 256)`),\n";
    out << "not from the closed form: the alignment rounding merges geometric steps, so `k` at\n";
    out << "`g = 1.10` is " << make_table(1.10).count() << " and not the 96 of `ceil(log_g R) + 1`.\n\n";

    out << "## Table A — sweep\n\n";
    out << "| growth | classes | internal waste | tail waste | total |\n";
    out << "|---:|---:|---:|---:|---:|\n";
    for (const Row& row : rows) {
        out << "| " << fmt(row.growth, 2) << " | " << row.classes << " | " << pct(row.internal)
            << " | " << pct(row.tail) << " | " << pct(row.total()) << " |\n";
    }

    out << "\n## Minimum of `E[W_int] + E[W_tail]`\n\n";
    out << "| growth | classes | internal waste | tail waste | total |\n";
    out << "|---:|---:|---:|---:|---:|\n";
    out << "| " << fmt(best.growth, 2) << " | " << best.classes << " | " << pct(best.internal)
        << " | " << pct(best.tail) << " | " << pct(best.total()) << " |\n\n";
    out << "The minimum sits on the lower edge of the swept range: with resident weight spread\n";
    out << "evenly over the table, `E[W_int]` grows monotonically with `g` while `E[W_tail]` only\n";
    out << "oscillates (see Table A), so the sum has no interior optimum. The objective above\n";
    out << "therefore cannot pick 1.5 by itself — the justification for `g = 1.5` includes a term\n";
    out << "this sweep does not model: a class retains at least one resident chunk, so with\n";
    out << "`n` classes the retention bound is `n * chunk_min` bytes. At `g = 1.10`\n";
    out << "(`" << best.classes << "` classes) that bound is " << best.classes * kChunkMin / (std::size_t{1} << 20)
        << " MiB of an 8 GB card, at `g = 1.50` it is " << make_table(1.5).count() * kChunkMin / (std::size_t{1} << 20)
        << " MiB. Section 5 of cost-model.md works through the trade-off.\n";

    const auto chosen = make_table(1.5);
    out << "\n## Table B — geometric classes at `g = 1.50` (`k = " << chosen.count() << "`)\n\n";
    out << "The geometric progression alone, which is what the closed form describes. The table\n";
    out << "the library ships is the union with the powers of two — Table F quantifies the\n";
    out << "difference in `k`, `E_int` and retention.\n\n";
    out << chosen.to_markdown_table();

    out << "\n## Table C — slab layout at `g = 1.50`, `chunk_min = " << kChunkMin << "` B\n\n";
    out << "Geometric classes only. Every power of two in [256 B, 2 MiB] divides the 2 MiB\n";
    out << "chunk exactly, so the powers add classes with `tail = 0` and can only lower\n";
    out << "`E_tail`; a class above 2 MiB gets a dedicated chunk of its own size, also tail 0.\n\n";
    out << "| class | bytes | blocks per chunk | tail bytes | tail fraction |\n";
    out << "|---:|---:|---:|---:|---:|\n";
    for (std::size_t i = 0; i < chosen.count(); ++i) {
        const auto layout = poolkit::slab_layout(chosen.class_size(i), kChunkMin);
        out << "| " << i << " | " << layout.class_size << " | " << layout.blocks << " | "
            << layout.tail << " | " << pct(layout.tail_fraction()) << " |\n";
    }

    out << "\n## Table D — retention frontier, `Cost(g) = E_int(g)·L + k(g)·C`\n\n";
    out << "`k_used = k` in the shipped `suggest_growth()`: the worst case in which every class\n";
    out << "ends up holding at least one chunk. `C = " << kChunkMin << "` B.\n\n";
    out << "| live set | optimal `g` | `k` | `E_int` | internal waste | retention | total | total as % of live |\n";
    out << "|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    const std::vector<std::size_t> live_sets{64 * kMiB, 256 * kMiB, 1024 * kMiB};
    for (const std::size_t live : live_sets) {
        const FrontierPoint point = frontier_point(live);
        const double live_d = static_cast<double>(live);
        out << "| " << mib(live_d) << " MiB | " << fmt(point.growth, 2) << " | " << point.classes
            << " | "
            << pct(make_table(point.growth).expected_internal_waste_log_uniform()) << " | "
            << mib(point.internal) << " MiB | " << mib(point.retention) << " MiB | "
            << mib(point.total) << " MiB | " << pct(point.total / live_d) << " |\n";
    }

    out << "\n## Table E — closed form vs integral over the rounded table\n\n";
    out << "Log-uniform density over `[s_min, s_max]`, " << kIntegralSamples
        << " samples, capacity-normalised. `E_int` of Table A is the closed form; this is the\n";
    out << "number a log-uniform workload would actually see.\n\n";
    out << "| growth | `k` | closed form | integrated | difference |\n";
    out << "|---:|---:|---:|---:|---:|\n";
    const std::vector<double> samples = log_uniform_samples(kIntegralSamples);
    for (const double checkpoint : {1.10, 1.25, 1.50, 2.00}) {
        const auto table = make_table(checkpoint);
        const double closed = table.expected_internal_waste_log_uniform();
        const double integrated = table.expected_internal_waste(samples);
        out << "| " << fmt(checkpoint, 2) << " | " << table.count() << " | " << pct(closed)
            << " | " << pct(integrated) << " | " << fmt((integrated - closed) * 100.0, 3)
            << " pp |\n";
    }

    // Table F: the union with the powers of two buys exact fits at a cost in classes, i.e. in retention.
    out << "\n## Table F — geometric progression vs the shipped union (powers of two)\n\n";
    out << "`E_int` is the integral over the rounded table, log-uniform density,\n";
    out << "capacity-normalised (" << kIntegralSamples << " samples); `E_tail` weights each class\n";
    out << "equally. Retention is `k · chunk_min`, the §5.2 worst case.\n\n";
    out << "| growth | `k` geometric | `k` union | `E_int` geometric | `E_int` union | delta | `E_tail` union | retention geometric | retention union |\n";
    out << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const double checkpoint : {1.25, 1.50, 2.00}) {
        const auto geometric = make_table(checkpoint);
        const auto union_table = make_union_table(checkpoint);
        const double int_geometric = geometric.expected_internal_waste(samples);
        const double int_union = union_table.expected_internal_waste(samples);
        out << "| " << fmt(checkpoint, 2) << " | " << geometric.count() << " | "
            << union_table.count() << " | " << pct(int_geometric) << " | " << pct(int_union)
            << " | " << fmt((int_union - int_geometric) * 100.0, 3) << " pp | "
            << pct(poolkit::expected_tail_waste(uniform_weights(union_table), kChunkMin)) << " | "
            << geometric.count() * kChunkMin / kMiB << " MiB | "
            << union_table.count() * kChunkMin / kMiB << " MiB |\n";
    }
    const std::size_t extra_classes = make_union_table(1.50).count() - make_table(1.50).count();
    // Share of the geometric class count: retention grows from k to k + extra_classes, so the ratio is
    // extra_classes / k, not extra_classes / (k/2) which a doubled denominator would say.
    const double extra_class_share = static_cast<double>(extra_classes) * 100.0 /
                                     static_cast<double>(make_table(1.50).count());
    const double delta_150 = make_union_table(1.50).expected_internal_waste(samples) -
                             make_table(1.50).expected_internal_waste(samples);
    out << "\nThe powers of two never raise the integrated waste, and they lower it wherever the\n";
    out << "geometric table does not already contain them, but they add classes: at `g = 1.50`\n";
    out << "that is " << extra_classes << " extra classes (" << extra_classes * kChunkMin / kMiB
        << " MiB more worst-case retention, "
        << fmt(extra_class_share, 1)
        << " % more) for " << fmt(delta_150 * 100.0, 3) << " pp on a *log-uniform* density — which is\n";
    out << "the wrong density to judge them by. They pay on a workload that asks for powers of\n";
    out << "two, and `tools/measure_classes.cpp` measures exactly that on the committed trace.\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: poolkit_sweep_growth [output.md]\n";
        return 2;
    }

    const std::vector<Row> rows = run_sweep();
    const Row& best = minimum_row(rows);
    const std::string document = markdown(rows, best);

    if (argc == 2) {
        std::ofstream file(argv[1], std::ios::binary | std::ios::trunc);
        if (!file) {
            std::cerr << "poolkit_sweep_growth: cannot write " << argv[1] << "\n";
            return 1;
        }
        file << document;
        if (!file) {
            std::cerr << "poolkit_sweep_growth: write failed for " << argv[1] << "\n";
            return 1;
        }
        std::cout << "wrote " << argv[1] << " (" << rows.size() << " rows)\n";
    } else {
        std::cout << document;
    }

    std::cout.imbue(std::locale::classic());
    std::cout << "minimum total waste: g = " << fmt(best.growth, 2) << ", k = " << best.classes
              << ", internal " << pct(best.internal) << ", tail " << pct(best.tail) << ", total "
              << pct(best.total()) << "\n";
    for (const double checkpoint : {1.25, 1.5, 2.0}) {
        const auto table = make_table(checkpoint);
        std::cout << "g = " << fmt(checkpoint, 2) << ": k = " << table.count()
                  << ", internal " << pct(table.expected_internal_waste_log_uniform()) << "\n";
    }
    for (const std::size_t live : {64 * kMiB, 1024 * kMiB}) {
        std::cout << "suggest_growth(" << mib(static_cast<double>(live)) << " MiB, "
                  << kChunkMin << " B chunk) = "
                  << fmt(poolkit::suggest_growth(static_cast<double>(live), kChunkMin, kMinClass,
                                                 kMaxClass),
                         2)
                  << "\n";
    }
    return 0;
}

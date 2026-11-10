// The M6a tools: the advisor and the gate. The tests are the specification: determinism,
// "the advice is not worse than the reference", SimBackend peak == FakeBackend peak, four
// verdicts, and JSON that refuses NaN and survives a decimal-comma locale.

#include "advise.hpp"
#include "gate.hpp"
#include "report_json.hpp"

#include <poolkit/backend.hpp>
#include <poolkit/sim_backend.hpp>
#include <poolkit/trace.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace tools = poolkit::tools;

const char* kSmallTrace =
    "{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0}\n"
    "{\"op\":\"alloc\",\"size\":300,\"ptr\":1,\"stream\":0,\"ordinal\":1}\n"
    "{\"op\":\"alloc\",\"size\":1000,\"ptr\":2,\"stream\":0,\"ordinal\":2}\n"
    "{\"op\":\"free\",\"size\":0,\"ptr\":1,\"stream\":0,\"ordinal\":3}\n"
    "{\"op\":\"alloc\",\"size\":4096,\"ptr\":3,\"stream\":0,\"ordinal\":4}\n"
    "{\"op\":\"free\",\"size\":0,\"ptr\":0,\"stream\":0,\"ordinal\":5}\n"
    "{\"op\":\"free\",\"size\":0,\"ptr\":2,\"stream\":0,\"ordinal\":6}\n"
    "{\"op\":\"free\",\"size\":0,\"ptr\":3,\"stream\":0,\"ordinal\":7}\n";

// One pool-path allocation and one above max_class_bytes: the large path must reach the peak.
const char* kLargeTrace =
    "{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0}\n"
    "{\"op\":\"alloc\",\"size\":3145728,\"ptr\":1,\"stream\":0,\"ordinal\":1}\n"
    "{\"op\":\"free\",\"size\":0,\"ptr\":0,\"stream\":0,\"ordinal\":2}\n"
    "{\"op\":\"free\",\"size\":0,\"ptr\":1,\"stream\":0,\"ordinal\":3}\n";

std::string source_dir() {
#ifdef POOLKIT_SOURCE_DIR
    return POOLKIT_SOURCE_DIR;
#else
    return std::string();
#endif
}

std::filesystem::path scratch_dir() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "poolkit_tools_test";
    std::filesystem::create_directories(dir);
    return dir;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream out(path, std::ios::binary);
    out << contents;
}

// Built by hand so a test can move one number and see what the gate does.
tools::Report sample_report(const std::string& trace_name = "trace_a") {
    tools::Report report;
    report.schema = tools::kReportSchema;
    report.tool = "poolkit_advise";
    report.trace.path = trace_name + ".jsonl";
    report.trace.ops = 3076;
    report.trace.allocs = 1538;
    report.trace.sha256 = tools::sha256_hex(trace_name);
    report.classes = 34;
    report.waste.internal = 0.15234;
    report.waste.tail = 0.00543;
    report.peak.resident_bytes = std::size_t{68} << 20;
    report.peak.blocks = 1538;
    report.retention_bound_bytes = std::size_t{68} << 20;
    return report;
}

struct CommaPunct : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
};

}  // namespace

TEST(ToolsAdvise, TableAndJsonAreByteIdenticalAcrossRuns) {
    const poolkit::Trace trace = poolkit::parse_trace(kSmallTrace);
    tools::AdviseOptions options;
    options.trace_path = "small.jsonl";
    options.top = 5;

    const tools::AdviseResult first = tools::advise(options, trace);
    const tools::AdviseResult second = tools::advise(options, trace);

    const std::string table_a = tools::advise_markdown(first, options.top);
    const std::string table_b = tools::advise_markdown(second, options.top);
    EXPECT_EQ(table_a, table_b);
    EXPECT_NE(table_a.find("## recommended"), std::string::npos);

    const std::string json_a = tools::to_json(tools::report_for(first, first.recommended));
    const std::string json_b = tools::to_json(tools::report_for(second, second.recommended));
    EXPECT_EQ(json_a, json_b);

    EXPECT_EQ(json_a, tools::to_json(tools::from_json(json_a)));
}

// A number that can be checked by hand: 100, 300, 1000 and 4096 land in 256, 512, 1024 and
// 4096, so the waste is (156+212+24+0)/(256+512+1024+4096) = 392/5888.
TEST(ToolsAdvise, ReferenceWasteMatchesTheHandComputedValue) {
    const poolkit::Trace trace = poolkit::parse_trace(kSmallTrace);
    tools::AdviseOptions options;
    options.trace_path = "small.jsonl";
    const tools::AdviseResult result = tools::advise(options, trace);

    EXPECT_EQ(result.reference.table_config.growth, 1.5);
    EXPECT_TRUE(result.reference.table_config.include_power_of_two);
    EXPECT_EQ(result.reference.chunk_bytes, std::size_t{2} << 20);
    EXPECT_NEAR(result.reference.measurement.internal_waste, 392.0 / 5888.0, 1e-12);
    EXPECT_EQ(result.reference.measurement.classes, std::size_t{34});
    EXPECT_EQ(result.reference.measurement.retention_bound_bytes, std::size_t{34} << 21);
}

// The extra capacities must reach every swept table: a flag that only edits the header line
// would be a lie in the JSON.
TEST(ToolsAdvise, OptionsReachEverySweptTableAndTheReport) {
    const poolkit::Trace trace = poolkit::parse_trace(kSmallTrace);
    tools::AdviseOptions options;
    options.trace_path = "small.jsonl";
    options.growth = 1.25;
    options.chunk_bytes = std::size_t{4} << 20;
    options.max_class_bytes = std::size_t{2} << 20;
    options.power_of_two = false;
    options.extra_sizes = {std::size_t{1} << 20};

    const tools::AdviseResult result = tools::advise(options, trace);
    ASSERT_FALSE(result.ranked.empty());
    for (const tools::Candidate& candidate : result.ranked) {
        EXPECT_EQ(candidate.table_config.extra_sizes, options.extra_sizes);
        EXPECT_EQ(candidate.table_config.min_class_bytes, options.min_class_bytes);
    }
    EXPECT_EQ(result.reference.table_config.growth, 1.25);
    EXPECT_EQ(result.reference.chunk_bytes, std::size_t{4} << 20);
    EXPECT_FALSE(result.reference.table_config.include_power_of_two);
    EXPECT_EQ(result.reference.table_config.extra_sizes, options.extra_sizes);

    const tools::Report report = tools::report_for(result, result.reference);
    EXPECT_EQ(report.config.growth, 1.25);
    EXPECT_EQ(report.config.chunk_bytes, std::size_t{4} << 20);
    EXPECT_FALSE(report.config.power_of_two);
    ASSERT_EQ(report.config.extra_sizes.size(), std::size_t{1});
    EXPECT_EQ(report.config.extra_sizes.front(), std::size_t{1} << 20);
    EXPECT_EQ(tools::to_json(report), tools::to_json(tools::from_json(tools::to_json(report))));
}

TEST(ToolsAdvise, CommittedTraceRecommendationBeatsTheReferencePeak) {
    const std::string dir = source_dir();
    if (dir.empty()) {
        GTEST_SKIP() << "POOLKIT_SOURCE_DIR not defined at compile time";
    }
    const std::string trace_path = dir + "/traces/inference_transformer.jsonl";

    tools::AdviseOptions options;
    options.trace_path = trace_path;
    options.top = 10;
    options.json_path = (scratch_dir() / "advise_committed.json").string();

    std::ostringstream out;
    std::ostringstream err;
    tools::AdviseResult result;
    ASSERT_EQ(tools::run_advise(options, out, err, &result), 0) << err.str();

    EXPECT_LE(result.recommended.measurement.peak_bytes,
              result.reference.measurement.peak_bytes);
    // The sweep must be the full grid: the reference row is added on top.
    EXPECT_EQ(result.sweep.configurations, std::size_t{2 * 91 * 4 * 2 + 1});
    EXPECT_GT(result.sweep.distinct_rows, std::size_t{0});
    EXPECT_GT(result.trace.sha256.size(), std::size_t{0});
    EXPECT_TRUE(tools::is_sha256_hex(result.trace.sha256));

    const std::string document = read_file(options.json_path);
    EXPECT_FALSE(document.empty());
    const tools::Report report = tools::from_json(document);
    EXPECT_EQ(report.schema, tools::kReportSchema);
    EXPECT_EQ(report.tool, "poolkit_advise");
    EXPECT_EQ(report.trace.ops, result.trace.ops);
    EXPECT_EQ(report.trace.allocs, result.trace.allocs);
    EXPECT_EQ(report.trace.sha256, result.trace.sha256);
    EXPECT_EQ(report.classes, result.recommended.measurement.classes);
    EXPECT_EQ(report.peak.resident_bytes, result.recommended.measurement.peak_bytes);
    EXPECT_EQ(report.peak.blocks, result.recommended.measurement.peak_blocks);
    EXPECT_EQ(report.config.chunk_bytes, result.recommended.chunk_bytes);
    EXPECT_NEAR(report.waste.internal, result.recommended.measurement.internal_waste, 1e-6);
    EXPECT_GT(report.waste.internal, 0.0);
    EXPECT_GT(report.peak.resident_bytes, std::size_t{0});

    std::cout << "[advise] recommended: " << tools::describe_config(result.recommended)
              << " | peak " << tools::human_mib(result.recommended.measurement.peak_bytes)
              << " vs reference " << tools::human_mib(result.reference.measurement.peak_bytes)
              << "\n";
}

TEST(ToolsSimBackend, FakePointersAreAlignedAndUnique) {
    poolkit::SimBackend backend;
    void* a = backend.allocate_raw(1000);
    void* b = backend.allocate_raw(4096);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % poolkit::SimBackend::alignment, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b) % poolkit::SimBackend::alignment, 0u);
    EXPECT_NE(a, b);
    EXPECT_EQ(backend.bytes_resident(), std::size_t{5096});
    EXPECT_EQ(backend.chunk_count(), std::size_t{2});
    backend.free_raw(a);
    EXPECT_EQ(backend.bytes_resident(), std::size_t{4096});
    backend.free_raw(a);  // not ours any more: counted, never silently ignored
    EXPECT_EQ(backend.invalid_frees(), std::size_t{1});
    EXPECT_EQ(backend.chunk_count(), std::size_t{1});
}

namespace {

struct ReplayPair {
    poolkit::TraceReplayResult fake;
    poolkit::TraceReplayResult sim;
    std::size_t fake_chunks = 0;
    std::size_t fake_bytes = 0;
    std::size_t sim_chunks = 0;
    std::size_t sim_bytes = 0;
};

ReplayPair replay_both(const poolkit::Trace& trace,
                       poolkit::Pool::Config config = poolkit::Pool::Config{}) {
    ReplayPair pair;
    poolkit::FakeBackend fake_backend;
    poolkit::Pool fake_pool(fake_backend, config);
    pair.fake = poolkit::replay_trace(fake_pool, trace);
    pair.fake_chunks = fake_backend.chunk_count();
    pair.fake_bytes = fake_backend.bytes_resident();

    poolkit::SimBackend sim_backend;
    poolkit::Pool sim_pool(sim_backend, config);
    pair.sim = poolkit::replay_trace(sim_pool, trace);
    pair.sim_chunks = sim_backend.chunk_count();
    pair.sim_bytes = sim_backend.bytes_resident();
    return pair;
}

}  // namespace

TEST(ToolsSimBackend, SamePeakAsFakeBackend) {
    const ReplayPair pair = replay_both(poolkit::parse_trace(kSmallTrace));
    EXPECT_EQ(pair.sim.peak_bytes, pair.fake.peak_bytes);
    EXPECT_EQ(pair.sim.peak_live_blocks, pair.fake.peak_live_blocks);
    EXPECT_EQ(pair.sim.snapshot.bytes_resident, pair.fake.snapshot.bytes_resident);
    EXPECT_EQ(pair.sim.snapshot.high_water_mark_bytes, pair.fake.snapshot.high_water_mark_bytes);
    EXPECT_EQ(pair.sim.snapshot.growth_count, pair.fake.snapshot.growth_count);
    EXPECT_EQ(pair.sim.snapshot.reuse_count, pair.fake.snapshot.reuse_count);
    EXPECT_EQ(pair.sim.snapshot.tail_bytes, pair.fake.snapshot.tail_bytes);
}

TEST(ToolsSimBackend, PeakIncludesTheLargePath) {
    const ReplayPair pair = replay_both(poolkit::parse_trace(kLargeTrace));
    EXPECT_EQ(pair.sim.peak_bytes, pair.fake.peak_bytes);
    // The large path is not in bytes_resident, and a peak that missed it would under-report by 3 MiB.
    EXPECT_EQ(pair.sim.peak_bytes, std::size_t{5} << 20);
    EXPECT_EQ(pair.sim.snapshot.bytes_resident, std::size_t{2} << 20);
    EXPECT_EQ(pair.sim_chunks, pair.fake_chunks);
    EXPECT_EQ(pair.sim_bytes, pair.fake_bytes);
}

// 12.8 GiB of device memory analysed on a machine without 12.8 GiB of free host RAM; through
// FakeBackend the same replay would be a genuine 12.8 GiB allocation.
TEST(ToolsSimBackend, ReplaysATraceLargerThanHostRam) {
    constexpr int blocks = 200;
    constexpr std::size_t block_bytes = std::size_t{64} << 20;  // above max class
    std::string text;
    for (int i = 0; i < blocks; ++i) {
        text += "{\"op\":\"alloc\",\"size\":" + std::to_string(block_bytes) +
                ",\"ptr\":" + std::to_string(i) + ",\"stream\":0,\"ordinal\":" +
                std::to_string(i) + "}\n";
    }
    const poolkit::Trace trace = poolkit::parse_trace(text);

    poolkit::SimBackend backend;
    poolkit::Pool pool(backend);
    const poolkit::TraceReplayResult replay = poolkit::replay_trace(pool, trace);

    EXPECT_EQ(replay.peak_bytes, static_cast<std::size_t>(blocks) * block_bytes);
    EXPECT_EQ(replay.peak_live_blocks, static_cast<std::size_t>(blocks));
    EXPECT_EQ(replay.snapshot.large_path_count, static_cast<std::size_t>(blocks));
    EXPECT_EQ(replay.snapshot.bytes_resident, std::size_t{0});
    EXPECT_EQ(backend.bytes_resident(), std::size_t{0});
    EXPECT_EQ(backend.invalid_frees(), std::size_t{0});
}

TEST(ToolsGate, PassWhenNothingMoves) {
    const tools::GateOptions options;
    const tools::Report baseline = sample_report();
    const tools::GateResult result = tools::compare_reports(baseline, baseline, options);
    EXPECT_EQ(result.verdict, tools::Verdict::pass);
    EXPECT_EQ(tools::exit_code(result.verdict), 0);
}

TEST(ToolsGate, RegressionAtTenPercentPeakGrowth) {
    const tools::GateOptions options;
    const tools::Report baseline = sample_report();
    tools::Report current = baseline;
    current.peak.resident_bytes = baseline.peak.resident_bytes + baseline.peak.resident_bytes / 10;
    const tools::GateResult result = tools::compare_reports(baseline, current, options);
    EXPECT_EQ(result.verdict, tools::Verdict::regression);
    EXPECT_EQ(tools::exit_code(result.verdict), 1);
    EXPECT_NE(result.reason.find("peak_resident_bytes"), std::string::npos);
}

TEST(ToolsGate, ImprovementAtTenPercentPeakDrop) {
    const tools::GateOptions options;
    const tools::Report baseline = sample_report();
    tools::Report current = baseline;
    current.peak.resident_bytes = baseline.peak.resident_bytes - baseline.peak.resident_bytes / 10;
    const tools::GateResult result = tools::compare_reports(baseline, current, options);
    EXPECT_EQ(result.verdict, tools::Verdict::improvement);
    EXPECT_EQ(tools::exit_code(result.verdict), 0);
}

TEST(ToolsGate, BelowThresholdIsAPass) {
    const tools::GateOptions options;
    const tools::Report baseline = sample_report();
    tools::Report current = baseline;
    current.peak.resident_bytes = baseline.peak.resident_bytes +
                                  baseline.peak.resident_bytes / 100;  // +1 % < 5 %
    current.waste.tail = baseline.waste.tail + 0.01;                    // +1 pp < 2 pp
    const tools::GateResult result = tools::compare_reports(baseline, current, options);
    EXPECT_EQ(result.verdict, tools::Verdict::pass);
}

TEST(ToolsGate, WasteRegressionIsRegressionToo) {
    const tools::GateOptions options;
    const tools::Report baseline = sample_report();
    tools::Report current = baseline;
    current.waste.internal = baseline.waste.internal + 0.03;  // +3 pp > 2 pp
    const tools::GateResult result = tools::compare_reports(baseline, current, options);
    EXPECT_EQ(result.verdict, tools::Verdict::regression);
    EXPECT_EQ(tools::exit_code(result.verdict), 1);
    EXPECT_NE(result.reason.find("internal_waste"), std::string::npos);
}

TEST(ToolsGate, DifferentTraceIsSuspectNotRegression) {
    const tools::GateOptions options;
    const tools::Report baseline = sample_report("trace_a");
    tools::Report current = sample_report("trace_b");
    // The honest answer is "these are not comparable", never "improvement".
    current.peak.resident_bytes = baseline.peak.resident_bytes / 2;
    const tools::GateResult result = tools::compare_reports(baseline, current, options);
    EXPECT_EQ(result.verdict, tools::Verdict::suspect);
    EXPECT_EQ(tools::exit_code(result.verdict), 2);
    EXPECT_NE(result.reason.find("different trace"), std::string::npos);
}

TEST(ToolsGate, DifferentConfigIsSuspect) {
    const tools::GateOptions options;
    const tools::Report baseline = sample_report();
    tools::Report current = baseline;
    current.config.growth = 1.1;
    const tools::GateResult result = tools::compare_reports(baseline, current, options);
    EXPECT_EQ(result.verdict, tools::Verdict::suspect);
    EXPECT_EQ(tools::exit_code(result.verdict), 2);
    EXPECT_NE(result.reason.find("different configuration"), std::string::npos);
}

TEST(ToolsGate, UnknownSchemaIsSuspect) {
    const tools::GateOptions options;
    const tools::Report baseline = sample_report();
    tools::Report current = baseline;
    current.schema = 2;
    const tools::GateResult result = tools::compare_reports(baseline, current, options);
    EXPECT_EQ(result.verdict, tools::Verdict::suspect);
}

TEST(ToolsGate, TraceIdentityFallsBackToCountersAndSaysSo) {
    const tools::GateOptions options;
    const tools::Report baseline = sample_report();
    tools::Report current = baseline;
    current.trace.sha256.clear();
    const tools::GateResult result = tools::compare_reports(baseline, current, options);
    EXPECT_EQ(result.verdict, tools::Verdict::pass);
    ASSERT_FALSE(result.notes.empty());
    EXPECT_NE(result.notes.front().find("ops/allocs"), std::string::npos);
}

TEST(ToolsGate, FilesAndExitCodesThroughTheCliPath) {
    const std::filesystem::path dir = scratch_dir();
    const tools::Report baseline = sample_report();
    tools::Report regression = baseline;
    regression.peak.resident_bytes = baseline.peak.resident_bytes * 11 / 10;

    write_file(dir / "gate_baseline.json", tools::to_json(baseline));
    write_file(dir / "gate_regression.json", tools::to_json(regression));

    tools::GateOptions options;
    options.baseline_report = (dir / "gate_baseline.json").string();
    options.current_report = (dir / "gate_regression.json").string();
    options.json_path = (dir / "gate_verdict.json").string();
    options.quiet = true;

    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(tools::run_gate(options, out, err), 1);
    EXPECT_TRUE(out.str().empty());  // --quiet
    EXPECT_NE(err.str().find("regression"), std::string::npos);

    const std::string verdict_document = read_file(options.json_path);
    EXPECT_NE(verdict_document.find("\"verdict\": \"regression\""), std::string::npos);
    EXPECT_NE(verdict_document.find("\"tool\": \"poolkit_gate\""), std::string::npos);

    options.current_report = options.baseline_report;
    EXPECT_EQ(tools::run_gate(options, out, err), 0);
}

TEST(ToolsGate, UnreadableReportIsSuspectAndExitTwo) {
    const std::filesystem::path dir = scratch_dir();
    write_file(dir / "gate_broken.json", "{\"schema\": 1, \"tool\": \n");

    tools::GateOptions options;
    options.baseline_report = (dir / "gate_broken.json").string();
    options.current_report = (dir / "gate_broken.json").string();
    options.json_path = (dir / "gate_unreadable_verdict.json").string();
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(tools::run_gate(options, out, err), 2);
    EXPECT_NE(err.str().find("suspect"), std::string::npos);
    const std::string verdict_document = read_file(options.json_path);
    EXPECT_NE(verdict_document.find("\"verdict\": \"suspect\""), std::string::npos);
}

TEST(ToolsGate, TraceModeUsesTheSameConfigurationOnBothSides) {
    const std::filesystem::path dir = scratch_dir();
    write_file(dir / "gate_trace_a.jsonl", kSmallTrace);
    write_file(dir / "gate_trace_b.jsonl", kSmallTrace);

    tools::GateOptions options;
    options.baseline_trace = (dir / "gate_trace_a.jsonl").string();
    options.current_trace = (dir / "gate_trace_b.jsonl").string();
    options.quiet = true;
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(tools::run_gate(options, out, err), 0);
    EXPECT_NE(err.str().find("pass"), std::string::npos);
}

TEST(ToolsJson, Sha256MatchesTheKnownVectors) {
    EXPECT_EQ(tools::sha256_hex(""),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(tools::sha256_hex("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(tools::sha256_hex(std::string(1000, 'a')),
              "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

TEST(ToolsJson, SerialisationAndParsingIgnoreADecimalCommaLocale) {
    const std::locale previous =
        std::locale::global(std::locale(std::locale::classic(), new CommaPunct));

    const tools::Report report = sample_report();
    const std::string document = tools::to_json(report);
    double growth_read_back = 0.0;
    std::size_t classes_read_back = 0;
    try {
        const tools::Report parsed = tools::from_json(document);
        growth_read_back = parsed.config.growth;
        classes_read_back = parsed.classes;
    } catch (const std::exception&) {
        std::locale::global(previous);
        FAIL() << "the report did not parse under a decimal-comma locale";
    }
    std::locale::global(previous);

    // A stream that forgot to imbue classic would write "1,5" and the document would be invalid JSON.
    EXPECT_NE(document.find("\"growth\": 1.5"), std::string::npos);
    EXPECT_EQ(document.find("1,5"), std::string::npos);
    EXPECT_NE(document.find("\"internal\": 0.15234"), std::string::npos);
    EXPECT_DOUBLE_EQ(growth_read_back, 1.5);
    EXPECT_EQ(classes_read_back, report.classes);
}

TEST(ToolsJson, NonFiniteNumbersAreRefusedOnWrite) {
    tools::Report report = sample_report();
    report.waste.internal = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(tools::to_json(report), tools::ReportError);

    report = sample_report();
    report.waste.tail = std::numeric_limits<double>::infinity();
    EXPECT_THROW(tools::to_json(report), tools::ReportError);

    report = sample_report();
    report.config.growth = -std::numeric_limits<double>::infinity();
    EXPECT_THROW(tools::to_json(report), tools::ReportError);
}

TEST(ToolsJson, NonFiniteNumbersAreRefusedOnRead) {
    std::string document = tools::to_json(sample_report());
    const std::string good = "\"internal\": 0.15234";
    ASSERT_NE(document.find(good), std::string::npos);
    document.replace(document.find(good), good.size(), "\"internal\": NaN");
    EXPECT_THROW(tools::from_json(document), tools::ReportError);

    document = tools::to_json(sample_report());
    const std::string growth = "\"growth\": 1.5";
    ASSERT_NE(document.find(growth), std::string::npos);
    document.replace(document.find(growth), growth.size(), "\"growth\": 1e999");
    EXPECT_THROW(tools::from_json(document), tools::ReportError);
}

TEST(ToolsJson, UnknownFieldsAndSchemasAreRejectedWithAReason) {
    std::string document = tools::to_json(sample_report());
    document.replace(document.find("\"classes\""), 9, "\"class_count\"");
    EXPECT_THROW(tools::from_json(document), tools::ReportError);

    document = tools::to_json(sample_report());
    document.replace(document.find("\"schema\": 1"), 11, "\"schema\": 2");
    EXPECT_THROW(tools::from_json(document), tools::ReportError);
}

TEST(ToolsJson, IntegerFieldsRefuseValuesADoubleCannotHoldExactly) {
    // 2^53 and 2^64 both used to pass: the old bound 18446744073709551615.0 *is* 2^64 once
    // stored in a double, so the cast to size_t that followed was undefined behaviour.
    for (const char* huge : {"9007199254740992", "18446744073709551616"}) {
        std::string document = tools::to_json(sample_report());
        const std::string good = "\"classes\": 34";
        ASSERT_NE(document.find(good), std::string::npos);
        document.replace(document.find(good), good.size(),
                          std::string("\"classes\": ") + huge);
        try {
            (void)tools::from_json(document);
            FAIL() << "classes = " << huge << " was accepted";
        } catch (const tools::ReportError& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("classes"), std::string::npos) << message;
            EXPECT_NE(message.find("2^53"), std::string::npos) << message;
        }
    }

    std::string document = tools::to_json(sample_report());
    const std::string empty = "\"extra_sizes\": []";
    ASSERT_NE(document.find(empty), std::string::npos);
    document.replace(document.find(empty), empty.size(),
                      "\"extra_sizes\": [9007199254740992]");
    EXPECT_THROW(tools::from_json(document), tools::ReportError);

    tools::Report report = sample_report();
    report.peak.resident_bytes = std::size_t{8} << 30;          // 8 GiB
    report.retention_bound_bytes = (std::size_t{1} << 53) - 1;  // 2^53 - 1
    const tools::Report parsed = tools::from_json(tools::to_json(report));
    EXPECT_EQ(parsed.peak.resident_bytes, report.peak.resident_bytes);
    EXPECT_EQ(parsed.retention_bound_bytes, report.retention_bound_bytes);
}

TEST(ToolsJson, EscapesSurviveTheRoundTrip) {
    tools::Report report = sample_report();
    report.trace.path = "C:\\traces\\a \"quoted\"\ttrace.jsonl";
    report.config.extra_sizes = {1048576, 2097152};
    const tools::Report parsed = tools::from_json(tools::to_json(report));
    EXPECT_EQ(parsed.trace.path, report.trace.path);
    ASSERT_EQ(parsed.config.extra_sizes.size(), std::size_t{2});
    EXPECT_EQ(parsed.config.extra_sizes[0], std::size_t{1} << 20);
    EXPECT_EQ(parsed.config.extra_sizes[1], std::size_t{2} << 20);
}

// A median below the min, a split that does not add up to the allocation count, or a sampled
// peak with no period are refusals, not warnings.

tools::Report bench_report() {
    tools::Report report = sample_report();
    report.schema = tools::kBenchReportSchema;
    report.tool = "poolkit_bench";
    report.has_bench = true;
    report.bench.schema = tools::kBenchSectionSchema;
    report.bench.env.gpu = "Fake GPU";
    report.bench.env.compute = "sm_75";
    report.bench.env.driver_version = 13030;
    report.bench.env.runtime_version = 12060;
    report.bench.env.arch = "x86_64";
    report.bench.env.chunk_bytes = std::size_t{2} << 20;
    report.bench.env.growth = 1.5;
    report.bench.env.min_class_bytes = 256;
    report.bench.env.max_class_bytes = std::size_t{2} << 20;
    report.bench.env.warmup = 1;
    report.bench.env.repeat = 5;

    tools::BenchMode pool;
    pool.mode = "pool";
    pool.ops = 6;
    pool.allocs = 3;
    pool.frees = 3;
    pool.wall_ns = {1000.0, 1100.0, 900.0, 1200.0};
    pool.peak_bytes = std::size_t{4} << 20;
    pool.alloc_ns = {400.0, 500.0, 300.0, 900.0};
    pool.free_ns = {600.0, 800.0, 500.0, 900.0};
    pool.hit_ns = {350.0, 400.0, 300.0, 500.0};
    pool.hit_ops = 2;
    pool.growth_ns = {60000.0, 70000.0, 50000.0, 80000.0};
    pool.growth_ops = 1;
    // The one growth op was a chunk, not a slot: the split has to add up, and the reader checks it.
    pool.slot_ns = {};
    pool.slot_ops = 0;
    pool.chunk_ns = {60000.0, 70000.0, 50000.0, 80000.0};
    pool.chunk_ops = 1;
    pool.large_ns = {};
    pool.large_ops = 0;
    report.bench.modes.push_back(pool);

    tools::BenchMode driver;
    driver.mode = "cudamalloc";
    driver.ops = 6;
    driver.allocs = 3;
    driver.frees = 3;
    driver.wall_ns = {10000.0, 11000.0, 9000.0, 12000.0};
    driver.peak_bytes = std::size_t{3} << 20;
    driver.peak_sampled = true;
    driver.sample_every_ops = 64;
    driver.alloc_ns = {2300.0, 50000.0, 2000.0, 60000.0};
    driver.free_ns = {1800.0, 80000.0, 1500.0, 90000.0};
    report.bench.modes.push_back(driver);
    return report;
}

TEST(ToolsJson, BenchSectionRoundTripsByteForByte) {
    const tools::Report report = bench_report();
    const std::string document = tools::to_json(report);
    EXPECT_NE(document.find("\"schema\": 2"), std::string::npos);
    EXPECT_NE(document.find("\"bench\""), std::string::npos);

    const tools::Report parsed = tools::from_json(document);
    EXPECT_TRUE(parsed.has_bench);
    EXPECT_EQ(parsed.schema, tools::kBenchReportSchema);
    ASSERT_EQ(parsed.bench.modes.size(), std::size_t{2});
    EXPECT_EQ(parsed.bench.modes[0].mode, "pool");
    EXPECT_EQ(parsed.bench.modes[0].hit_ops, std::size_t{2});
    EXPECT_EQ(parsed.bench.modes[0].chunk_ops, std::size_t{1});
    EXPECT_TRUE(parsed.bench.modes[1].peak_sampled);
    EXPECT_EQ(parsed.bench.modes[1].sample_every_ops, std::size_t{64});
    // The document is a contract: reading it and writing it again is the same bytes.
    EXPECT_EQ(tools::to_json(parsed), document);
}

TEST(ToolsJson, InconsistentBenchSectionsAreRefused) {
    // A bench section in a schema-1 document: the version must describe the contents.
    tools::Report wrong = sample_report();
    wrong.has_bench = true;
    EXPECT_THROW(tools::to_json(wrong), tools::ReportError);

    std::string document = tools::to_json(bench_report());
    const std::string hit = "\"hit_ops\": 2";
    ASSERT_NE(document.find(hit), std::string::npos);
    document.replace(document.find(hit), hit.size(), "\"hit_ops\": 3");
    EXPECT_THROW(tools::from_json(document), tools::ReportError);

    // A sampled peak with a zero period.
    document = tools::to_json(bench_report());
    const std::string period = "\"sample_every_ops\": 64";
    ASSERT_NE(document.find(period), std::string::npos);
    document.replace(document.find(period), period.size(), "\"sample_every_ops\": 0");
    EXPECT_THROW(tools::from_json(document), tools::ReportError);

    // An unordered distribution (p99 below the median).
    document = tools::to_json(bench_report());
    const std::string wall = "\"median\": 1000, \"p99\": 1100";
    ASSERT_NE(document.find(wall), std::string::npos);
    document.replace(document.find(wall), wall.size(), "\"median\": 1000, \"p99\": 900");
    EXPECT_THROW(tools::from_json(document), tools::ReportError);
}

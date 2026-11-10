// Trace reader and replay: deterministic, and strict about invalid input. Determinism is the
// load-bearing property: two replays of one trace must agree on every counter, or the
// measurement is a lottery.

#include <poolkit/size_classes.hpp>
#include <poolkit/trace.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

const char* committed_trace_path() {
#ifdef POOLKIT_SOURCE_DIR
    return POOLKIT_SOURCE_DIR "/traces/inference_transformer.jsonl";
#else
    return nullptr;
#endif
}

// A valid, minimal three-op trace (alloc, alloc, free).
const char* kValidTrace =
    "{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0}\n"
    "{\"op\":\"alloc\",\"size\":300,\"ptr\":1,\"stream\":0,\"ordinal\":1}\n"
    "{\"op\":\"free\",\"size\":0,\"ptr\":1,\"stream\":0,\"ordinal\":2}\n";

}  // namespace

TEST(Trace, ParsesAValidTrace) {
    const poolkit::Trace trace = poolkit::parse_trace(kValidTrace);
    ASSERT_EQ(trace.ops.size(), std::size_t{3});
    EXPECT_EQ(trace.ops[0].kind, poolkit::TraceOpKind::alloc);
    EXPECT_EQ(trace.ops[0].size, std::size_t{100});
    EXPECT_EQ(trace.ops[0].ptr, std::uint64_t{0});
    EXPECT_EQ(trace.ops[1].kind, poolkit::TraceOpKind::alloc);
    EXPECT_EQ(trace.ops[2].kind, poolkit::TraceOpKind::free);
    EXPECT_EQ(trace.ops[2].ordinal, std::uint64_t{2});
}

TEST(Trace, AcceptsAnEmptySizeOnFree) {
    // A free carries size 0; the parser must not insist on a positive size there.
    const poolkit::Trace trace = poolkit::parse_trace(kValidTrace);
    EXPECT_EQ(trace.ops[2].size, std::size_t{0});
}

TEST(Trace, MalformedJsonIsRejected) {
    EXPECT_THROW(poolkit::parse_trace("{\"op\":\"alloc\",\"size\":100"),
                 poolkit::TraceError);
    EXPECT_THROW(poolkit::parse_trace("not json at all"), poolkit::TraceError);
    EXPECT_THROW(poolkit::parse_trace(
                     "{\"op\":\"alloc\",\"size\":\"x\",\"ptr\":0,\"stream\":0,\"ordinal\":0}"),
                 poolkit::TraceError);
    EXPECT_THROW(poolkit::parse_trace(
                     "{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0} trailing"),
                 poolkit::TraceError);
}

TEST(Trace, MissingFieldIsRejected) {
    // Every field is mandatory: a missing one is an invalid trace, not a default.
    EXPECT_THROW(poolkit::parse_trace("{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0}"),
                 poolkit::TraceError);
    EXPECT_THROW(poolkit::parse_trace("{\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0}"),
                 poolkit::TraceError);
    EXPECT_THROW(poolkit::parse_trace("{\"op\":\"alloc\",\"ptr\":0,\"stream\":0,\"ordinal\":0}"),
                 poolkit::TraceError);
}

TEST(Trace, UnknownOpIsRejected) {
    EXPECT_THROW(poolkit::parse_trace(
                     "{\"op\":\"frob\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0}"),
                 poolkit::TraceError);
}

TEST(Trace, ZeroSizedAllocIsRejected) {
    EXPECT_THROW(poolkit::parse_trace(
                     "{\"op\":\"alloc\",\"size\":0,\"ptr\":0,\"stream\":0,\"ordinal\":0}"),
                 poolkit::TraceError);
}

TEST(Trace, NonMonotonicOrdinalIsRejected) {
    const char* trace =
        "{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":1}\n"
        "{\"op\":\"free\",\"size\":0,\"ptr\":0,\"stream\":0,\"ordinal\":0}\n";
    EXPECT_THROW(poolkit::parse_trace(trace), poolkit::TraceError);
    const char* repeated =
        "{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0}\n"
        "{\"op\":\"free\",\"size\":0,\"ptr\":0,\"stream\":0,\"ordinal\":0}\n";
    EXPECT_THROW(poolkit::parse_trace(repeated), poolkit::TraceError);
}

TEST(Trace, EmptyTraceIsRejected) {
    EXPECT_THROW(poolkit::parse_trace(""), poolkit::TraceError);
}

TEST(Trace, IntegerOverflowIsRejectedWithTheLineNumber) {
    // 2^64 - 1 fits the field; 2^64 and above must be refused instead of wrapping, as a size of
    // 18446744073709551620 once did.
    const poolkit::Trace at_the_limit = poolkit::parse_trace(
        "{\"op\":\"alloc\",\"size\":100,\"ptr\":18446744073709551615,\"stream\":0,\"ordinal\":0}\n");
    ASSERT_EQ(at_the_limit.ops.size(), std::size_t{1});
    EXPECT_EQ(at_the_limit.ops[0].ptr, std::numeric_limits<std::uint64_t>::max());

    EXPECT_THROW(poolkit::parse_trace(
                     "{\"op\":\"alloc\",\"size\":100,\"ptr\":18446744073709551616,"
                     "\"stream\":0,\"ordinal\":0}\n"),
                 poolkit::TraceError);

    const char* wrapped =
        "{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0}\n"
        "{\"op\":\"free\",\"size\":0,\"ptr\":0,\"stream\":0,\"ordinal\":1}\n"
        "{\"op\":\"alloc\",\"size\":18446744073709551620,\"ptr\":2,\"stream\":0,\"ordinal\":2}\n";
    try {
        const poolkit::Trace parsed = poolkit::parse_trace(wrapped);
        FAIL() << "a size of 18446744073709551620 was accepted as "
               << (parsed.ops.size() > 2 ? parsed.ops[2].size : std::size_t{0});
    } catch (const poolkit::TraceError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("line 3"), std::string::npos) << message;
        EXPECT_NE(message.find("too large"), std::string::npos) << message;
    }
}

TEST(Trace, FreeOfAnInactivePtrIsRejected) {
    const poolkit::Trace trace = poolkit::parse_trace(
        "{\"op\":\"free\",\"size\":0,\"ptr\":5,\"stream\":0,\"ordinal\":0}\n");
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend);
    EXPECT_THROW(poolkit::replay_trace(pool, trace), poolkit::TraceError);
}

TEST(Trace, DoubleFreeIsRejected) {
    const poolkit::Trace trace = poolkit::parse_trace(
        "{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0}\n"
        "{\"op\":\"free\",\"size\":0,\"ptr\":0,\"stream\":0,\"ordinal\":1}\n"
        "{\"op\":\"free\",\"size\":0,\"ptr\":0,\"stream\":0,\"ordinal\":2}\n");
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend);
    EXPECT_THROW(poolkit::replay_trace(pool, trace), poolkit::TraceError);
}

TEST(Trace, AllocReusingALivePtrIsRejected) {
    const poolkit::Trace trace = poolkit::parse_trace(
        "{\"op\":\"alloc\",\"size\":100,\"ptr\":0,\"stream\":0,\"ordinal\":0}\n"
        "{\"op\":\"alloc\",\"size\":200,\"ptr\":0,\"stream\":0,\"ordinal\":1}\n");
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend);
    EXPECT_THROW(poolkit::replay_trace(pool, trace), poolkit::TraceError);
}

TEST(Trace, ReplayIsDeterministic) {
    const poolkit::Trace trace = poolkit::parse_trace(kValidTrace);
    poolkit::FakeBackend backend_a;
    poolkit::FakeBackend backend_b;
    poolkit::Pool pool_a(backend_a);
    poolkit::Pool pool_b(backend_b);

    const poolkit::TraceReplayResult a = poolkit::replay_trace(pool_a, trace);
    const poolkit::TraceReplayResult b = poolkit::replay_trace(pool_b, trace);

    EXPECT_EQ(a.alloc_count, b.alloc_count);
    EXPECT_EQ(a.free_count, b.free_count);

    const poolkit::PoolSnapshot& sa = a.snapshot;
    const poolkit::PoolSnapshot& sb = b.snapshot;
    EXPECT_EQ(sa.reuse_count, sb.reuse_count);
    EXPECT_EQ(sa.growth_count, sb.growth_count);
    EXPECT_EQ(sa.recovered_count, sb.recovered_count);
    EXPECT_EQ(sa.large_path_count, sb.large_path_count);
    EXPECT_EQ(sa.bytes_resident, sb.bytes_resident);
    EXPECT_EQ(sa.bytes_requested_live, sb.bytes_requested_live);
    EXPECT_EQ(sa.bytes_capacity_live, sb.bytes_capacity_live);
    EXPECT_EQ(sa.tail_bytes, sb.tail_bytes);
    EXPECT_EQ(sa.high_water_mark_bytes, sb.high_water_mark_bytes);
    EXPECT_DOUBLE_EQ(sa.hit_rate, sb.hit_rate);
    EXPECT_DOUBLE_EQ(sa.internal_fragmentation, sb.internal_fragmentation);
    EXPECT_DOUBLE_EQ(sa.tail_fragmentation, sb.tail_fragmentation);
    EXPECT_DOUBLE_EQ(sa.resident_slack, sb.resident_slack);
}

// The committed trace: replay it and measure the internal waste the cost model predicts.

TEST(Trace, CommittedTraceReplaysAndHasSaneInternalWaste) {
    const char* path = committed_trace_path();
    if (path == nullptr) {
        GTEST_SKIP() << "POOLKIT_SOURCE_DIR not defined at compile time";
    }
    const poolkit::Trace trace = poolkit::parse_trace_file(path);
    ASSERT_GT(trace.ops.size(), std::size_t{0});

    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend);
    const poolkit::TraceReplayResult result = poolkit::replay_trace(pool, trace);
    EXPECT_EQ(result.alloc_count, trace.ops.size() - result.free_count);

    // The "measured" internal waste, against the log-uniform placeholder's 18.028 %.
    poolkit::SizeClassTable table;
    std::vector<double> sizes;
    for (const poolkit::TraceOp& op : trace.ops) {
        if (op.kind == poolkit::TraceOpKind::alloc &&
            op.size <= table.config().max_class_bytes) {
            sizes.push_back(static_cast<double>(op.size));
        }
    }
    ASSERT_FALSE(sizes.empty());
    const double waste_capacity = table.expected_internal_waste(sizes);

    // A real workload is not log-uniform, so this pins a plausible band and prints the value.
    EXPECT_GT(waste_capacity, 0.0);
    EXPECT_LT(waste_capacity, 0.5);

    std::cout << "[trace] alloc ops           = " << result.alloc_count << '\n';
    std::cout << "[trace] free ops            = " << result.free_count << '\n';
    std::cout << "[trace] internal waste (cap-normalised) = "
              << waste_capacity * 100.0 << " %  (log-uniform model: 18.028 %)\n";
    std::cout << "[trace] internal_fragmentation (req-normalised) = "
              << result.snapshot.internal_fragmentation * 100.0 << " %\n";
}

// §9: the powers of two are a measurement, not a preference. If they stop lowering the waste on
// this trace, the table is wrong, not the test.
TEST(Trace, PowersOfTwoLowerTheMeasuredInternalWaste) {
    const char* path = committed_trace_path();
    if (path == nullptr) {
        GTEST_SKIP() << "POOLKIT_SOURCE_DIR not defined at compile time";
    }
    const poolkit::Trace trace = poolkit::parse_trace_file(path);

    poolkit::SizeClassTable::Config plain;
    plain.include_power_of_two = false;
    const poolkit::SizeClassTable geometric(plain);
    const poolkit::SizeClassTable shipped;  // g = 1.5, powers of two on

    std::vector<double> sizes;
    for (const poolkit::TraceOp& op : trace.ops) {
        if (op.kind == poolkit::TraceOpKind::alloc && op.size <= shipped.config().max_class_bytes) {
            sizes.push_back(static_cast<double>(op.size));
        }
    }
    ASSERT_FALSE(sizes.empty());

    const double before = geometric.expected_internal_waste(sizes);
    const double after = shipped.expected_internal_waste(sizes);
    std::cout << "[trace] internal waste, geometric only (k = " << geometric.count()
              << ")      = " << before * 100.0 << " %\n";
    std::cout << "[trace] internal waste, with powers of two (k = " << shipped.count()
              << ") = " << after * 100.0 << " %\n";

    EXPECT_LT(after, before);
    EXPECT_GT(shipped.count(), geometric.count());
}

TEST(Trace, ReplayCountersFollowTheTrace) {
    const poolkit::Trace trace = poolkit::parse_trace(kValidTrace);
    poolkit::FakeBackend backend;
    poolkit::Pool pool(backend);
    const poolkit::TraceReplayResult result = poolkit::replay_trace(pool, trace);
    EXPECT_EQ(result.alloc_count, std::size_t{2});
    EXPECT_EQ(result.free_count, std::size_t{1});
    // 100 B -> class 0, 300 B -> class 1: two growths, and the free of ptr 1 leaves 100 B live.
    EXPECT_EQ(result.snapshot.growth_count, std::size_t{2});
    EXPECT_EQ(result.snapshot.bytes_requested_live, std::size_t{100});
}

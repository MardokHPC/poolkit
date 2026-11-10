// Cost model, size class table and slab arithmetic against numbers computed by hand in
// docs/cost-model.md: every literal is a derivation or an exact integer, nothing measured.

#include <poolkit/size_classes.hpp>
#include <poolkit/slab.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMin = 256;
constexpr std::size_t kMax = std::size_t{2} << 20;   // 2 MiB, = 8192 * 256
constexpr std::size_t kChunk = std::size_t{2} << 20;  // default chunk_bytes

poolkit::SizeClassTable make_table(double growth) {
    poolkit::SizeClassTable::Config cfg;
    cfg.min_class_bytes = kMin;
    cfg.max_class_bytes = kMax;
    cfg.growth = growth;
    // Powers of two are merged only in the shipped default; they are tested separately.
    cfg.include_power_of_two = false;
    return poolkit::SizeClassTable(cfg);
}

// The recurrence, re-implemented on purpose: the production table must match it class by
// class, so k falls out of the capacities instead of being copied from a document.
std::size_t round_up(std::size_t value, std::size_t align) {
    return ((value + align - 1) / align) * align;
}

std::vector<std::size_t> derive_capacities(std::size_t lo, std::size_t hi, double growth) {
    std::vector<std::size_t> sizes{lo};
    while (sizes.back() < hi) {
        const double stepped = std::ceil(static_cast<double>(sizes.back()) * growth);
        sizes.push_back(round_up(static_cast<std::size_t>(stepped), lo));
    }
    return sizes;
}

// The plan's formula in double; used only inside an interval, never on a boundary.
std::size_t reference_index(double bytes, double growth) {
    const double raw = std::log(bytes / static_cast<double>(kMin)) / std::log(growth);
    return static_cast<std::size_t>(std::ceil(raw));
}
// Midpoint of each step, so no sample lands on a boundary.
std::vector<double> log_uniform_samples(double lo, double hi, std::size_t steps) {
    std::vector<double> samples;
    samples.reserve(steps);
    const double log_ratio = std::log(hi / lo);
    for (std::size_t i = 0; i < steps; ++i) {
        const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(steps);
        samples.push_back(lo * std::exp(log_ratio * t));
    }
    return samples;
}

}  // namespace

TEST(SizeClassTable, CapacitiesFollowTheRoundedRecurrence) {
    // The definition the alignment guarantee forces: without the rounding class 5 at g = 1.5
    // is 1944 B, and the second block of a packed slab starts on a multiple of 8.
    for (const double growth : {1.25, 1.5, 2.0}) {
        const auto table = make_table(growth);
        const auto expected = derive_capacities(kMin, kMax, growth);
        ASSERT_EQ(table.count(), expected.size()) << "growth " << growth;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(table.class_size(i), expected[i]) << "growth " << growth << " class " << i;
        }
    }
}

TEST(SizeClassTable, ClassCountIsTheSmallestCoveringTable) {
    // k is read from the table, not from the closed form: rounding moves it.
    for (const double growth : {1.25, 1.5, 2.0}) {
        const auto table = make_table(growth);
        const std::size_t n = table.count();
        ASSERT_GE(n, std::size_t{2});
        EXPECT_GE(table.class_size(n - 1), kMax);
        EXPECT_LT(table.class_size(n - 2), kMax);
        EXPECT_EQ(n, derive_capacities(kMin, kMax, growth).size());
    }
}

TEST(SizeClassTable, ClassesAreStrictlyIncreasing) {
    for (const double growth : {1.25, 1.5, 2.0}) {
        const auto table = make_table(growth);
        EXPECT_EQ(table.class_size(0), kMin);
        for (std::size_t i = 1; i < table.count(); ++i) {
            EXPECT_LT(table.class_size(i - 1), table.class_size(i)) << "class " << i;
        }
    }
}

TEST(SizeClassTable, EveryCapacityIsAMultipleOfTheAlignmentFloor) {
    // "Every block is >= 256 B aligned" reduces to this: a slab is base + k*c.
    for (const double growth : {1.25, 1.5, 2.0}) {
        const auto table = make_table(growth);
        for (std::size_t i = 0; i < table.count(); ++i) {
            EXPECT_EQ(table.class_size(i) % kMin, std::size_t{0})
                << "growth " << growth << " class " << i << " capacity " << table.class_size(i);
        }
    }

    // The class that used to break it, as a named regression: 1944 B un-rounded, 3072 B here.
    const auto table = make_table(1.5);
    EXPECT_EQ(table.class_size(5), std::size_t{3072});
    EXPECT_EQ(table.class_size(5) % kMin, std::size_t{0});
}

TEST(SizeClassTable, RejectsInvalidConfigurations) {
    auto cfg = poolkit::SizeClassTable::Config{};

    cfg.growth = 1.0;
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);
    cfg.growth = 0.9;
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);
    cfg.growth = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);
    cfg.growth = std::numeric_limits<double>::infinity();
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);

    cfg = poolkit::SizeClassTable::Config{};
    cfg.min_class_bytes = 0;
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);

    cfg = poolkit::SizeClassTable::Config{};
    cfg.max_class_bytes = cfg.min_class_bytes - 1;
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);

    // growth = 1 + 1e-9 over 256 B..2 MiB needs ~16 G classes: it has to fail here, not eat memory.
    cfg = poolkit::SizeClassTable::Config{};
    cfg.growth = 1.0 + 1e-9;
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);
}

TEST(SizeClassTable, IndexForHonoursEveryClassBoundary) {
    for (const double growth : {1.25, 1.5, 2.0}) {
        const auto table = make_table(growth);
        const std::size_t n = table.count();

        EXPECT_EQ(table.index_for(kMin), std::size_t{0});
        // s_max maps to the last class; one byte above stays clamped there.
        EXPECT_EQ(table.index_for(kMax), n - 1);
        EXPECT_EQ(table.index_for(kMax + 1), n - 1);

        for (std::size_t i = 1; i < n; ++i) {
            const std::size_t c = table.class_size(i);
            EXPECT_EQ(table.index_for(c), i) << "exact capacity " << c;
            // c - 1 still needs class i: ceil() semantics, c_{i-1} is at most c/g.
            EXPECT_EQ(table.index_for(c - 1), i) << "capacity - 1 " << c;
            EXPECT_EQ(table.index_for(table.class_size(i - 1) + 1), i);
            if (i + 1 < n) {
                EXPECT_EQ(table.index_for(c + 1), i + 1) << "capacity + 1 " << c;
            }
        }
    }
}

TEST(SizeClassTable, IndexForAgreesWithTheClosedFormOnUnroundedTables) {
    // The closed form i(s) = ceil(log_g(s/s_min)) is exact only on an unrounded table. With g = 2
    // and a 256 B floor rounding is a no-op, so here the identity must hold; index_for is a binary
    // search precisely because it does not hold in general.
    poolkit::SizeClassTable::Config cfg;
    cfg.min_class_bytes = 256;
    cfg.max_class_bytes = 256 * 64;  // 16 KiB: powers of two, so rounding is a no-op
    cfg.growth = 2.0;
    const poolkit::SizeClassTable table(cfg);
    for (std::size_t i = 1; i < table.count(); ++i) {
        const double lo = static_cast<double>(table.class_size(i - 1));
        const double hi = static_cast<double>(table.class_size(i));
        for (const double frac : {0.1, 0.25, 0.5, 0.75}) {
            const auto bytes = static_cast<std::size_t>(lo + frac * (hi - lo));
            EXPECT_EQ(table.index_for(bytes), i);
            EXPECT_EQ(reference_index(static_cast<double>(bytes), cfg.growth), i)
                << "bytes " << bytes;
        }
    }
}

TEST(SizeClassTable, IndexForIsTheSmallestClassThatFits) {
    // The property that replaces the closed form: the class fits and the previous one does not.
    for (const double growth : {1.25, 1.5, 2.0}) {
        const auto table = make_table(growth);
        for (std::size_t bytes = 1; bytes <= 20000; ++bytes) {
            const std::size_t index = table.index_for(bytes);
            EXPECT_GE(table.class_size(index), bytes) << "bytes " << bytes;
            if (index > 0) {
                EXPECT_LT(table.class_size(index - 1), bytes) << "bytes " << bytes;
            }
        }
    }
}

TEST(SizeClassTable, IndexForRejectsDegenerateRequests) {
    const auto table = make_table(1.5);
    EXPECT_THROW(table.index_for(0), std::invalid_argument);
    EXPECT_THROW(table.class_size(table.count()), std::out_of_range);
    EXPECT_EQ(table.size_for_request(1), kMin);
    EXPECT_EQ(table.size_for_request(kMax), table.class_size(table.count() - 1));
}

TEST(InternalWaste, ClosedFormMatchesHandComputation) {
    // 1 - (g-1)/(g ln g) by hand: 0.1037, 0.1779, 0.2787 for g = 1.25, 1.5, 2.
    EXPECT_NEAR(make_table(1.25).expected_internal_waste_log_uniform(), 0.1037, 1e-4);
    EXPECT_NEAR(make_table(1.5).expected_internal_waste_log_uniform(), 0.1779, 1e-4);
    EXPECT_NEAR(make_table(2.0).expected_internal_waste_log_uniform(), 0.2787, 1e-4);
}

TEST(InternalWaste, IsMonotoneIncreasingInGrowth) {
    // E_int is 0 as g -> 1 and grows from there: without retention there is no interior minimum.
    double previous = 0.0;
    for (int step = 110; step <= 200; ++step) {
        const double value =
            make_table(static_cast<double>(step) / 100.0).expected_internal_waste_log_uniform();
        EXPECT_GT(value, previous) << "growth " << step / 100.0;
        previous = value;
    }
}

TEST(InternalWaste, EmpiricalIntegralMatchesTheClosedFormOnASingleClass) {
    // One class interval: {256, 512} with g = 2, so the closed form is exact. {256, 384} would
    // not work: 384 is not a multiple of 256, so the rounded table is {256, 512} anyway.
    poolkit::SizeClassTable::Config cfg;
    cfg.min_class_bytes = 256;
    cfg.max_class_bytes = 512;
    cfg.growth = 2.0;
    const poolkit::SizeClassTable table(cfg);
    ASSERT_EQ(table.count(), std::size_t{2});
    ASSERT_EQ(table.class_size(1), std::size_t{512});

    const auto samples = log_uniform_samples(256.0, 512.0, 200000);
    EXPECT_NEAR(table.expected_internal_waste(samples),
                table.expected_internal_waste_log_uniform(), 1e-3);
}

TEST(InternalWaste, EmpiricalIntegralExceedsTheClosedFormOnlyBecauseOfTheTopClass) {
    const auto exact = make_table(2.0);
    const double empirical_exact =
        exact.expected_internal_waste(log_uniform_samples(256.0, static_cast<double>(kMax), 20000));
    EXPECT_NEAR(empirical_exact, exact.expected_internal_waste_log_uniform(), 2e-3);

    // g = 1.5 ends with a class of 3107072 B over a 2 MiB ceiling: the top class keeps only its
    // cheapest requests, so the measured waste differs (0.1803 against 0.1779).
    const auto truncated = make_table(1.5);
    const double empirical_truncated = truncated.expected_internal_waste(
        log_uniform_samples(256.0, static_cast<double>(kMax), 20000));
    EXPECT_NE(empirical_truncated, truncated.expected_internal_waste_log_uniform());
    EXPECT_NEAR(empirical_truncated, 0.18028, 2e-3);
}

TEST(InternalWaste, EmpiricalIntegralValidatesItsInput) {
    const auto table = make_table(1.5);
    EXPECT_EQ(table.expected_internal_waste({}), 0.0);
    EXPECT_THROW((void)table.expected_internal_waste(std::vector<double>{0.0}),
                 std::invalid_argument);
    EXPECT_THROW((void)table.expected_internal_waste(std::vector<double>{-8.0}),
                 std::invalid_argument);
    EXPECT_THROW((void)table.expected_internal_waste(std::vector<double>{1.5e6 * 4}),
                 std::invalid_argument);
    // A single request at a class capacity wastes nothing.
    EXPECT_NEAR(table.expected_internal_waste(std::vector<double>{256.0}), 0.0, 1e-12);
}

TEST(SizeClassTable, MarkdownTableHasOneRowPerClass) {
    const auto table = make_table(1.5);
    const std::string markdown = table.to_markdown_table();
    std::size_t rows = 0;
    for (const char ch : markdown) {
        if (ch == '\n') {
            ++rows;
        }
    }
    EXPECT_EQ(rows, table.count() + 2);  // header + separator + one row per class
    // The last class is not a whole MiB: the table must print the exact byte count.
    EXPECT_NE(markdown.find(std::to_string(table.class_size(table.count() - 1))), std::string::npos);
}

TEST(SizeClassTable, DefaultTableIncludesEveryPowerOfTwoInRange) {
    // jemalloc-style union: tensor dimensions are powers of two, so a geometric table that splits them pays twice.
    const poolkit::SizeClassTable table;  // shipped defaults
    std::vector<std::size_t> powers;
    for (std::size_t power = kMin; power <= kMax; power <<= 1) {
        powers.push_back(power);
    }
    ASSERT_GE(powers.size(), std::size_t{8});
    for (const std::size_t power : powers) {
        EXPECT_EQ(table.class_size(table.index_for(power)), power)
            << "power of two " << power << " does not map onto itself";
    }
    for (std::size_t i = 0; i < table.count(); ++i) {
        EXPECT_EQ(table.class_size(i) % kMin, std::size_t{0}) << "class " << i;
        if (i > 0) {
            EXPECT_LT(table.class_size(i - 1), table.class_size(i)) << "class " << i;
        }
    }
    EXPECT_GE(table.class_size(table.count() - 1), kMax);
}

TEST(SizeClassTable, IncludePowerOfTwoFalseReproducesTheGeometricTable) {
    // The switch makes the §9 comparison reproducible: off gives the pure geometric table back.
    poolkit::SizeClassTable::Config cfg;
    cfg.include_power_of_two = false;
    const poolkit::SizeClassTable table(cfg);
    ASSERT_EQ(table.count(), std::size_t{23});
    const auto expected = derive_capacities(kMin, kMax, 1.5);
    ASSERT_EQ(table.count(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(table.class_size(i), expected[i]) << "class " << i;
    }
}

TEST(SizeClassTable, ExtraSizesAreMergedRoundedUpAndDeduplicated) {
    poolkit::SizeClassTable::Config cfg;
    cfg.include_power_of_two = false;
    const poolkit::SizeClassTable plain(cfg);

    cfg.extra_sizes = {std::size_t{1} << 20};  // exactly 1 MiB, already a multiple of 256
    const poolkit::SizeClassTable with_mib(cfg);
    EXPECT_EQ(with_mib.count(), plain.count() + 1);
    const std::size_t index = with_mib.index_for(std::size_t{1} << 20);
    EXPECT_EQ(with_mib.class_size(index), std::size_t{1} << 20);

    // A duplicate of an existing capacity, a duplicate of the extra itself and
    // an unaligned value: the union absorbs all three.
    cfg.extra_sizes = {std::size_t{1} << 20, std::size_t{1} << 20, 256, 1000};
    const poolkit::SizeClassTable deduped(cfg);
    EXPECT_EQ(deduped.count(), with_mib.count() + 1);  // only 1000 -> 1024 is new
    EXPECT_EQ(deduped.class_size(deduped.index_for(1000)), std::size_t{1024});
    for (std::size_t i = 0; i < deduped.count(); ++i) {
        EXPECT_EQ(deduped.class_size(i) % kMin, std::size_t{0}) << "class " << i;
        if (i > 0) {
            EXPECT_LT(deduped.class_size(i - 1), deduped.class_size(i)) << "class " << i;
        }
    }
}

TEST(SizeClassTable, ExtraSizesOutsideTheRangeAreRejected) {
    poolkit::SizeClassTable::Config cfg;
    cfg.extra_sizes = {kMin - 1};
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);
    cfg.extra_sizes = {kMax + 1};
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);
    cfg.extra_sizes = {0};
    EXPECT_THROW(poolkit::SizeClassTable{cfg}, std::invalid_argument);
    cfg.extra_sizes = {kMin, kMax};
    EXPECT_NO_THROW(poolkit::SizeClassTable{cfg});
}

TEST(Slab, ArithmeticMatchesHandComputation) {
    // c = 256 B: 2097152 / 256 = 8192 exactly, tail 0.
    EXPECT_EQ(poolkit::chunk_bytes_for(256, kChunk), kChunk);
    EXPECT_EQ(poolkit::blocks_per_chunk(256, kChunk), std::size_t{8192});
    EXPECT_EQ(poolkit::tail_bytes(256, kChunk), std::size_t{0});

    // c = 1 MiB: two blocks per chunk, tail 0.
    EXPECT_EQ(poolkit::blocks_per_chunk(std::size_t{1} << 20, kChunk), std::size_t{2});
    EXPECT_EQ(poolkit::tail_bytes(std::size_t{1} << 20, kChunk), std::size_t{0});

    // c = 700000, chunk 2 MiB: floor(2097152/700000) = 2, tail = 2097152 - 1400000.
    EXPECT_EQ(poolkit::blocks_per_chunk(700000, kChunk), std::size_t{2});
    EXPECT_EQ(poolkit::tail_bytes(700000, kChunk), std::size_t{697152});

    // A class bigger than the chunk gets a dedicated chunk of its own size.
    EXPECT_EQ(poolkit::chunk_bytes_for(3000000, kChunk), std::size_t{3000000});
    EXPECT_EQ(poolkit::blocks_per_chunk(3000000, 3000000), std::size_t{1});
    EXPECT_EQ(poolkit::tail_bytes(3000000, 3000000), std::size_t{0});
}

TEST(Slab, FitsSingleSlabFollowsTheBlockCount) {
    EXPECT_TRUE(poolkit::fits_single_slab(kChunk, kChunk));
    EXPECT_FALSE(poolkit::fits_single_slab(kChunk / 2, kChunk));
    EXPECT_TRUE(poolkit::fits_single_slab(kChunk / 2 + 1, kChunk));
    EXPECT_FALSE(poolkit::fits_single_slab(700000, kChunk));
    EXPECT_TRUE(poolkit::fits_single_slab(3000000, kChunk));
}

TEST(Slab, LayoutInvariantsHoldForEveryClassOfEveryTable) {
    for (const double growth : {1.25, 1.5, 2.0}) {
        const auto table = make_table(growth);
        for (const std::size_t chunk_min : {std::size_t{4096}, std::size_t{64} << 10, kChunk,
                                            std::size_t{8} << 20}) {
            for (std::size_t i = 0; i < table.count(); ++i) {
                const std::size_t c = table.class_size(i);
                const auto layout = poolkit::slab_layout(c, chunk_min);
                EXPECT_GE(layout.blocks, std::size_t{1});
                EXPECT_EQ(layout.blocks * c + layout.tail, layout.chunk_bytes);
                EXPECT_LT(layout.tail, c);
                EXPECT_GE(layout.chunk_bytes, c);
                EXPECT_GE(layout.chunk_bytes, chunk_min);
                EXPECT_GE(layout.tail_fraction(), 0.0);
                EXPECT_LT(layout.tail_fraction(), 1.0);
            }
        }
    }
}

TEST(Slab, ClassesJustBelowTheChunkSizeStrandTail) {
    // chunk = max(chunk_min, c), so for C/2 < c < C the slab holds one block and the tail is
    // C - c, not 0: the plan's "dedicated slab, tail = 0" only holds for c >= C.
    const std::size_t c = 1200000;  // 1.144 MiB, inside (1 MiB, 2 MiB)
    const auto layout = poolkit::slab_layout(c, kChunk);
    EXPECT_EQ(layout.chunk_bytes, kChunk);
    EXPECT_EQ(layout.blocks, std::size_t{1});
    EXPECT_EQ(layout.tail, kChunk - c);
    EXPECT_NEAR(layout.tail_fraction(), 897152.0 / 2097152.0, 1e-12);

    // Above the chunk size the dedicated slab really does waste nothing.
    const auto big = poolkit::slab_layout(3000000, kChunk);
    EXPECT_EQ(big.chunk_bytes, std::size_t{3000000});
    EXPECT_EQ(big.tail, std::size_t{0});
}

TEST(Slab, ExpectedTailWasteIsTheWeightedMeanOfTailFractions) {
    // 1 MiB -> tau = 0; 700000 -> tau = 697152 / 2097152. Equal weights -> 1/2.
    const double tau_700000 = 697152.0 / 2097152.0;
    const std::vector<std::pair<std::size_t, double>> weights{
        {std::size_t{1} << 20, 1.0}, {700000, 1.0}};
    EXPECT_NEAR(poolkit::expected_tail_waste(weights, kChunk), 0.5 * tau_700000, 1e-12);

    // Only the ratios matter: scaling every weight changes nothing.
    const std::vector<std::pair<std::size_t, double>> scaled{
        {std::size_t{1} << 20, 7.0}, {700000, 7.0}};
    EXPECT_NEAR(poolkit::expected_tail_waste(scaled, kChunk),
                poolkit::expected_tail_waste(weights, kChunk), 1e-12);

    // A class mix that divides the chunk exactly strands nothing.
    const std::vector<std::pair<std::size_t, double>> clean{
        {std::size_t{1} << 20, 3.0}, {256, 1.0}};
    EXPECT_NEAR(poolkit::expected_tail_waste(clean, kChunk), 0.0, 1e-12);
}

TEST(Slab, ExpectedTailWasteRejectsDegenerateInput) {
    const std::vector<std::pair<std::size_t, double>> empty{};
    EXPECT_EQ(poolkit::expected_tail_waste(empty, kChunk), 0.0);

    const std::vector<std::pair<std::size_t, double>> zero_weight{{256, 0.0}};
    EXPECT_EQ(poolkit::expected_tail_waste(zero_weight, kChunk), 0.0);

    const std::vector<std::pair<std::size_t, double>> negative{{256, -1.0}};
    EXPECT_THROW(poolkit::expected_tail_waste(negative, kChunk), std::invalid_argument);

    const std::vector<std::pair<std::size_t, double>> zero_class{{0, 1.0}};
    EXPECT_THROW(poolkit::expected_tail_waste(zero_class, kChunk), std::invalid_argument);
}

TEST(SuggestGrowth, LargeLiveSetLandsInTheCompetitiveBand) {
    // Cost(g) = E_int(g)*L + k(g)*C: only the sum has an interior minimum, near g = 1.2 for 1 GiB live.
    const double growth = poolkit::suggest_growth(1024.0 * 1024.0 * 1024.0, kChunk);
    EXPECT_GE(growth, 1.15);
    EXPECT_LE(growth, 1.25);
}

TEST(SuggestGrowth, SmallLiveSetPrefersFewerClasses) {
    // 64 MiB live against 23 classes of 2 MiB: retention is most of the cost, so fewer classes.
    const double growth = poolkit::suggest_growth(64.0 * 1024.0 * 1024.0, kChunk);
    EXPECT_GE(growth, 1.5);
    EXPECT_LE(growth, 4.0);
}

TEST(SuggestGrowth, IsNonIncreasingInLiveBytes) {
    // The argmin can only move towards smaller g as L grows.
    double previous = 0.0;
    for (const double mib : {4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0}) {
        const double growth = poolkit::suggest_growth(mib * 1024.0 * 1024.0, kChunk);
        EXPECT_LE(growth, previous == 0.0 ? 4.0 : previous) << "live " << mib << " MiB";
        previous = growth;
    }
}

TEST(SuggestGrowth, RejectsInvalidInput) {
    EXPECT_THROW((void)poolkit::suggest_growth(-1.0, kChunk), std::invalid_argument);
    EXPECT_THROW((void)poolkit::suggest_growth(std::numeric_limits<double>::quiet_NaN(), kChunk),
                 std::invalid_argument);
    EXPECT_THROW((void)poolkit::suggest_growth(1.0, 0), std::invalid_argument);
}

// Out-of-tree consumer: the only code in the repo that sees poolkit the way a user does, through an
// installed prefix + find_package(poolkit CONFIG).

#include <poolkit/config.hpp>
#include <poolkit/pool_error.hpp>
#include <poolkit/size_classes.hpp>
#include <poolkit/slab.hpp>

#include <cstddef>
#include <iostream>

namespace {

int fail(const char* what) {
    std::cerr << "poolkit consumer FAILED: " << what << "\n";
    return 1;
}

}  // namespace

int main() {
    poolkit::SizeClassTable::Config cfg;  // defaults: 256 B, 2 MiB, g = 1.5
    const poolkit::SizeClassTable table(cfg);

    // Structural invariants, not a pinned class count: the count moves with the alignment rounding, and
    // a check that has to be edited every time a formula changes stops being a check.
    if (table.count() < std::size_t{2}) {
        return fail("size class table is degenerate");
    }
    if (table.class_size(table.count() - 1) < cfg.max_class_bytes) {
        return fail("last size class does not cover max_class_bytes");
    }
    // The shipped union is not minimal, so the property worth checking is that the ceiling is an exact class.
    if (table.class_size(table.index_for(cfg.max_class_bytes)) != cfg.max_class_bytes) {
        return fail("max_class_bytes is not an exact size class");
    }
    for (std::size_t i = 0; i < table.count(); ++i) {
        if (table.class_size(i) % cfg.min_class_bytes != 0) {
            return fail("a size class is not a multiple of min_class_bytes");
        }
        if (i > 0 && table.class_size(i) <= table.class_size(i - 1)) {
            return fail("size classes are not strictly increasing");
        }
    }

    const auto layout = poolkit::slab_layout(700000, std::size_t{2} << 20);
    if (layout.blocks != std::size_t{2} || layout.tail != std::size_t{697152}) {
        return fail("slab layout for c = 700000 is not (2 blocks, 697152 B tail)");
    }

    bool caught = false;
    try {
        throw poolkit::PoolError("synthetic failure");
    } catch (const poolkit::PoolError& err) {
        caught = err.code() == poolkit::success_code() && !err.has_code();
    }
    if (!caught) {
        return fail("PoolError does not propagate its error code");
    }

    std::cout << "poolkit consumer OK: version " << POOLKIT_VERSION_STRING << ", " << table.count()
              << " classes, g = " << cfg.growth << ", last class " << table.class_size(table.count() - 1)
              << " B\n";
    return 0;
}

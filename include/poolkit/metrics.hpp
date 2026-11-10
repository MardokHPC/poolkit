#pragma once

// The metrics the API reports. "Fragmentation" without an adjective means nothing, so the
// snapshot carries internal (size class rounding), tail (stranded chunk bytes) and
// resident_slack (paid for, not used); docs/metrics.md defines each with a worked example.

#include <cstddef>
#include <vector>

#include <poolkit/config.hpp>

namespace poolkit {

struct PoolSnapshot {
    std::size_t bytes_requested_live = 0;
    std::size_t bytes_live_in_pool = 0;
    std::size_t bytes_capacity_live = 0;
    std::size_t bytes_resident = 0;
    std::size_t bytes_large_live = 0;
    std::size_t high_water_mark_bytes = 0;
    std::size_t chunk_count = 0;

    // resident == capacity_live + free_in_pool + quarantined_bytes + tail_bytes (asserted in tests).
    std::size_t bytes_free_in_pool = 0;
    std::size_t bytes_free_ready = 0;
    std::size_t bytes_free_pending = 0;
    std::size_t bytes_unused_slots = 0;
    std::size_t quarantined_blocks = 0;
    std::size_t quarantined_bytes = 0;
    std::size_t tail_bytes = 0;

    double internal_fragmentation = 0.0;
    double tail_fragmentation = 0.0;
    double resident_slack = 0.0;
    double free_ratio = 0.0;

    std::size_t reuse_count = 0;
    std::size_t growth_count = 0;
    std::size_t recovered_count = 0;
    std::size_t large_path_count = 0;
    std::size_t free_errors = 0;
    std::size_t release_all_refused = 0;
    double hit_rate = 0.0;

    struct ClassStats {
        std::size_t index = 0;
        std::size_t class_size = 0;
        std::size_t slab_count = 0;
        std::size_t total_blocks = 0;
        std::size_t live_blocks = 0;
        std::size_t ready_blocks = 0;
        std::size_t pending_blocks = 0;
        std::size_t unused_blocks = 0;
        std::size_t quarantined_blocks = 0;
        std::size_t tail_bytes_per_slab = 0;
        std::size_t alloc_count = 0;
        std::size_t reuse_count = 0;
        double hit_rate = 0.0;

        POOLKIT_NODISCARD std::size_t free_blocks() const noexcept {
            return ready_blocks + pending_blocks + unused_blocks;
        }
    };
    std::vector<ClassStats> classes;

    double alloc_avg_ns = 0.0;
    double alloc_max_ns = 0.0;
    double free_avg_ns = 0.0;
    double free_max_ns = 0.0;

    // Deliberately not a sum over every class >= request: the pool never borrows a block from
    // a larger class, so a free 512 B block is not usable for a 256 B request.
    POOLKIT_NODISCARD std::size_t usable_free(std::size_t request) const noexcept {
        for (const ClassStats& stats : classes) {
            if (stats.class_size >= request) {
                return (stats.ready_blocks + stats.unused_blocks) * stats.class_size;
            }
        }
        return 0;
    }

    // 1 - usable_free(request)/bytes_free_in_pool. Not a single number on purpose: the same pool
    // answers 0.0 for one request and 1.0 for another.
    POOLKIT_NODISCARD double external_fragmentation(std::size_t request) const noexcept {
        if (bytes_free_in_pool == 0) {
            return 0.0;
        }
        const double free_bytes = static_cast<double>(bytes_free_in_pool);
        return 1.0 - static_cast<double>(usable_free(request)) / free_bytes;
    }
};

}  // namespace poolkit

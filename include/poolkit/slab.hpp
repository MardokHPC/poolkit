#pragma once

// Slab carving arithmetic: a chunk is cut into whole blocks of one size class and the remainder is
// memory no allocation can ever use. Invariants checked in tests/test_size_classes.cpp:
// blocks >= 1, blocks*c + tail == chunk, tail < c.

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <poolkit/config.hpp>

namespace poolkit {

// A class never gets a chunk smaller than itself, so every slab holds at least one block. For c in
// (chunk_min/2, chunk_min) that single block leaves a tail of chunk_min - c, a documented wart.
POOLKIT_NODISCARD constexpr std::size_t chunk_bytes_for(std::size_t class_size,
                                                        std::size_t chunk_min) noexcept {
    return class_size > chunk_min ? class_size : chunk_min;
}

POOLKIT_NODISCARD constexpr std::size_t blocks_per_chunk(std::size_t class_size,
                                                         std::size_t chunk_bytes) noexcept {
    return class_size == 0 ? 0 : chunk_bytes / class_size;
}

POOLKIT_NODISCARD constexpr std::size_t tail_bytes(std::size_t class_size,
                                                   std::size_t chunk_bytes) noexcept {
    return chunk_bytes - blocks_per_chunk(class_size, chunk_bytes) * class_size;
}

// Defined through the arithmetic instead of a magic `c > chunk_min/2`, so it cannot drift.
POOLKIT_NODISCARD constexpr bool fits_single_slab(std::size_t class_size,
                                                  std::size_t chunk_min) noexcept {
    return blocks_per_chunk(class_size, chunk_bytes_for(class_size, chunk_min)) == 1;
}

struct SlabLayout {
    std::size_t class_size = 0;
    std::size_t chunk_bytes = 0;
    std::size_t blocks = 0;
    std::size_t tail = 0;

    POOLKIT_NODISCARD double tail_fraction() const noexcept {
        return chunk_bytes == 0 ? 0.0 : static_cast<double>(tail) / static_cast<double>(chunk_bytes);
    }
};

POOLKIT_NODISCARD inline SlabLayout slab_layout(std::size_t class_size,
                                                std::size_t chunk_min) noexcept {
    SlabLayout layout;
    layout.class_size = class_size;
    layout.chunk_bytes = chunk_bytes_for(class_size, chunk_min);
    layout.blocks = blocks_per_chunk(class_size, layout.chunk_bytes);
    layout.tail = layout.chunk_bytes - layout.blocks * class_size;
    return layout;
}

// E[W_tail] = sum_i (R_i / sum_j R_j) * tau_i, tau_i = tail_i / chunk_i. Only the ratios of R_i
// matter; "1 per class" is the assumption behind the published sweep.
inline double expected_tail_waste(const std::vector<std::pair<std::size_t, double>>& resident_weight,
                                 std::size_t chunk_min) {
    double total_weight = 0.0;
    for (const auto& entry : resident_weight) {
        if (entry.first == 0) {
            throw std::invalid_argument("poolkit: a size class of 0 bytes has no slab layout");
        }
        if (!(entry.second >= 0.0)) {
            throw std::invalid_argument("poolkit: resident weights must be finite and >= 0");
        }
        total_weight += entry.second;
    }
    if (!(total_weight > 0.0)) {
        return 0.0;
    }

    double waste = 0.0;
    for (const auto& entry : resident_weight) {
        waste += entry.second * slab_layout(entry.first, chunk_min).tail_fraction();
    }
    return waste / total_weight;
}

}  // namespace poolkit

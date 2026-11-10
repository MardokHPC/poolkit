#pragma once

// Geometric size classes. Pure C++17: no CUDA header, no nvcc, no GPU. The table is a
// parameterised object because the cost model has no closed-form optimum, so the sweep
// tool has to be able to ask for any growth factor.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <poolkit/config.hpp>

namespace poolkit {

namespace detail {

inline std::string human_bytes(std::size_t bytes) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    if (bytes >= (std::size_t{1} << 20) && bytes % (std::size_t{1} << 20) == 0) {
        out << (bytes >> 20) << " MiB";
    } else if (bytes >= (std::size_t{1} << 10) && bytes % (std::size_t{1} << 10) == 0) {
        out << (bytes >> 10) << " KiB";
    } else {
        out << bytes << " B";
    }
    return out.str();
}

}  // namespace detail

class SizeClassTable {
public:
    struct Config {
        // 256 B is cudaMalloc's practical alignment floor, not a tuning knob below it.
        std::size_t min_class_bytes = 256;
        std::size_t max_class_bytes = std::size_t{2} << 20;
        double growth = 1.5;
        bool include_power_of_two = true;
        // Caller-known capacities, rounded up and merged; outside the range is a configuration error, not a silent clamp.
        std::vector<std::size_t> extra_sizes{};
    };

    // A growth of 1+1e-12 would ask for 10^12 classes: fail fast, not out of memory in a constructor.
    static constexpr std::size_t max_classes = 4096;

    explicit SizeClassTable(Config config) : config_(config) {
        validate(config_);
        build();
    }

    // Overload, not a default argument: GCC rejects Config config = Config{} before the class is complete.
    SizeClassTable();

    POOLKIT_NODISCARD const Config& config() const noexcept { return config_; }

    POOLKIT_NODISCARD std::size_t count() const noexcept { return sizes_.size(); }

    std::size_t class_size(std::size_t index) const {
        if (index >= sizes_.size()) {
            throw std::out_of_range("poolkit: size class index " + std::to_string(index) +
                                    " out of range (count = " + std::to_string(sizes_.size()) + ")");
        }
        return sizes_[index];
    }

    // Smallest class whose capacity is >= bytes. Binary search, not ceil(log(s/s_min)/log(g)):
    // capacities are rounded up, so class 1 is 512 B and not 384, and the logarithm is off by an
    // ulp exactly on a boundary — where an off-by-one means a wasted class or a buffer overflow.
    std::size_t index_for(std::size_t bytes) const {
        if (bytes == 0) {
            throw std::invalid_argument("poolkit: a zero-byte request has no size class");
        }
        const auto it = std::lower_bound(sizes_.begin(), sizes_.end(), bytes);
        if (it == sizes_.end()) {
            return sizes_.size() - 1;
        }
        return static_cast<std::size_t>(it - sizes_.begin());
    }

    POOLKIT_NODISCARD std::size_t size_for_request(std::size_t bytes) const {
        return sizes_[index_for(bytes)];
    }

    // Closed form for log-uniform requests over the *geometric* progression:
    //   E[waste] / E[capacity] = 1 - (g-1)/(g ln g)
    POOLKIT_NODISCARD double expected_internal_waste_log_uniform() const noexcept {
        const double g = config_.growth;
        return 1.0 - (g - 1.0) / (g * std::log(g));
    }

    // Empirical internal waste, capacity-normalised:
    //   sum_i (c_{i(s)} - s) / sum_i c_{i(s)}
    // Samples above max_class_bytes are rejected, not clamped: they belong to the large path.
    POOLKIT_NODISCARD double expected_internal_waste(const std::vector<double>& request_sizes) const {
        if (request_sizes.empty()) {
            return 0.0;
        }
        const double max_bytes = static_cast<double>(config_.max_class_bytes);
        double waste = 0.0;
        double capacity = 0.0;
        for (const double bytes : request_sizes) {
            if (!(bytes > 0.0)) {
                throw std::invalid_argument("poolkit: request sizes must be finite and > 0");
            }
            if (bytes > max_bytes) {
                throw std::invalid_argument(
                    "poolkit: request size " + std::to_string(bytes) +
                    " is above max_class_bytes and belongs to the large path, not to the cost model");
            }
            const double c = static_cast<double>(sizes_[index_for_approx(bytes)]);
            waste += c - bytes;
            capacity += c;
        }
        return capacity > 0.0 ? waste / capacity : 0.0;
    }

    POOLKIT_NODISCARD std::string to_markdown_table() const {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << "| class | bytes | size | growth vs previous | capacity vs s_min |\n";
        out << "|---:|---:|---:|---:|---:|\n";
        out << std::fixed << std::setprecision(4);
        for (std::size_t i = 0; i < sizes_.size(); ++i) {
            const double ratio =
                i == 0 ? 1.0
                       : static_cast<double>(sizes_[i]) / static_cast<double>(sizes_[i - 1]);
            const double cumulative =
                static_cast<double>(sizes_[i]) / static_cast<double>(sizes_[0]);
            out << "| " << i << " | " << sizes_[i] << " | " << detail::human_bytes(sizes_[i])
                << " | " << ratio << " | " << cumulative << " |\n";
        }
        return out.str();
    }

private:
    static void validate(const Config& cfg) {
        if (cfg.min_class_bytes < 1) {
            throw std::invalid_argument("poolkit: min_class_bytes must be >= 1");
        }
        if (cfg.max_class_bytes < cfg.min_class_bytes) {
            throw std::invalid_argument("poolkit: max_class_bytes must be >= min_class_bytes");
        }
        // !(g > 1) also rejects NaN: a NaN growth would make every comparison in build() false.
        if (!(cfg.growth > 1.0) || !std::isfinite(cfg.growth)) {
            throw std::invalid_argument("poolkit: growth must be finite and > 1");
        }
    }

    void build() {
        const std::size_t align = config_.min_class_bytes;
        const std::size_t size_limit = std::numeric_limits<std::size_t>::max();

        // Capacities are multiples of min_class_bytes, not raw ceil(s_min*g^i): block 1 of a 1944 B
        // class would start at a multiple of 8, so the 256 B alignment promise fails on the first slab.
        // c_0 = min_class_bytes, c_{i+1} = round_up(ceil(c_i * g), min_class_bytes)
        std::vector<std::size_t> geometric;
        geometric.reserve(64);
        geometric.push_back(align);

        while (geometric.back() < config_.max_class_bytes) {
            if (geometric.size() >= max_classes) {
                throw std::invalid_argument(
                    "poolkit: growth " + std::to_string(config_.growth) + " needs more than " +
                    std::to_string(max_classes) + " classes between " +
                    std::to_string(config_.min_class_bytes) + " and " +
                    std::to_string(config_.max_class_bytes) + " bytes");
            }
            const double stepped = std::ceil(static_cast<double>(geometric.back()) * config_.growth);
            if (!(stepped < static_cast<double>(size_limit - align))) {
                throw std::length_error("poolkit: size class exceeds the addressable range");
            }
            const std::size_t unaligned = static_cast<std::size_t>(stepped);
            // round_up(unaligned, align), written so the addition cannot wrap.
            const std::size_t next = ((unaligned + align - 1) / align) * align;
            geometric.push_back(next > geometric.back() ? next : geometric.back() + align);
        }

        std::vector<std::size_t> merged;
        merged.reserve(geometric.size() + 32 + config_.extra_sizes.size());
        merged.insert(merged.end(), geometric.begin(), geometric.end());

        if (config_.include_power_of_two) {
            for (std::size_t power = 1; power <= config_.max_class_bytes;) {
                if (power >= config_.min_class_bytes && power % align == 0) {
                    merged.push_back(power);
                }
                if (power > config_.max_class_bytes / 2) {
                    break;
                }
                power <<= 1;
            }
        }

        for (const std::size_t requested : config_.extra_sizes) {
            if (requested < config_.min_class_bytes || requested > config_.max_class_bytes) {
                throw std::invalid_argument(
                    "poolkit: extra size " + std::to_string(requested) +
                    " is outside [min_class_bytes, max_class_bytes] = [" +
                    std::to_string(config_.min_class_bytes) + ", " +
                    std::to_string(config_.max_class_bytes) + "]");
            }
            if (requested > size_limit - align) {
                throw std::length_error("poolkit: extra size exceeds the addressable range");
            }
            merged.push_back(((requested + align - 1) / align) * align);
        }

        std::sort(merged.begin(), merged.end());
        merged.erase(std::unique(merged.begin(), merged.end()), merged.end());

        if (merged.size() > max_classes) {
            throw std::invalid_argument("poolkit: the size class union needs " +
                                        std::to_string(merged.size()) + " classes, more than " +
                                        std::to_string(max_classes));
        }

        sizes_ = std::move(merged);
    }

    // Same lower_bound on doubles: the size_t -> double conversion is a /W4 error inside std::lower_bound.
    POOLKIT_NODISCARD std::size_t index_for_approx(double bytes) const noexcept {
        std::size_t low = 0;
        std::size_t high = sizes_.size() - 1;
        while (low < high) {
            const std::size_t mid = low + (high - low) / 2;
            if (static_cast<double>(sizes_[mid]) < bytes) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }

    Config config_;
    std::vector<std::size_t> sizes_;
};

inline SizeClassTable::SizeClassTable() : SizeClassTable(Config{}) {}

// Growth that minimises Cost(g) = E_int(g) * live_bytes + k_used(g) * chunk_bytes: the first term
// is what the table throws away, the second what it retains. Without the second there is no interior
// optimum, because E_int is monotone in g (docs/cost-model.md §5.1). g in [1.05, 4.00], step 0.01.
POOLKIT_NODISCARD inline double suggest_growth(double live_bytes,
                                               std::size_t chunk_bytes,
                                               std::size_t min_class_bytes = 256,
                                               std::size_t max_class_bytes = std::size_t{2} << 20) {
    if (!(live_bytes >= 0.0) || !std::isfinite(live_bytes)) {
        throw std::invalid_argument("poolkit: live_bytes must be finite and >= 0");
    }
    if (chunk_bytes == 0) {
        throw std::invalid_argument("poolkit: chunk_bytes must be > 0");
    }

    const double retention_per_class = static_cast<double>(chunk_bytes);
    double best_growth = 1.05;
    double best_cost = std::numeric_limits<double>::infinity();

    for (int step = 105; step <= 400; ++step) {
        const double growth = static_cast<double>(step) / 100.0;
        SizeClassTable::Config cfg;
        cfg.min_class_bytes = min_class_bytes;
        cfg.max_class_bytes = max_class_bytes;
        cfg.growth = growth;
        cfg.include_power_of_two = false;
        const SizeClassTable table(cfg);
        const double cost = table.expected_internal_waste_log_uniform() * live_bytes +
                            static_cast<double>(table.count()) * retention_per_class;
        if (cost < best_cost) {
            best_cost = cost;
            best_growth = growth;
        }
    }
    return best_growth;
}

}  // namespace poolkit

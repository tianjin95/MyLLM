#pragma once

#include <cstdint>
#include <limits>

namespace llm {

// Persistent/payload memory owned by one backend. These values intentionally
// exclude C++ object overhead, allocator metadata, command buffers, and the
// small CPU-visible token-id/argmax staging buffers.
struct MemoryStats {
    std::uint64_t weight_bytes = 0;
    std::uint64_t kv_cache_bytes = 0;
    std::uint64_t intermediate_bytes = 0;
    bool intermediate_is_estimate = false;

    std::uint64_t total_bytes() const noexcept {
        const std::uint64_t max_value =
            std::numeric_limits<std::uint64_t>::max();
        if (weight_bytes > max_value - kv_cache_bytes) {
            return max_value;
        }
        const std::uint64_t persistent = weight_bytes + kv_cache_bytes;
        if (persistent > max_value - intermediate_bytes) {
            return max_value;
        }
        return persistent + intermediate_bytes;
    }
};

} // namespace llm

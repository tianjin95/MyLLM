#pragma once

#include "memory_stats.h"
#include "metal_model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

struct ExpertCacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;

    std::uint64_t requests() const noexcept {
        return hits + misses;
    }
};

// Text-only Metal runtime for qwen35moe models (Qwen3.5/Qwen3.6 A3B).
// It owns native GGUF weight buffers, full-attention KV caches, and DeltaNet
// recurrent state. Optional NextN/MTP blocks are never loaded or executed.
class MoeLLM {
public:
    explicit MoeLLM(const std::string& gguf_path,
                    std::size_t max_sequence = 0,
                    const std::string& shader_path = {},
                    std::size_t expert_cache_count = 8);
    ~MoeLLM();

    MoeLLM(const MoeLLM&) = delete;
    MoeLLM& operator=(const MoeLLM&) = delete;
    MoeLLM(MoeLLM&&) noexcept;
    MoeLLM& operator=(MoeLLM&&) noexcept;

    void reset();
    std::int32_t prefill(const std::vector<std::int32_t>& token_ids);
    std::int32_t decode(std::int32_t token_id);

    void enable_profiling(const std::string& csv_path,
                          bool detailed_kernel_timestamps = false);
    void disable_profiling();

    const MetalModelConfig& config() const;
    std::size_t position() const noexcept;
    std::size_t max_sequence() const noexcept;
    std::size_t expert_cache_count() const noexcept;
    ExpertCacheStats expert_cache_stats() const noexcept;
    bool uses_gpu() const noexcept;
    MemoryStats memory_stats() const noexcept;

    bool available() const noexcept;
    const std::string& device_name() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llm

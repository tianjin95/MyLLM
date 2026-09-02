#pragma once

#include "memory_stats.h"
#include "metal_model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

// Complete Metal runtime for one loaded model. The constructor streams native
// GGUF tensor bytes into persistent GPU-visible buffers without dequantizing
// matrix weights, then allocates fixed-capacity FP32 KV buffers. Prefill uses an
// arena sized to the actual prompt; decode lazily allocates and reuses a one-row
// arena. reset() starts a new conversation while retaining weights and KV
// allocation.
class MetalLLM {
public:
    explicit MetalLLM(const std::string& gguf_path,
                      std::size_t max_sequence = 0,
                      const std::string& shader_path = {});
    ~MetalLLM();

    MetalLLM(const MetalLLM&) = delete;
    MetalLLM& operator=(const MetalLLM&) = delete;
    MetalLLM(MetalLLM&&) noexcept;
    MetalLLM& operator=(MetalLLM&&) noexcept;

    void reset();
    std::int32_t prefill(const std::vector<std::int32_t>& token_ids);
    std::int32_t decode(std::int32_t token_id);

    void enable_profiling(const std::string& csv_path,
                          bool detailed_kernel_timestamps = false);
    void disable_profiling();

    const MetalModelConfig& config() const;
    std::size_t position() const noexcept;
    std::size_t max_sequence() const noexcept;
    bool uses_gpu() const noexcept;
    MemoryStats memory_stats() const noexcept;

    bool available() const noexcept;
    const std::string& device_name() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llm

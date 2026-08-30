#pragma once

#include "memory_stats.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

struct ModelConfig;

// Complete Metal runtime for one loaded model. The constructor parses GGUF,
// uploads every model weight, allocates fixed-capacity GPU KV buffers and one
// activation Arena, and then releases the temporary CPU tensors. reset() starts
// a new conversation while retaining the model and all planned allocations.
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

    void enable_profiling(const std::string& csv_path);
    void disable_profiling();

    const ModelConfig& config() const;
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

#pragma once

#include "model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

class MetalLLM;

class llm_runtime {
public:
    explicit llm_runtime(const std::string& gguf_path);
    ~llm_runtime();

    llm_runtime(const llm_runtime&) = delete;
    llm_runtime& operator=(const llm_runtime&) = delete;
    llm_runtime(llm_runtime&&) noexcept;
    llm_runtime& operator=(llm_runtime&&) noexcept;

    // Run a complete no-KV-cache forward pass over token_ids and return the
    // greedy next token predicted from the final position.
    int32_t forward(const std::vector<int32_t>& token_ids) const;

    // Report which linear-algebra backend is active.  Metal is selected by
    // default on Apple platforms; setting MYLLM_DISABLE_METAL=1 forces the
    // original CPU reference path.
    bool metal_enabled() const noexcept;
    const std::string& metal_device_name() const noexcept;

    // Layer-level helpers used by chat's KV-cache path.  They fall back to the
    // original free functions when Metal is unavailable.
    Matrix prefill_layer(const Matrix& hidden,
                         size_t layer_index,
                         Matrix& key_cache,
                         Matrix& value_cache) const;
    Vector decode_layer(const Vector& hidden,
                        size_t layer_index,
                        Matrix& key_cache,
                        Matrix& value_cache) const;
    Vector project_logits(const Vector& hidden) const;

    // Enable detailed operator timing and logical FLOP/traffic estimates.
    // The CSV is truncated when profiling is enabled.
    void enable_profiling(const std::string& csv_path);
    void disable_profiling();

    std::string architecture;
    size_t tensor_count = 0;
    size_t parameter_count = 0;
    size_t dequantized_bytes = 0;
    size_t layer_count = 0;
    size_t embedding_size = 0;
    size_t feed_forward_size = 0;
    size_t attention_head_count = 0;
    size_t kv_head_count = 0;
    size_t head_size = 0;
    size_t rotary_dimension = 0;
    size_t vocabulary_size = 0;
    size_t context_length = 0;
    int32_t eos_token_id = 0;
    float rope_theta = 1000000.0f;
    float norm_epsilon = 1e-6f;
    std::vector<std::string> token_vocabulary;
    std::vector<std::string> token_merges;
    std::string tokenizer_pre;

    Matrix token_embedding_weight;
    Vector output_norm_weight;
    Matrix output_weight;
    std::vector<Layer> layers;

private:
    std::unique_ptr<Profiler> profiler_;
    std::unique_ptr<MetalLLM> metal_backend_;
};

} // namespace llm

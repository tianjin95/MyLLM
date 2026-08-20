#pragma once

#include "model.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llm {

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
    Vector output_bias;
    std::vector<Layer> layers;
};

} // namespace llm

#pragma once

#include "llm.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

struct ModelConfig {
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
    std::vector<std::string> vocabulary;
    std::vector<std::string> merges;
    std::string tokenizer_pre;
};

class ModelFile {
public:
    explicit ModelFile(const std::string& gguf_path);
    ~ModelFile();

    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;
    ModelFile(ModelFile&&) = delete;
    ModelFile& operator=(ModelFile&&) = delete;

    const ModelConfig& config() const;
    // The token embedding remains [vocab, input] because it is accessed by
    // token id rather than multiplied as a linear layer.
    Matrix load_token_embedding();
    Vector load_output_norm();
    // The output LM-head weight is returned as [input, vocab], ready for an
    // input-row * weight GEMM/GEMV without a runtime transpose.
    Matrix load_output_weight();
    Vector load_output_bias();
    void validate_all_tensors_loaded() const;

private:
    friend class Layer;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Layer {
public:
    Layer(ModelFile& model_file, size_t layer_index);

    Vector attn_norm_weight;
    // All projection matrices below use [input dimension, output dimension].
    // This is the transpose of the [output, input] layout in GGUF.
    Matrix attn_q_weight;
    Vector attn_q_bias;
    Matrix attn_k_weight;
    Vector attn_k_bias;
    Matrix attn_v_weight;
    Vector attn_v_bias;
    Matrix attn_output_weight;
    Vector attn_output_bias;
    Vector ffn_norm_weight;
    Matrix ffn_gate_weight;
    Vector ffn_gate_bias;
    Matrix ffn_down_weight;
    Vector ffn_down_bias;
    Matrix ffn_up_weight;
    Vector ffn_up_bias;
};

} // namespace llm

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

// GGML tensor type ids are part of the GGUF file format.  Keep the numeric
// values explicit so a RawTensor can be passed to the Metal upload path
// without translating or losing its on-disk representation.
enum class MetalGgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
};

const char* metal_ggml_type_name(MetalGgmlType type) noexcept;
std::size_t metal_ggml_block_size(MetalGgmlType type);
std::size_t metal_ggml_type_size(MetalGgmlType type);
std::size_t metal_ggml_row_bytes(MetalGgmlType type,
                                 std::size_t columns);

struct MetalModelConfig {
    std::string architecture;
    std::size_t tensor_count = 0;
    std::size_t parameter_count = 0;
    std::size_t stored_weight_bytes = 0;
    std::size_t layer_count = 0;
    std::size_t embedding_size = 0;
    std::size_t feed_forward_size = 0;
    std::size_t attention_head_count = 0;
    std::size_t kv_head_count = 0;
    std::size_t head_size = 0;
    std::size_t rotary_dimension = 0;
    std::size_t vocabulary_size = 0;
    std::size_t context_length = 0;
    std::int32_t eos_token_id = 0;
    float rope_theta = 1000000.0f;
    float norm_epsilon = 1e-6f;
    std::vector<std::string> vocabulary;
    std::vector<std::string> merges;
    std::string tokenizer_pre;
};

// One tensor in its exact GGUF byte representation.  MetalRawModel reads only
// the requested tensor into this object; callers upload it and then let the
// temporary byte vector go out of scope.  This avoids materializing either a
// complete second GGUF image or a complete FP32 model on the CPU.
struct RawTensor {
    std::string name;
    MetalGgmlType type = MetalGgmlType::F32;
    std::vector<std::uint64_t> dimensions;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t row_bytes = 0;
    std::vector<std::uint8_t> data;

    std::size_t byte_size() const noexcept {
        return data.size();
    }
};

// GPU-specific layer staging.  Unlike the CPU Layer in model.h, every matrix
// remains quantized and every F32 norm/bias remains in its original byte form.
struct MetalRawLayer {
    RawTensor attn_norm_weight;
    RawTensor attn_q_weight;
    RawTensor attn_q_bias;
    RawTensor attn_k_weight;
    RawTensor attn_k_bias;
    RawTensor attn_v_weight;
    RawTensor attn_v_bias;
    RawTensor attn_output_weight;
    RawTensor ffn_norm_weight;
    RawTensor ffn_gate_weight;
    RawTensor ffn_down_weight;
    RawTensor ffn_up_weight;
};

// Independent, streaming GGUF reader for the Metal runtime.  It intentionally
// has no dependency on model.h/model.cpp or the CPU Matrix/Vector classes.
class MetalRawModel {
public:
    explicit MetalRawModel(const std::string& gguf_path);
    ~MetalRawModel();

    MetalRawModel(const MetalRawModel&) = delete;
    MetalRawModel& operator=(const MetalRawModel&) = delete;
    MetalRawModel(MetalRawModel&&) noexcept;
    MetalRawModel& operator=(MetalRawModel&&) noexcept;

    const MetalModelConfig& config() const;
    RawTensor load_token_embedding();
    RawTensor load_output_norm();
    RawTensor load_output_weight();
    MetalRawLayer load_layer(std::size_t layer_index);
    void validate_all_tensors_loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llm

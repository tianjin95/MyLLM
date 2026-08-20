#include "runtime.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llm {

llm_runtime::llm_runtime(const std::string& gguf_path) {
    ModelFile model_file(gguf_path);
    const ModelConfig& config = model_file.config();

    architecture = config.architecture;
    tensor_count = config.tensor_count;
    parameter_count = config.parameter_count;
    dequantized_bytes = config.dequantized_bytes;
    layer_count = config.layer_count;
    embedding_size = config.embedding_size;
    feed_forward_size = config.feed_forward_size;
    attention_head_count = config.attention_head_count;
    kv_head_count = config.kv_head_count;
    head_size = config.head_size;
    rotary_dimension = config.rotary_dimension;
    vocabulary_size = config.vocabulary_size;
    context_length = config.context_length;
    eos_token_id = config.eos_token_id;
    rope_theta = config.rope_theta;
    norm_epsilon = config.norm_epsilon;
    token_vocabulary = config.vocabulary;
    token_merges = config.merges;
    tokenizer_pre = config.tokenizer_pre;

    token_embedding_weight = model_file.load_token_embedding();
    output_norm_weight = model_file.load_output_norm();
    output_weight = model_file.load_output_weight();
    output_bias = model_file.load_output_bias();

    layers.reserve(layer_count);
    for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        layers.emplace_back(model_file, layer_index);
    }

    model_file.validate_all_tensors_loaded();
}

llm_runtime::~llm_runtime() = default;

llm_runtime::llm_runtime(llm_runtime&&) noexcept = default;

llm_runtime& llm_runtime::operator=(llm_runtime&&) noexcept = default;

int32_t llm_runtime::forward(const std::vector<int32_t>& token_ids) const {
    if (token_ids.empty()) {
        throw std::invalid_argument("Forward requires at least one token id");
    }
    if (token_ids.size() > context_length) {
        throw std::invalid_argument("Token sequence exceeds model context length");
    }
    if (layers.size() != layer_count) {
        throw std::runtime_error("Runtime layer storage is incomplete");
    }

    // The embedding lookup produces one hidden row per input token.
    Matrix hidden = embedding(token_ids, token_embedding_weight);

    // Qwen2.5 is a pre-norm decoder block: attention and FFN are both
    // residual sublayers, and each layer consumes the previous layer output.
    for (const Layer& layer : layers) {
        hidden = attention(
            hidden, layer, norm_epsilon, rotary_dimension, rope_theta, head_size);
        hidden = ffn(hidden, layer, norm_epsilon);
    }

    // The LM head reads the final normalized hidden row only. gevm uses the
    // runtime's [input, output] weight layout and therefore needs no transpose.
    hidden.rmsnorm(output_norm_weight, norm_epsilon);
    const Vector& last_hidden = hidden.values.back();
    Vector logits;
    if (output_bias.lens == 0 && output_bias.values.empty()) {
        logits = gevm(output_weight, last_hidden);
    } else {
        logits = gevmb(output_weight, last_hidden, output_bias);
    }

    if (logits.values.empty()) {
        throw std::runtime_error("LM head produced no logits");
    }
    const auto best = std::max_element(logits.values.begin(), logits.values.end());
    const size_t token = static_cast<size_t>(std::distance(logits.values.begin(), best));
    if (token > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error("Generated token id does not fit in int32_t");
    }
    return static_cast<int32_t>(token);
}

} // namespace llm

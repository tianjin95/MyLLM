#include "runtime.h"

#include "profiler.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llm {

namespace {

uint64_t float_matrix_bytes(size_t rows, size_t cols) {
    return static_cast<uint64_t>(rows) *
           static_cast<uint64_t>(cols) * sizeof(float);
}

} // namespace

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

    layers.reserve(layer_count);
    for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        layers.emplace_back(model_file, layer_index);
    }

    model_file.validate_all_tensors_loaded();
}

llm_runtime::~llm_runtime() = default;

llm_runtime::llm_runtime(llm_runtime&&) noexcept = default;

llm_runtime& llm_runtime::operator=(llm_runtime&&) noexcept = default;

void llm_runtime::enable_profiling(const std::string& csv_path) {
    profiler_ = std::make_unique<Profiler>(csv_path);
}

void llm_runtime::disable_profiling() {
    profiler_.reset();
}

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

    Profiler::ForwardScope forward_profile(profiler_.get(), token_ids.size());

    // The embedding lookup produces one hidden row per input token.
    Matrix hidden;
    {
        ProfileMetrics metrics = profile_copy_metrics(
            float_matrix_bytes(token_ids.size(), embedding_size),
            static_cast<uint64_t>(token_ids.size()) + 1);
        Profiler::Scope scope(profiler_.get(), "forward.embedding", metrics);
        hidden = embedding(token_ids, token_embedding_weight);
    }

    // Qwen2.5 is a pre-norm decoder block: attention and FFN are both
    // residual sublayers, and each layer consumes the previous layer output.
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const Layer& layer = layers[layer_index];
        Profiler::LayerScope layer_profile(profiler_.get(), layer_index);
        hidden = attention(
            hidden, layer, norm_epsilon, rotary_dimension, rope_theta, head_size,
            profiler_.get());
        hidden = ffn(hidden, layer, norm_epsilon, profiler_.get());
    }

    // The LM head reads the final normalized hidden row only. GGUF stores this
    // weight as [vocabulary, hidden], so hidden * weight^T uses GEVMT.
    {
        const size_t elements = hidden.rows * hidden.cols;
        ProfileMetrics metrics = profile_elementwise_metrics(
            elements, 5, 2, 1, 0, 0);
        Profiler::Scope scope(profiler_.get(), "forward.final_rmsnorm", metrics);
        hidden.rmsnorm(output_norm_weight, norm_epsilon);
    }
    const Vector& last_hidden = hidden.values.back();
    Vector logits;
    {
        ProfileMetrics metrics = profile_gemv_metrics(
            output_weight.cols, output_weight.rows);
        Profiler::Scope scope(profiler_.get(), "forward.lm_head.gevmt", metrics);
        logits = gevmt(output_weight, last_hidden);
    }

    if (logits.values.empty()) {
        throw std::runtime_error("LM head produced no logits");
    }
    int32_t result_token = 0;
    {
        ProfileMetrics metrics = profile_elementwise_metrics(
            logits.values.size(), 1, 1, 0, 0, 0);
        Profiler::Scope scope(profiler_.get(), "forward.lm_head.argmax", metrics);
        const auto best = std::max_element(
            logits.values.begin(), logits.values.end());
        const size_t token = static_cast<size_t>(
            std::distance(logits.values.begin(), best));
        if (token > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::overflow_error("Generated token id does not fit in int32_t");
        }
        result_token = static_cast<int32_t>(token);
    }
    return result_token;
}

} // namespace llm

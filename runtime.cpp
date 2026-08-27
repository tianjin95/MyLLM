#include "runtime.h"

#include "metal_llm.h"
#include "profiler.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
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

    const char* disable_metal = std::getenv("MYLLM_DISABLE_METAL");
    const bool metal_is_disabled =
        disable_metal != nullptr &&
        (std::strcmp(disable_metal, "1") == 0 ||
         std::strcmp(disable_metal, "true") == 0 ||
         std::strcmp(disable_metal, "TRUE") == 0);
    if (!metal_is_disabled) {
        try {
            metal_backend_ = std::make_unique<MetalLLM>();
            metal_backend_->prepare(layers, output_weight);
            std::cerr << "[MyLLM] backend=Metal device="
                      << metal_backend_->device_name() << '\n';
        } catch (const std::exception& error) {
            // Keep the reference implementation usable on machines without a
            // Metal device, an installed Metal source compiler, or enough
            // shared memory for the cached FP32 weights.
            std::cerr << "[MyLLM] Metal backend unavailable; using CPU: "
                      << error.what() << '\n';
            metal_backend_.reset();
        }
    } else {
        std::cerr << "[MyLLM] backend=CPU (MYLLM_DISABLE_METAL is set)\n";
    }
}

llm_runtime::~llm_runtime() = default;

llm_runtime::llm_runtime(llm_runtime&&) noexcept = default;

llm_runtime& llm_runtime::operator=(llm_runtime&&) noexcept = default;

bool llm_runtime::metal_enabled() const noexcept {
    return metal_backend_ != nullptr && metal_backend_->available();
}

const std::string& llm_runtime::metal_device_name() const noexcept {
    static const std::string empty;
    return metal_backend_ == nullptr ? empty : metal_backend_->device_name();
}

Matrix llm_runtime::prefill_layer(const Matrix& hidden,
                                  size_t layer_index,
                                  Matrix& key_cache,
                                  Matrix& value_cache) const {
    if (layer_index >= layers.size()) {
        throw std::out_of_range("Prefill layer index is outside the runtime");
    }
    const Layer& layer = layers[layer_index];
    if (metal_enabled()) {
        return metal_backend_->prefill(
            hidden, layer, norm_epsilon, rotary_dimension, rope_theta,
            head_size, key_cache, value_cache, profiler_.get());
    }
    return prefill(hidden, layer, norm_epsilon, rotary_dimension, rope_theta,
                   head_size, key_cache, value_cache);
}

Vector llm_runtime::decode_layer(const Vector& hidden,
                                 size_t layer_index,
                                 Matrix& key_cache,
                                 Matrix& value_cache) const {
    if (layer_index >= layers.size()) {
        throw std::out_of_range("Decode layer index is outside the runtime");
    }
    const Layer& layer = layers[layer_index];
    if (metal_enabled()) {
        return metal_backend_->decode(
            hidden, layer, norm_epsilon, rotary_dimension, rope_theta,
            head_size, key_cache, value_cache, profiler_.get());
    }
    return decode(hidden, layer, norm_epsilon, rotary_dimension, rope_theta,
                  head_size, key_cache, value_cache);
}

Vector llm_runtime::project_logits(const Vector& hidden) const {
    if (metal_enabled()) {
        return metal_backend_->gevmt(output_weight, hidden);
    }
    return gevmt(output_weight, hidden);
}

void llm_runtime::enable_profiling(const std::string& csv_path) {
    profiler_ = std::make_unique<Profiler>(csv_path);
}

void llm_runtime::disable_profiling() {
    profiler_.reset();
}

int32_t llm_runtime::forward(const std::vector<int32_t>& token_ids) const {
    if (metal_enabled()) {
        return metal_backend_->forward(token_ids, *this, profiler_.get());
    }
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

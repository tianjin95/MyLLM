#include "moe_model.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llm {
namespace {

std::size_t checked_product(std::size_t left,
                            std::size_t right,
                            const char* description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left * right;
}

void require_dimensions(const RawTensor& tensor,
                        std::initializer_list<std::uint64_t> expected) {
    if (tensor.dimensions == std::vector<std::uint64_t>(expected)) {
        return;
    }
    throw std::runtime_error(
        "unexpected GGUF dimensions for tensor " + tensor.name);
}

void require_dimensions(const RawTensorDescriptor& tensor,
                        std::initializer_list<std::uint64_t> expected) {
    if (tensor.dimensions == std::vector<std::uint64_t>(expected)) {
        return;
    }
    throw std::runtime_error(
        "unexpected GGUF dimensions for tensor " + tensor.name);
}

void require_qwen35_config(const MetalModelConfig& config) {
    if (config.architecture != "qwen35moe") {
        throw std::runtime_error(
            "MoeRawModel requires qwen35moe, got " + config.architecture);
    }
    if (config.layer_count == 0 || config.full_attention_interval == 0 ||
        config.embedding_size == 0 || config.attention_head_count == 0 ||
        config.kv_head_count == 0 || config.head_size == 0 ||
        config.rotary_dimension == 0 || config.expert_count == 0 ||
        config.expert_used_count == 0 ||
        config.expert_used_count > config.expert_count ||
        config.expert_feed_forward_size == 0 ||
        config.shared_expert_feed_forward_size == 0 ||
        config.ssm_conv_kernel == 0 || config.ssm_state_size == 0 ||
        config.ssm_group_count == 0 || config.ssm_time_step_rank == 0 ||
        config.ssm_inner_size == 0) {
        throw std::runtime_error("incomplete qwen35moe GGUF metadata");
    }
    if (config.rotary_dimension > config.head_size ||
        config.rotary_dimension % 2 != 0) {
        throw std::runtime_error(
            "qwen35moe rotary dimension must be even and fit one head");
    }
    if (config.expert_used_count > 8) {
        throw std::runtime_error(
            "qwen35moe currently supports at most eight selected experts");
    }
    if (config.ssm_conv_kernel > 9) {
        throw std::runtime_error(
            "qwen35moe convolution kernel exceeds the Metal history limit");
    }
    if (config.ssm_inner_size % config.ssm_time_step_rank != 0 ||
        config.ssm_inner_size / config.ssm_time_step_rank !=
            config.ssm_state_size) {
        throw std::runtime_error(
            "qwen35moe DeltaNet value-head dimensions are inconsistent");
    }
    if (config.ssm_time_step_rank % config.ssm_group_count != 0) {
        throw std::runtime_error(
            "qwen35moe DeltaNet value heads must group key heads evenly");
    }
}

} // namespace

struct MoeRawModel::Impl {
    explicit Impl(const std::string& path) : raw(path) {
        require_qwen35_config(raw.config());
        const MetalModelConfig& config = raw.config();
        for (std::size_t layer = config.layer_count;
             layer < config.total_layer_count;
             ++layer) {
            raw.ignore_tensor_prefix(
                "blk." + std::to_string(layer) + ".");
        }
    }

    RawTensor load_vector(const std::string& name, std::size_t length) {
        RawTensor tensor = raw.load_tensor(name, 1, length);
        require_dimensions(tensor, {length});
        return tensor;
    }

    RawTensor load_matrix(const std::string& name,
                          std::size_t rows,
                          std::size_t cols) {
        RawTensor tensor = raw.load_tensor(name, rows, cols);
        require_dimensions(tensor, {cols, rows});
        return tensor;
    }

    RawTensorDescriptor describe_experts(const std::string& name,
                                         std::size_t rows,
                                         std::size_t cols,
                                         std::size_t experts) {
        RawTensorDescriptor tensor = raw.describe_tensor(
            name, checked_product(rows, experts, "expert tensor"), cols);
        require_dimensions(tensor, {cols, rows, experts});
        return tensor;
    }

    MetalRawModel raw;
};

MoeRawModel::MoeRawModel(const std::string& gguf_path)
    : impl_(std::make_unique<Impl>(gguf_path)) {}

MoeRawModel::~MoeRawModel() = default;
MoeRawModel::MoeRawModel(MoeRawModel&&) noexcept = default;
MoeRawModel& MoeRawModel::operator=(MoeRawModel&&) noexcept = default;

const MetalModelConfig& MoeRawModel::config() const {
    return impl_->raw.config();
}

RawTensor MoeRawModel::load_token_embedding() {
    const MetalModelConfig& config = this->config();
    return impl_->load_matrix(
        "token_embd.weight", config.vocabulary_size, config.embedding_size);
}

RawTensor MoeRawModel::load_output_norm() {
    return impl_->load_vector(
        "output_norm.weight", this->config().embedding_size);
}

RawTensor MoeRawModel::load_output_weight() {
    const MetalModelConfig& config = this->config();
    return impl_->load_matrix(
        "output.weight", config.vocabulary_size, config.embedding_size);
}

MoeRawLayer MoeRawModel::load_layer(std::size_t layer_index) {
    const MetalModelConfig& config = this->config();
    if (layer_index >= config.layer_count) {
        throw std::out_of_range("MoE layer index is outside the main graph");
    }

    const std::string prefix = "blk." + std::to_string(layer_index) + ".";
    const std::size_t hidden = config.embedding_size;
    const std::size_t expert_ff = config.expert_feed_forward_size;
    const std::size_t shared_ff = config.shared_expert_feed_forward_size;
    const std::size_t experts = config.expert_count;

    MoeRawLayer result;
    result.kind = (layer_index + 1) % config.full_attention_interval == 0
        ? MoeLayerKind::FullAttention
        : MoeLayerKind::GatedDeltaNet;
    result.attention_norm_weight = impl_->load_vector(
        prefix + "attn_norm.weight", hidden);
    result.post_attention_norm_weight = impl_->load_vector(
        prefix + "post_attention_norm.weight", hidden);

    if (result.kind == MoeLayerKind::FullAttention) {
        const std::size_t q_width = checked_product(
            checked_product(config.attention_head_count, config.head_size,
                            "attention query"),
            2, "attention query and gate");
        const std::size_t kv_width = checked_product(
            config.kv_head_count, config.head_size, "attention KV");
        result.attention.q_gate_weight = impl_->load_matrix(
            prefix + "attn_q.weight", q_width, hidden);
        result.attention.k_weight = impl_->load_matrix(
            prefix + "attn_k.weight", kv_width, hidden);
        result.attention.v_weight = impl_->load_matrix(
            prefix + "attn_v.weight", kv_width, hidden);
        result.attention.q_norm_weight = impl_->load_vector(
            prefix + "attn_q_norm.weight", config.head_size);
        result.attention.k_norm_weight = impl_->load_vector(
            prefix + "attn_k_norm.weight", config.head_size);
        result.attention.output_weight = impl_->load_matrix(
            prefix + "attn_output.weight", hidden, q_width / 2);
    } else {
        const std::size_t key_width = checked_product(
            config.ssm_group_count, config.ssm_state_size,
            "DeltaNet key width");
        const std::size_t qkv_width = checked_product(
            key_width, 2, "DeltaNet QK width") + config.ssm_inner_size;
        result.delta_net.qkv_weight = impl_->load_matrix(
            prefix + "attn_qkv.weight", qkv_width, hidden);
        result.delta_net.z_weight = impl_->load_matrix(
            prefix + "attn_gate.weight", config.ssm_inner_size, hidden);
        result.delta_net.alpha_weight = impl_->load_matrix(
            prefix + "ssm_alpha.weight", config.ssm_time_step_rank, hidden);
        result.delta_net.beta_weight = impl_->load_matrix(
            prefix + "ssm_beta.weight", config.ssm_time_step_rank, hidden);
        result.delta_net.conv_weight = impl_->load_matrix(
            prefix + "ssm_conv1d.weight", qkv_width, config.ssm_conv_kernel);
        result.delta_net.dt_bias = impl_->load_vector(
            prefix + "ssm_dt.bias", config.ssm_time_step_rank);
        result.delta_net.a = impl_->load_vector(
            prefix + "ssm_a", config.ssm_time_step_rank);
        result.delta_net.state_norm_weight = impl_->load_vector(
            prefix + "ssm_norm.weight", config.ssm_state_size);
        result.delta_net.output_weight = impl_->load_matrix(
            prefix + "ssm_out.weight", hidden, config.ssm_inner_size);
    }

    result.experts.router_weight = impl_->load_matrix(
        prefix + "ffn_gate_inp.weight", experts, hidden);
    result.experts.gate_weight = impl_->describe_experts(
        prefix + "ffn_gate_exps.weight", expert_ff, hidden, experts);
    result.experts.up_weight = impl_->describe_experts(
        prefix + "ffn_up_exps.weight", expert_ff, hidden, experts);
    result.experts.down_weight = impl_->describe_experts(
        prefix + "ffn_down_exps.weight", hidden, expert_ff, experts);
    result.experts.shared_router_weight = impl_->load_vector(
        prefix + "ffn_gate_inp_shexp.weight", hidden);
    result.experts.shared_gate_weight = impl_->load_matrix(
        prefix + "ffn_gate_shexp.weight", shared_ff, hidden);
    result.experts.shared_up_weight = impl_->load_matrix(
        prefix + "ffn_up_shexp.weight", shared_ff, hidden);
    result.experts.shared_down_weight = impl_->load_matrix(
        prefix + "ffn_down_shexp.weight", hidden, shared_ff);
    return result;
}

void MoeRawModel::read_tensor_slice(
    const RawTensorDescriptor& tensor,
    std::size_t relative_offset,
    void* destination,
    std::size_t bytes) const {
    impl_->raw.read_tensor_slice(
        tensor, relative_offset, destination, bytes);
}

void MoeRawModel::validate_all_tensors_loaded() const {
    impl_->raw.validate_all_tensors_loaded();
}

} // namespace llm

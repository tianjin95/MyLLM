#pragma once

#include "metal_model.h"

#include <cstddef>
#include <memory>
#include <string>

namespace llm {

enum class MoeLayerKind {
    GatedDeltaNet,
    FullAttention,
};

struct MoeRawFullAttention {
    RawTensor q_gate_weight;
    RawTensor k_weight;
    RawTensor v_weight;
    RawTensor q_norm_weight;
    RawTensor k_norm_weight;
    RawTensor output_weight;
};

struct MoeRawDeltaNet {
    RawTensor qkv_weight;
    RawTensor z_weight;
    RawTensor alpha_weight;
    RawTensor beta_weight;
    RawTensor conv_weight;
    RawTensor dt_bias;
    RawTensor a;
    RawTensor state_norm_weight;
    RawTensor output_weight;
};

struct MoeRawExperts {
    RawTensor router_weight;
    RawTensorDescriptor gate_weight;
    RawTensorDescriptor up_weight;
    RawTensorDescriptor down_weight;
    RawTensor shared_router_weight;
    RawTensor shared_gate_weight;
    RawTensor shared_up_weight;
    RawTensor shared_down_weight;
};

struct MoeRawLayer {
    MoeLayerKind kind = MoeLayerKind::GatedDeltaNet;
    RawTensor attention_norm_weight;
    RawTensor post_attention_norm_weight;
    MoeRawFullAttention attention;
    MoeRawDeltaNet delta_net;
    MoeRawExperts experts;
};

// qwen35moe-specific tensor mapping layered on top of the GPU-only streaming
// GGUF reader. Optional NextN/MTP blocks are deliberately ignored.
class MoeRawModel {
public:
    explicit MoeRawModel(const std::string& gguf_path);
    ~MoeRawModel();

    MoeRawModel(const MoeRawModel&) = delete;
    MoeRawModel& operator=(const MoeRawModel&) = delete;
    MoeRawModel(MoeRawModel&&) noexcept;
    MoeRawModel& operator=(MoeRawModel&&) noexcept;

    const MetalModelConfig& config() const;
    RawTensor load_token_embedding();
    RawTensor load_output_norm();
    RawTensor load_output_weight();
    MoeRawLayer load_layer(std::size_t layer_index);
    void read_tensor_slice(const RawTensorDescriptor& tensor,
                           std::size_t relative_offset,
                           void* destination,
                           std::size_t bytes) const;
    void validate_all_tensors_loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llm

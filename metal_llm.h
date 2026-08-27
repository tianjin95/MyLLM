#pragma once

#include "model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

class Profiler;
class llm_runtime;

// Small Metal backend used by MyLLM.  The public tensor types deliberately
// remain the CPU-side Vector/Matrix types from llm.h; this class only replaces
// the linear algebra kernels and keeps the rest of the reference forward path
// easy to inspect.
class MetalLLM {
public:
    explicit MetalLLM(const std::string& shader_path = {});
    ~MetalLLM();

    MetalLLM(const MetalLLM&) = delete;
    MetalLLM& operator=(const MetalLLM&) = delete;
    MetalLLM(MetalLLM&&) noexcept;
    MetalLLM& operator=(MetalLLM&&) noexcept;

    bool available() const noexcept;
    const std::string& device_name() const noexcept;

    // Upload model-side matrices/vectors once.  The model objects must remain
    // alive and immutable after this call; activation tensors are transient and
    // are uploaded by each operation as needed.
    void prepare(const std::vector<Layer>& layers,
                 const Matrix& output_weight);

    // FP32 matrix products.  Matrix dimensions and layouts match the existing
    // CPU operators in llm.h.  In particular, model weights stay [out, in],
    // so gemmt(A, W) evaluates A * W^T.
    Matrix gemm(const Matrix& left, const Matrix& right) const;
    Matrix gemmb(const Matrix& left, const Matrix& right,
                 const Vector& bias) const;
    Matrix gemmt(const Matrix& left, const Matrix& right) const;
    Matrix gemmts(const Matrix& left, const Matrix& right,
                  std::size_t d_head) const;
    Matrix gemmtb(const Matrix& left, const Matrix& right,
                  const Vector& bias) const;
    Matrix gemtm(const Matrix& left, const Matrix& right) const;
    Matrix gemtmb(const Matrix& left, const Matrix& right,
                  const Vector& bias) const;
    Matrix gemtmt(const Matrix& left, const Matrix& right) const;
    Matrix gemtmtb(const Matrix& left, const Matrix& right,
                   const Vector& bias) const;

    // FP32 vector/matrix products.  These are the Batch=1 paths used by
    // decode and by the final vocabulary projection.
    Vector gevm(const Matrix& matrix, const Vector& vector) const;
    Vector gevmb(const Matrix& matrix, const Vector& vector,
                 const Vector& bias) const;
    Vector gevmts(const Matrix& matrix, const Vector& vector,
                  std::size_t d_head) const;
    Vector gevmt(const Matrix& matrix, const Vector& vector) const;
    Vector gevmtb(const Matrix& matrix, const Vector& vector,
                  const Vector& bias) const;

    // Reference Qwen2 forward subgraphs using the Metal products above and
    // the existing CPU implementations for RMSNorm, RoPE, mask, Softmax,
    // concatenation, and residual operations.
    Matrix attention(const Matrix& hidden,
                     const Layer& layer,
                     float epsilon,
                     std::size_t d_rope,
                     float theta,
                     std::size_t d_head,
                     Profiler* profiler = nullptr) const;

    Matrix ffn(const Matrix& hidden,
               const Layer& layer,
               float epsilon,
               Profiler* profiler = nullptr) const;

    Matrix prefill(const Matrix& hidden,
                   const Layer& layer,
                   float epsilon,
                   std::size_t d_rope,
                   float theta,
                   std::size_t d_head,
                   Matrix& key_cache,
                   Matrix& value_cache,
                   Profiler* profiler = nullptr) const;

    Vector decode(const Vector& hidden,
                  const Layer& layer,
                  float epsilon,
                  std::size_t d_rope,
                  float theta,
                  std::size_t d_head,
                  Matrix& key_cache,
                  Matrix& value_cache,
                  Profiler* profiler = nullptr) const;

    // Complete no-KV-cache greedy forward used by llm_runtime::forward().
    int32_t forward(const std::vector<int32_t>& token_ids,
                    const llm_runtime& runtime,
                    Profiler* profiler = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace llm

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm {

class Vector {

    public:
    size_t  lens = 0;
    std::vector<float> values;

    Vector() = default;
    Vector(size_t len_count)
        : lens(len_count), values(len_count, 0.0f) {}

    // Normalize this vector in place: x = RMSNorm(x, gamma).
    void rmsnorm(const Vector& gamma, float epsilon = 1e-6f);

    // Apply stable Softmax to this vector in place.
    void softmax();

    // Replace this gate vector in place with SiLU(gate) * up.
    void swiglu(const Vector& up);

    // Apply NeoX RoPE in place. A zero rotary dimension means the whole
    // vector; a smaller dimension leaves the remaining tail unchanged.
    void rope(size_t position,
              float rope_theta = 1000000.0f,
              size_t rotary_dimension = 0);

    // Split contiguous vector ranges into independent per-head vectors.
    // The source vector is unchanged. Each result has length lens / head_count.
    std::vector<Vector> split_heads(size_t head_count) const;
};

class Matrix {

    public:
    size_t  rows = 0;
    size_t  cols = 0;
    std::vector<Vector> values;

    Matrix() = default;
    Matrix(size_t row_count, size_t col_count)
        : rows(row_count), cols(col_count), values(row_count, Vector{col_count}) {}

    // Append one vector as a new row in place. An empty Matrix adopts the
    // vector length as its column count; subsequent rows must match cols.
    void append(const Vector& row);

    // Normalize every row in place using the same gamma vector.
    void rmsnorm(const Vector& gamma, float epsilon = 1e-6f);

    // Apply Vector::softmax independently and in place to every row.
    void softmax();

    // Apply a causal attention mask in place: entries above the main
    // diagonal are set to negative infinity, while each row's prefix remains
    // unchanged.
    void causal_mask();

    // Replace every gate row in place using the corresponding up row.
    void swiglu(const Matrix& up);

    // Apply NeoX RoPE independently to every row in place. The argument is
    // the starting position, so row r uses position + r. A zero rotary
    // dimension means all columns; a smaller dimension leaves the tail
    // unchanged.
    void rope(size_t position,
              float rope_theta = 1000000.0f,
              size_t rotary_dimension = 0);

    // Split contiguous column ranges into independent per-head matrices.
    // The source matrix is unchanged. Each result has shape
    // [rows, cols / head_count].
    std::vector<Matrix> split_heads(size_t head_count) const;

};

class Layer;
class Profiler;

// Look up one embedding row for each token id. The embedding matrix uses
// [vocabulary, embedding] layout and is not a projection weight.
Matrix embedding(const std::vector<int32_t>& token_ids,
                 const Matrix& embedding_weight);

// y = vector^T * matrix; matrix is [input, output], vector is [input].
Vector gevm(const Matrix& matrix, const Vector& vector);
// y = vector^T * matrix + bias; bias is shared by the resulting vector.
Vector gevmb(const Matrix& matrix, const Vector& vector, const Vector& bias);
// y = matrix * vector / sqrt(d_head); matrix is [sequence, head_dim].
Vector gevmts(const Matrix& matrix, const Vector& vector, size_t d_head);
// y = matrix * vector; matrix is [output, input], vector is [input].
Vector gevmt(const Matrix& matrix, const Vector& vector);
Vector gevmtb(const Matrix& matrix, const Vector& vector, const Vector& bias);
Matrix gemm(const Matrix& left, const Matrix& right);
// C = left * right + bias, with bias shared by every output row.
Matrix gemmb(const Matrix& left, const Matrix& right, const Vector& bias);
// C = left * right^T; left is [M, K], right is [N, K].
Matrix gemmt(const Matrix& left, const Matrix& right);
Matrix gemmts(const Matrix& left, const Matrix& right, size_t d_head);
// C = left * right^T + bias, with bias shared by every output row.
Matrix gemmtb(const Matrix& left, const Matrix& right, const Vector& bias);
Matrix gemtm(const Matrix& left, const Matrix& right);
// C = left^T * right + bias, with bias shared by every output row.
Matrix gemtmb(const Matrix& left, const Matrix& right, const Vector& bias);
Matrix gemtmt(const Matrix& left, const Matrix& right);
// C = left^T * right^T + bias, with bias shared by every output row.
Matrix gemtmtb(const Matrix& left, const Matrix& right, const Vector& bias);
    
Matrix attention(const Matrix& hidden,
                 const Layer& layer,
                 float epsilon,
                 size_t d_rope,
                 float theta,
                 size_t d_head,
                 Profiler* profiler = nullptr);

// Execute one Qwen2.5 feed-forward sublayer:
// RMSNorm -> SwiGLU (gate/up) -> down projection -> residual.
Matrix ffn(const Matrix& hidden,
           const Layer& layer,
           float epsilon,
           Profiler* profiler = nullptr);

// Concatenate matrices horizontally. All inputs must have the same row count;
// the result contains their columns in input order.
Matrix concat(const std::vector<Matrix>& matrices);

// Add two matrices elementwise and return the residual result. Both inputs
// must have identical shapes; neither input is modified.
Matrix residual(const Matrix& left, const Matrix& right);

// Concatenate vectors in input order. The result owns independent storage.
Vector concat(const std::vector<Vector>& vectors);

// Add two vectors elementwise and return the residual result. Both inputs
// must have identical lengths; neither input is modified.
Vector residual(const Vector& left, const Vector& right);

// Run one transformer layer over a prompt matrix. kc/vc are replaced with
// [sequence, kv_heads * head_dim] caches; K is stored after per-head RoPE.
Matrix prefill(const Matrix& hidden,
               const Layer& layer,
               float epsilon,
               size_t d_rope,
               float theta,
               size_t d_head,
               Matrix& kc,
               Matrix& vc);

// Run one decode token through one transformer layer. The new rotated K/V
// row is appended to kc/vc, whose previous rows must belong to this layer.
Vector decode(const Vector& hidden,
              const Layer& layer,
              float epsilon,
              size_t d_rope,
              float theta,
              size_t d_head,
              Matrix& kc,
              Matrix& vc);

}

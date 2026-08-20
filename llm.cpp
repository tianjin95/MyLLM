#include "llm.h"
#include "model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace llm {

namespace {

void validate_vector(const Vector& vector, const char* name) {
    if (vector.lens != vector.values.size()) {
        throw std::invalid_argument(std::string(name) + " length does not match its storage");
    }
}

void validate_matrix(const Matrix& matrix, const char* name) {
    if (matrix.rows != matrix.values.size()) {
        throw std::invalid_argument(std::string(name) + " row count does not match its storage");
    }
    for (const Vector& row : matrix.values) {
        validate_vector(row, name);
        if (row.lens != matrix.cols) {
            throw std::invalid_argument(std::string(name) + " column count does not match its rows");
        }
    }
}

} // namespace

Matrix embedding(const std::vector<int32_t>& token_ids,
                 const Matrix& embedding_weight) {
    validate_matrix(embedding_weight, "Embedding weight");
    if (embedding_weight.rows == 0 || embedding_weight.cols == 0) {
        throw std::invalid_argument("Embedding weight cannot be empty");
    }
    if (token_ids.empty()) {
        throw std::invalid_argument("Embedding requires at least one token id");
    }

    Matrix result(token_ids.size(), embedding_weight.cols);
    for (size_t position = 0; position < token_ids.size(); ++position) {
        const int32_t token_id = token_ids[position];
        if (token_id < 0 || static_cast<size_t>(token_id) >= embedding_weight.rows) {
            throw std::out_of_range("Token id is outside the embedding vocabulary");
        }
        std::copy(
            embedding_weight.values[static_cast<size_t>(token_id)].values.begin(),
            embedding_weight.values[static_cast<size_t>(token_id)].values.end(),
            result.values[position].values.begin());
    }
    return result;
}

void Vector::rmsnorm(const Vector& gamma, float epsilon) {
    validate_vector(*this, "RMSNorm input");
    validate_vector(gamma, "RMSNorm gamma");
    if (values.empty()) {
        throw std::invalid_argument("RMSNorm input cannot be empty");
    }
    if (gamma.lens != lens) {
        throw std::invalid_argument("RMSNorm gamma dimension mismatch");
    }
    if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
        throw std::invalid_argument("RMSNorm epsilon must be finite and positive");
    }

    double sum_of_squares = 0.0;
    for (float value : values) {
        sum_of_squares += static_cast<double>(value) * static_cast<double>(value);
    }
    const float mean_square = static_cast<float>(sum_of_squares / static_cast<double>(lens));
    const float inverse_rms = 1.0f / std::sqrt(mean_square + epsilon);

    for (size_t index = 0; index < lens; ++index) {
        values[index] *= inverse_rms * gamma.values[index];
    }
}

void Vector::softmax() {
    validate_vector(*this, "Softmax input");
    if (values.empty()) {
        throw std::invalid_argument("Softmax input cannot be empty");
    }

    const float maximum = *std::max_element(values.begin(), values.end());
    double denominator = 0.0;
    for (size_t index = 0; index < lens; ++index) {
        values[index] = std::exp(values[index] - maximum);
        denominator += static_cast<double>(values[index]);
    }
    if (!std::isfinite(denominator) || denominator == 0.0) {
        throw std::runtime_error("Softmax normalization is not finite");
    }
    for (float& value : values) {
        value = static_cast<float>(static_cast<double>(value) / denominator);
    }
}

void Vector::swiglu(const Vector& up) {
    validate_vector(*this, "SwiGLU gate");
    validate_vector(up, "SwiGLU up");
    if (up.lens != lens) {
        throw std::invalid_argument("SwiGLU dimension mismatch");
    }

    for (size_t index = 0; index < lens; ++index) {
        const float gate = values[index];
        const float sigmoid = gate >= 0.0f
            ? 1.0f / (1.0f + std::exp(-gate))
            : std::exp(gate) / (1.0f + std::exp(gate));
        values[index] = gate * sigmoid * up.values[index];
    }
}

void Vector::rope(size_t position, float rope_theta, size_t rotary_dimension) {
    validate_vector(*this, "RoPE input");
    if (values.empty()) {
        throw std::invalid_argument("RoPE input cannot be empty");
    }
    if (rotary_dimension == 0) {
        rotary_dimension = lens;
    }
    if (rotary_dimension > lens || rotary_dimension == 0 ||
        rotary_dimension % 2 != 0) {
        throw std::invalid_argument("RoPE rotary dimension must be even and fit the vector");
    }
    if (!std::isfinite(rope_theta) || rope_theta <= 0.0f) {
        throw std::invalid_argument("RoPE theta must be finite and positive");
    }

    const size_t half = rotary_dimension / 2;
    for (size_t pair = 0; pair < half; ++pair) {
        const float frequency = std::pow(
            rope_theta, -2.0f * static_cast<float>(pair) /
                            static_cast<float>(rotary_dimension));
        const float angle = static_cast<float>(position) * frequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);

        const size_t second_index = pair + half;
        const float first = values[pair];
        const float second = values[second_index];
        values[pair] = first * cosine - second * sine;
        values[second_index] = first * sine + second * cosine;
    }
}

void Matrix::rmsnorm(const Vector& gamma, float epsilon) {
    validate_matrix(*this, "RMSNorm matrix input");
    validate_vector(gamma, "RMSNorm gamma");
    if (gamma.lens != cols) {
        throw std::invalid_argument("RMSNorm gamma dimension mismatch");
    }

    for (size_t row = 0; row < rows; ++row) {
        values[row].rmsnorm(gamma, epsilon);
    }
}

void Matrix::softmax() {
    validate_matrix(*this, "Softmax matrix input");
    for (size_t row = 0; row < rows; ++row) {
        values[row].softmax();
    }
}

void Matrix::swiglu(const Matrix& up) {
    validate_matrix(*this, "SwiGLU gate matrix");
    validate_matrix(up, "SwiGLU up matrix");
    if (up.rows != rows || up.cols != cols) {
        throw std::invalid_argument("SwiGLU matrix dimension mismatch");
    }

    for (size_t row = 0; row < rows; ++row) {
        values[row].swiglu(up.values[row]);
    }
}

void Matrix::rope(size_t position, float rope_theta, size_t rotary_dimension) {
    validate_matrix(*this, "RoPE matrix input");
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("RoPE matrix cannot be empty");
    }
    if (rotary_dimension == 0) {
        rotary_dimension = cols;
    }
    if (rotary_dimension > cols || rotary_dimension == 0 ||
        rotary_dimension % 2 != 0) {
        throw std::invalid_argument("RoPE rotary dimension must be even and fit the matrix");
    }
    if (rows - 1 > std::numeric_limits<size_t>::max() - position) {
        throw std::invalid_argument("RoPE position overflows matrix row positions");
    }
    for (size_t row = 0; row < rows; ++row) {
        values[row].rope(position + row, rope_theta, rotary_dimension);
    }
}

void Matrix::causal_mask() {
    validate_matrix(*this, "Causal mask input");
    const float masked_value = -std::numeric_limits<float>::infinity();

    for (size_t row = 0; row < rows; ++row) {
        const size_t first_masked_column = row + 1;
        if (first_masked_column < cols) {
            std::fill(
                values[row].values.begin() + first_masked_column,
                values[row].values.end(),
                masked_value);
        }
    }
}

std::vector<Matrix> Matrix::split_heads(size_t head_count) const {
    validate_matrix(*this, "Head split input");
    if (head_count == 0) {
        throw std::invalid_argument("Head count must be greater than zero");
    }
    if (cols == 0) {
        throw std::invalid_argument("Head split input cannot have zero columns");
    }
    if (cols % head_count != 0) {
        throw std::invalid_argument("Matrix column count must be divisible by head count");
    }

    const size_t head_columns = cols / head_count;
    std::vector<Matrix> heads;
    heads.reserve(head_count);

    for (size_t head = 0; head < head_count; ++head) {
        heads.emplace_back(rows, head_columns);
        Matrix& destination = heads.back();
        const size_t source_column = head * head_columns;

        for (size_t row = 0; row < rows; ++row) {
            std::copy_n(
                values[row].values.begin() + source_column,
                head_columns,
                destination.values[row].values.begin());
        }
    }

    return heads;
}

Matrix concat(const std::vector<Matrix>& matrices) {
    if (matrices.empty()) {
        throw std::invalid_argument("Cannot concatenate an empty matrix list");
    }

    const size_t row_count = matrices.front().rows;
    size_t total_columns = 0;
    for (const Matrix& matrix : matrices) {
        validate_matrix(matrix, "Concat input");
        if (matrix.rows != row_count) {
            throw std::invalid_argument("Concat inputs must have the same row count");
        }
        if (matrix.cols > std::numeric_limits<size_t>::max() - total_columns) {
            throw std::invalid_argument("Concat column count overflows size_t");
        }
        total_columns += matrix.cols;
    }

    Matrix result(row_count, total_columns);
    size_t destination_column = 0;
    for (const Matrix& matrix : matrices) {
        for (size_t row = 0; row < row_count; ++row) {
            std::copy(
                matrix.values[row].values.begin(),
                matrix.values[row].values.end(),
                result.values[row].values.begin() + destination_column);
        }
        destination_column += matrix.cols;
    }
    return result;
}

Matrix residual(const Matrix& left, const Matrix& right) {
    validate_matrix(left, "Residual left");
    validate_matrix(right, "Residual right");
    if (left.rows != right.rows || left.cols != right.cols) {
        throw std::invalid_argument("Residual inputs must have identical shapes");
    }

    Matrix result(left.rows, left.cols);
    for (size_t row = 0; row < left.rows; ++row) {
        for (size_t col = 0; col < left.cols; ++col) {
            result.values[row].values[col] =
                left.values[row].values[col] + right.values[row].values[col];
        }
    }
    return result;
}

Vector gevm(const Matrix& matrix, const Vector& vector) {
    Vector result(matrix.cols);
    for(size_t col = 0; col < matrix.cols; ++col) {
        for(size_t row = 0; row < matrix.rows; ++row) {
            result.values[col] += vector.values[row] * matrix.values[row].values[col];
        }
    }
    return result;
}

Vector gevmb(const Matrix& matrix, const Vector& vector, const Vector& bias) {
    Vector result(matrix.cols);
    for(size_t col = 0; col < matrix.cols; ++col) {
        for(size_t row = 0; row < matrix.rows; ++row) {
            result.values[col] += vector.values[row] * matrix.values[row].values[col];
        }
        result.values[col] += bias.values[col];
    }
    return result;
}

Vector gevmt(const Matrix& matrix, const Vector& vector) {
    Vector result(matrix.rows);
    for(size_t row = 0; row < matrix.rows; ++row) {
        for(size_t col = 0; col < matrix.cols; ++col) {
            result.values[row] += vector.values[col] * matrix.values[row].values[col];
        }
    }
    return result;
}

Vector gevmtb(const Matrix& matrix, const Vector& vector, const Vector& bias) {
    Vector result(matrix.rows);
    for(size_t row = 0; row < matrix.rows; ++row) {
        for(size_t col = 0; col < matrix.cols; ++col) {
            result.values[row] += vector.values[col] * matrix.values[row].values[col];
        }
        result.values[row] += bias.values[row];
    }
    return result;
}

Matrix gemm(const Matrix& left, const Matrix& right) {
    Matrix result(left.rows, right.cols);
    for(size_t row = 0; row < left.rows; ++row){
        for(size_t col = 0; col < right.cols; ++col) {
            for(size_t step = 0; step < left.cols; ++step) {
                result.values[row].values[col] += left.values[row].values[step] * right.values[step].values[col];
            }
        }
    }
    return result;
}

Matrix gemmb(const Matrix& left, const Matrix& right, const Vector& bias) {
    validate_matrix(left, "gemmb left");
    validate_matrix(right, "gemmb right");
    validate_vector(bias, "gemmb bias");
    if (left.cols != right.rows) {
        throw std::invalid_argument("gemmb inner dimensions do not match");
    }
    if (bias.lens != right.cols) {
        throw std::invalid_argument("gemmb bias dimension does not match output columns");
    }

    Matrix result(left.rows, right.cols);
    for (size_t row = 0; row < left.rows; ++row) {
        for (size_t col = 0; col < right.cols; ++col) {
            for (size_t step = 0; step < left.cols; ++step) {
                result.values[row].values[col] +=
                    left.values[row].values[step] * right.values[step].values[col];
            }
            result.values[row].values[col] += bias.values[col];
        }
    }
    return result;
}

Matrix gemmt(const Matrix& left, const Matrix& right) {
    Matrix result(left.rows, right.rows);
    for(size_t row = 0; row < left.rows; ++row){
        for(size_t col = 0; col < right.rows; ++col) {
            for(size_t step = 0; step < left.cols; ++step) {
                result.values[row].values[col] += left.values[row].values[step] * right.values[col].values[step];
            }
        }
    }
    return result;
}

Matrix gemmts(const Matrix& left, const Matrix& right, size_t d_head) {
    Matrix result(left.rows, right.rows);
    for(size_t row = 0; row < left.rows; ++row){
        for(size_t col = 0; col < right.rows; ++col) {
            for(size_t step = 0; step < left.cols; ++step) {
                result.values[row].values[col] += left.values[row].values[step] * right.values[col].values[step];
            }
            result.values[row].values[col] /= std::sqrt(static_cast<float>(d_head));
        }
    }
    return result;
}

Matrix gemmtb(const Matrix& left, const Matrix& right, const Vector& bias) {
    validate_matrix(left, "gemmtb left");
    validate_matrix(right, "gemmtb right");
    validate_vector(bias, "gemmtb bias");
    if (left.cols != right.cols) {
        throw std::invalid_argument("gemmtb inner dimensions do not match");
    }
    if (bias.lens != right.rows) {
        throw std::invalid_argument("gemmtb bias dimension does not match output columns");
    }

    Matrix result(left.rows, right.rows);
    for (size_t row = 0; row < left.rows; ++row) {
        for (size_t col = 0; col < right.rows; ++col) {
            for (size_t step = 0; step < left.cols; ++step) {
                result.values[row].values[col] +=
                    left.values[row].values[step] * right.values[col].values[step];
            }
            result.values[row].values[col] += bias.values[col];
        }
    }
    return result;
}

Matrix gemtm(const Matrix& left, const Matrix& right) {
    Matrix result(left.cols, right.cols);
    for(size_t row = 0; row < left.cols; ++row){
        for(size_t col = 0; col < right.cols; ++col) {
            for(size_t step = 0; step < left.rows; ++step) {
                result.values[row].values[col] += left.values[step].values[row] * right.values[step].values[col];
            }
        }
    }
    return result;
}

Matrix gemtmb(const Matrix& left, const Matrix& right, const Vector& bias) {
    validate_matrix(left, "gemtmb left");
    validate_matrix(right, "gemtmb right");
    validate_vector(bias, "gemtmb bias");
    if (left.rows != right.rows) {
        throw std::invalid_argument("gemtmb inner dimensions do not match");
    }
    if (bias.lens != right.cols) {
        throw std::invalid_argument("gemtmb bias dimension does not match output columns");
    }

    Matrix result(left.cols, right.cols);
    for (size_t row = 0; row < left.cols; ++row) {
        for (size_t col = 0; col < right.cols; ++col) {
            for (size_t step = 0; step < left.rows; ++step) {
                result.values[row].values[col] +=
                    left.values[step].values[row] * right.values[step].values[col];
            }
            result.values[row].values[col] += bias.values[col];
        }
    }
    return result;
}

Matrix gemtmt(const Matrix& left, const Matrix& right) {
    Matrix result(left.cols, right.rows);
    for(size_t row = 0; row < left.cols; ++row){
        for(size_t col = 0; col < right.rows; ++col) {
            for(size_t step = 0; step < left.rows; ++step) {
                result.values[row].values[col] += left.values[step].values[row] * right.values[col].values[step];
            }
        }
    }
    return result;
}

Matrix gemtmtb(const Matrix& left, const Matrix& right, const Vector& bias) {
    validate_matrix(left, "gemtmtb left");
    validate_matrix(right, "gemtmtb right");
    validate_vector(bias, "gemtmtb bias");
    if (left.rows != right.cols) {
        throw std::invalid_argument("gemtmtb inner dimensions do not match");
    }
    if (bias.lens != right.rows) {
        throw std::invalid_argument("gemtmtb bias dimension does not match output columns");
    }

    Matrix result(left.cols, right.rows);
    for (size_t row = 0; row < left.cols; ++row) {
        for (size_t col = 0; col < right.rows; ++col) {
            for (size_t step = 0; step < left.rows; ++step) {
                result.values[row].values[col] +=
                    left.values[step].values[row] * right.values[col].values[step];
            }
            result.values[row].values[col] += bias.values[col];
        }
    }
    return result;
}
    
Matrix attention(const Matrix& hidden, const Layer& layer, float epsilon, size_t d_rope, float theta, size_t d_head) {
    if (d_head == 0 || d_rope == 0 || d_rope > d_head || d_rope % 2 != 0) {
        throw std::invalid_argument("invalid Qwen2.5 attention head dimensions");
    }

    // Qwen2.5 pre-attention RMSNorm.
    Matrix A = hidden;
    A.rmsnorm(layer.attn_norm_weight, epsilon);

    // Q/K/V projections. The model loader stores weights as [input, output].
    // Qwen2.5 has Q/K/V bias tensors. Keep the operation visible here and
    // use only the existing GEMM operators.
    Matrix Q = gemmb(A, layer.attn_q_weight, layer.attn_q_bias);
    Matrix K = gemmb(A, layer.attn_k_weight, layer.attn_k_bias);
    Matrix V = gemmb(A, layer.attn_v_weight, layer.attn_v_bias);

    if (Q.cols % d_head != 0 || K.cols % d_head != 0 || V.cols != K.cols) {
        throw std::invalid_argument("Qwen2.5 Q/K/V dimensions are not head-aligned");
    }
    const size_t kv_head_num = K.cols / d_head;
    const size_t q_head_num = Q.cols / d_head;
    if (kv_head_num == 0 || q_head_num == 0 || q_head_num % kv_head_num != 0) {
        throw std::invalid_argument("invalid Qwen2.5 GQA head configuration");
    }

    std::vector<Matrix> HQ = Q.split_heads(q_head_num);
    std::vector<Matrix> HK = K.split_heads(kv_head_num);
    std::vector<Matrix> HV = V.split_heads(kv_head_num);

    // Matrix::rope applies position + row to every row, which is the token
    // position used during a contiguous prefill.
    for (size_t head = 0; head < q_head_num; ++head) {
        HQ[head].rope(0, theta, d_rope);
    }
    for (size_t head = 0; head < kv_head_num; ++head) {
        HK[head].rope(0, theta, d_rope);
    }

    // Causal scaled dot-product attention with GQA.
    std::vector<Matrix> HS;
    const size_t group_size = q_head_num / kv_head_num;
    HS.reserve(q_head_num);
    for (size_t head = 0; head < q_head_num; ++head) {
        const size_t group = head / group_size;
        Matrix S = gemmts(HQ[head], HK[group], d_head);
        S.causal_mask();
        S.softmax();
        Matrix P = gemm(S, HV[group]);
        HS.push_back(P);
    }

    // Concat
    Matrix C = concat(HS);

    // Qwen2.5 has no attention-output bias.
    Matrix O = gemm(C, layer.attn_output_weight);

    // Attention residual.
    return residual(hidden, O);
}

Matrix ffn(const Matrix& hidden, const Layer& layer, float epsilon) {
    // Qwen2.5 pre-FFN RMSNorm.
    Matrix normalized = hidden;
    normalized.rmsnorm(layer.ffn_norm_weight, epsilon);

    // SwiGLU: SiLU(gate) * up. Qwen2.5 has no FFN bias tensors.
    Matrix gate = gemm(normalized, layer.ffn_gate_weight);
    Matrix up = gemm(normalized, layer.ffn_up_weight);
    gate.swiglu(up);

    Matrix down = gemm(gate, layer.ffn_down_weight);
    return residual(hidden, down);
}

}

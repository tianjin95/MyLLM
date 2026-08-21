#include "llm.h"
#include "model.h"
#include "profiler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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

uint64_t matrix_storage_bytes(size_t rows, size_t cols) {
    return static_cast<uint64_t>(rows) *
           static_cast<uint64_t>(cols) * sizeof(float);
}

ProfileMetrics matrix_copy_metrics(size_t rows, size_t cols) {
    return profile_copy_metrics(
        matrix_storage_bytes(rows, cols),
        static_cast<uint64_t>(rows) + 1);
}

ProfileMetrics residual_profile_metrics(size_t rows, size_t cols) {
    const uint64_t bytes = matrix_storage_bytes(rows, cols);
    ProfileMetrics result;
    result.read_bytes = bytes * 2;
    result.write_bytes = bytes;
    result.temporary_bytes = bytes;
    result.allocations = static_cast<uint64_t>(rows) + 1;
    return result;
}

ProfileMetrics mask_profile_metrics(size_t rows, size_t cols) {
    size_t masked_elements = 0;
    for (size_t row = 0; row < rows; ++row) {
        if (row + 1 < cols) {
            masked_elements += cols - row - 1;
        }
    }
    ProfileMetrics result;
    result.write_bytes = static_cast<uint64_t>(masked_elements) * sizeof(float);
    return result;
}

ProfileMetrics attention_profile_metrics(const Matrix& hidden,
                                         const Layer& layer,
                                         size_t d_head) {
    const size_t sequence_length = hidden.rows;
    const size_t d_model = hidden.cols;
    const size_t q_dimension = layer.attn_q_weight.rows;
    const size_t kv_dimension = layer.attn_k_weight.rows;
    const size_t q_heads = q_dimension / d_head;
    const size_t kv_heads = kv_dimension / d_head;

    ProfileMetrics result = matrix_copy_metrics(sequence_length, d_model);
    result += profile_elementwise_metrics(
        sequence_length * d_model, 5, 3, 1);
    result += profile_gemmb_metrics(
        sequence_length, q_dimension, d_model);
    result += profile_gemmb_metrics(
        sequence_length, kv_dimension, d_model);
    result += profile_gemmb_metrics(
        sequence_length, kv_dimension, d_model);

    const uint64_t split_bytes =
        matrix_storage_bytes(sequence_length, q_dimension) +
        2 * matrix_storage_bytes(sequence_length, kv_dimension);
    const uint64_t split_allocations =
        static_cast<uint64_t>(q_heads + 2 * kv_heads) *
        (static_cast<uint64_t>(sequence_length) + 1);
    result += profile_copy_metrics(split_bytes, split_allocations);
    result += profile_elementwise_metrics(
        sequence_length * (q_dimension + kv_dimension), 6, 1, 1);

    for (size_t head = 0; head < q_heads; ++head) {
        ProfileMetrics qk = profile_gemm_metrics(
            sequence_length, sequence_length, d_head);
        qk.flops += static_cast<uint64_t>(sequence_length) * sequence_length;
        result += qk;
        result += mask_profile_metrics(sequence_length, sequence_length);
        result += profile_elementwise_metrics(
            sequence_length * sequence_length, 4, 3, 2);
        result += profile_gemm_metrics(
            sequence_length, d_head, sequence_length);
    }

    const uint64_t head_output_bytes =
        matrix_storage_bytes(sequence_length, q_dimension);
    result += profile_copy_metrics(
        head_output_bytes, static_cast<uint64_t>(q_heads) *
                               (static_cast<uint64_t>(sequence_length) + 1));
    result += profile_copy_metrics(
        head_output_bytes, static_cast<uint64_t>(sequence_length) + 1);
    result += profile_gemm_metrics(
        sequence_length, d_model, q_dimension);
    result += residual_profile_metrics(sequence_length, d_model);
    return result;
}

ProfileMetrics ffn_profile_metrics(const Matrix& hidden, const Layer& layer) {
    const size_t sequence_length = hidden.rows;
    const size_t d_model = hidden.cols;
    const size_t feed_forward = layer.ffn_gate_weight.rows;
    ProfileMetrics result = matrix_copy_metrics(sequence_length, d_model);
    result += profile_elementwise_metrics(
        sequence_length * d_model, 5, 3, 1);
    result += profile_gemm_metrics(
        sequence_length, feed_forward, d_model);
    result += profile_gemm_metrics(
        sequence_length, feed_forward, d_model);
    result += profile_elementwise_metrics(
        sequence_length * feed_forward, 8, 2, 1);
    result += profile_gemm_metrics(
        sequence_length, d_model, feed_forward);
    result += residual_profile_metrics(sequence_length, d_model);
    return result;
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

void Matrix::append(const Vector& row) {
    validate_matrix(*this, "Matrix append input");
    validate_vector(row, "Matrix append row");
    if (rows != 0 && row.lens != cols) {
        throw std::invalid_argument(
            "Matrix append row length does not match matrix columns");
    }
    if (rows == std::numeric_limits<size_t>::max()) {
        throw std::length_error("Matrix append would overflow row count");
    }

    values.push_back(row);
    if (rows == 0) {
        cols = row.lens;
    }
    rows = values.size();
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

std::vector<Vector> Vector::split_heads(size_t head_count) const {
    validate_vector(*this, "Head split input");
    if (head_count == 0) {
        throw std::invalid_argument("Head count must be greater than zero");
    }
    if (lens == 0) {
        throw std::invalid_argument("Head split input cannot be empty");
    }
    if (lens % head_count != 0) {
        throw std::invalid_argument(
            "Vector length must be divisible by head count");
    }

    const size_t head_length = lens / head_count;
    std::vector<Vector> heads;
    heads.reserve(head_count);
    for (size_t head = 0; head < head_count; ++head) {
        Vector destination(head_length);
        const size_t source_offset = head * head_length;
        std::copy_n(
            values.begin() + source_offset,
            head_length,
            destination.values.begin());
        heads.push_back(std::move(destination));
    }
    return heads;
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

Vector concat(const std::vector<Vector>& vectors) {
    if (vectors.empty()) {
        throw std::invalid_argument("Cannot concatenate an empty vector list");
    }

    size_t total_length = 0;
    for (const Vector& vector : vectors) {
        validate_vector(vector, "Vector concat input");
        if (vector.lens > std::numeric_limits<size_t>::max() - total_length) {
            throw std::invalid_argument("Vector concat length overflows size_t");
        }
        total_length += vector.lens;
    }

    Vector result(total_length);
    size_t destination_offset = 0;
    for (const Vector& vector : vectors) {
        std::copy(
            vector.values.begin(),
            vector.values.end(),
            result.values.begin() + destination_offset);
        destination_offset += vector.lens;
    }
    return result;
}

Vector residual(const Vector& left, const Vector& right) {
    validate_vector(left, "Vector residual left");
    validate_vector(right, "Vector residual right");
    if (left.lens != right.lens) {
        throw std::invalid_argument(
            "Vector residual inputs must have identical lengths");
    }

    Vector result(left.lens);
    for (size_t index = 0; index < left.lens; ++index) {
        result.values[index] = left.values[index] + right.values[index];
    }
    return result;
}

Vector gevm(const Matrix& matrix, const Vector& vector) {
    validate_matrix(matrix, "gevm matrix");
    validate_vector(vector, "gevm vector");
    if (matrix.rows != vector.lens) {
        throw std::invalid_argument("gevm inner dimensions do not match");
    }

    Vector result(matrix.cols);
    for (size_t col = 0; col < matrix.cols; ++col) {
        for (size_t row = 0; row < matrix.rows; ++row) {
            result.values[col] += vector.values[row] * matrix.values[row].values[col];
        }
    }
    return result;
}

Vector gevmb(const Matrix& matrix, const Vector& vector, const Vector& bias) {
    validate_matrix(matrix, "gevmb matrix");
    validate_vector(vector, "gevmb vector");
    validate_vector(bias, "gevmb bias");
    if (matrix.rows != vector.lens) {
        throw std::invalid_argument("gevmb inner dimensions do not match");
    }
    if (bias.lens != matrix.cols) {
        throw std::invalid_argument("gevmb bias dimension does not match output");
    }

    Vector result(matrix.cols);
    for (size_t col = 0; col < matrix.cols; ++col) {
        for (size_t row = 0; row < matrix.rows; ++row) {
            result.values[col] += vector.values[row] * matrix.values[row].values[col];
        }
        result.values[col] += bias.values[col];
    }
    return result;
}

Vector gevmts(const Matrix& matrix, const Vector& vector, size_t d_head) {
    validate_matrix(matrix, "gevmts matrix");
    validate_vector(vector, "gevmts vector");
    if (d_head == 0) {
        throw std::invalid_argument("gevmts head dimension must be greater than zero");
    }
    if (matrix.cols != vector.lens || matrix.cols != d_head) {
        throw std::invalid_argument("gevmts dimensions do not match");
    }

    Vector result(matrix.rows);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    for (size_t row = 0; row < matrix.rows; ++row) {
        float dot = 0.0f;
        for (size_t col = 0; col < matrix.cols; ++col) {
            dot += matrix.values[row].values[col] * vector.values[col];
        }
        result.values[row] = dot * scale;
    }
    return result;
}

Vector gevmt(const Matrix& matrix, const Vector& vector) {
    validate_matrix(matrix, "gevmt matrix");
    validate_vector(vector, "gevmt vector");
    if (matrix.cols != vector.lens) {
        throw std::invalid_argument("gevmt inner dimensions do not match");
    }

    Vector result(matrix.rows);
    for (size_t row = 0; row < matrix.rows; ++row) {
        for (size_t col = 0; col < matrix.cols; ++col) {
            result.values[row] += vector.values[col] * matrix.values[row].values[col];
        }
    }
    return result;
}

Vector gevmtb(const Matrix& matrix, const Vector& vector, const Vector& bias) {
    validate_matrix(matrix, "gevmtb matrix");
    validate_vector(vector, "gevmtb vector");
    validate_vector(bias, "gevmtb bias");
    if (matrix.cols != vector.lens) {
        throw std::invalid_argument("gevmtb inner dimensions do not match");
    }
    if (bias.lens != matrix.rows) {
        throw std::invalid_argument("gevmtb bias dimension does not match output");
    }

    Vector result(matrix.rows);
    for (size_t row = 0; row < matrix.rows; ++row) {
        for (size_t col = 0; col < matrix.cols; ++col) {
            result.values[row] += vector.values[col] * matrix.values[row].values[col];
        }
        result.values[row] += bias.values[row];
    }
    return result;
}

Matrix gemm(const Matrix& left, const Matrix& right) {
    validate_matrix(left, "gemm left");
    validate_matrix(right, "gemm right");
    if (left.cols != right.rows) {
        throw std::invalid_argument("gemm inner dimensions do not match");
    }

    Matrix result(left.rows, right.cols);
    for (size_t row = 0; row < left.rows; ++row) {
        for (size_t col = 0; col < right.cols; ++col) {
            for (size_t step = 0; step < left.cols; ++step) {
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
    validate_matrix(left, "gemmt left");
    validate_matrix(right, "gemmt right");
    if (left.cols != right.cols) {
        throw std::invalid_argument("gemmt inner dimensions do not match");
    }

    Matrix result(left.rows, right.rows);
    for (size_t row = 0; row < left.rows; ++row) {
        for (size_t col = 0; col < right.rows; ++col) {
            for (size_t step = 0; step < left.cols; ++step) {
                result.values[row].values[col] += left.values[row].values[step] * right.values[col].values[step];
            }
        }
    }
    return result;
}

Matrix gemmts(const Matrix& left, const Matrix& right, size_t d_head) {
    validate_matrix(left, "gemmts left");
    validate_matrix(right, "gemmts right");
    if (left.cols != right.cols) {
        throw std::invalid_argument("gemmts inner dimensions do not match");
    }
    if (d_head == 0 || left.cols != d_head) {
        throw std::invalid_argument("gemmts head dimension does not match inputs");
    }

    Matrix result(left.rows, right.rows);
    for (size_t row = 0; row < left.rows; ++row) {
        for (size_t col = 0; col < right.rows; ++col) {
            for (size_t step = 0; step < left.cols; ++step) {
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
    
Matrix attention(const Matrix& hidden,
                 const Layer& layer,
                 float epsilon,
                 size_t d_rope,
                 float theta,
                 size_t d_head,
                 Profiler* profiler) {
    if (d_head == 0 || d_rope == 0 || d_rope > d_head || d_rope % 2 != 0) {
        throw std::invalid_argument("invalid Qwen2.5 attention head dimensions");
    }

    Profiler::Scope attention_total(
        profiler, "attention.total", attention_profile_metrics(hidden, layer, d_head));

    // Qwen2.5 pre-attention RMSNorm.
    Matrix A;
    {
        ProfileMetrics metrics = matrix_copy_metrics(hidden.rows, hidden.cols);
        metrics += profile_elementwise_metrics(
            hidden.rows * hidden.cols, 5, 3, 1);
        Profiler::Scope scope(profiler, "attention.rmsnorm", metrics);
        A = hidden;
        A.rmsnorm(layer.attn_norm_weight, epsilon);
    }

    // Q/K/V projections. The model loader keeps weights as [output, input],
    // so input * weight^T is expressed by the existing GEMMTB operator.
    // Qwen2.5 has Q/K/V bias tensors. Keep the operation visible here.
    Matrix Q;
    {
        Profiler::Scope scope(
            profiler, "attention.q_proj",
            profile_gemmb_metrics(A.rows, layer.attn_q_weight.rows,
                                  A.cols));
        Q = gemmtb(A, layer.attn_q_weight, layer.attn_q_bias);
    }
    Matrix K;
    {
        Profiler::Scope scope(
            profiler, "attention.k_proj",
            profile_gemmb_metrics(A.rows, layer.attn_k_weight.rows,
                                  A.cols));
        K = gemmtb(A, layer.attn_k_weight, layer.attn_k_bias);
    }
    Matrix V;
    {
        Profiler::Scope scope(
            profiler, "attention.v_proj",
            profile_gemmb_metrics(A.rows, layer.attn_v_weight.rows,
                                  A.cols));
        V = gemmtb(A, layer.attn_v_weight, layer.attn_v_bias);
    }

    if (Q.cols % d_head != 0 || K.cols % d_head != 0 || V.cols != K.cols) {
        throw std::invalid_argument("Qwen2.5 Q/K/V dimensions are not head-aligned");
    }
    const size_t kv_head_num = K.cols / d_head;
    const size_t q_head_num = Q.cols / d_head;
    if (kv_head_num == 0 || q_head_num == 0 || q_head_num % kv_head_num != 0) {
        throw std::invalid_argument("invalid Qwen2.5 GQA head configuration");
    }

    std::vector<Matrix> HQ;
    std::vector<Matrix> HK;
    std::vector<Matrix> HV;
    {
        const uint64_t split_bytes =
            matrix_storage_bytes(Q.rows, Q.cols) +
            matrix_storage_bytes(K.rows, K.cols) +
            matrix_storage_bytes(V.rows, V.cols);
        const uint64_t split_allocations =
            static_cast<uint64_t>(q_head_num + 2 * kv_head_num) *
            (static_cast<uint64_t>(Q.rows) + 1);
        Profiler::Scope scope(
            profiler, "attention.split_heads",
            profile_copy_metrics(split_bytes, split_allocations));
        HQ = Q.split_heads(q_head_num);
        HK = K.split_heads(kv_head_num);
        HV = V.split_heads(kv_head_num);
    }

    // Matrix::rope applies position + row to every row, which is the token
    // position used during a contiguous prefill.
    {
        ProfileMetrics metrics = profile_elementwise_metrics(
            Q.rows * (Q.cols + K.cols), 6, 1, 1);
        Profiler::Scope scope(profiler, "attention.rope", metrics);
        for (size_t head = 0; head < q_head_num; ++head) {
            HQ[head].rope(0, theta, d_rope);
        }
        for (size_t head = 0; head < kv_head_num; ++head) {
            HK[head].rope(0, theta, d_rope);
        }
    }

    // Causal scaled dot-product attention with GQA.
    std::vector<Matrix> HS;
    const size_t group_size = q_head_num / kv_head_num;
    HS.reserve(q_head_num);
    for (size_t head = 0; head < q_head_num; ++head) {
        const size_t group = head / group_size;
        Matrix S;
        {
            ProfileMetrics metrics = profile_gemm_metrics(
                HQ[head].rows, HK[group].rows, d_head);
            metrics.flops += static_cast<uint64_t>(HQ[head].rows) *
                             HK[group].rows;
            Profiler::Scope scope(profiler, "attention.qk", metrics);
            S = gemmts(HQ[head], HK[group], d_head);
        }
        {
            Profiler::Scope scope(
                profiler, "attention.mask",
                mask_profile_metrics(S.rows, S.cols));
            S.causal_mask();
        }
        {
            Profiler::Scope scope(
                profiler, "attention.softmax",
                profile_elementwise_metrics(S.rows * S.cols, 4, 3, 2));
            S.softmax();
        }
        Matrix P;
        {
            Profiler::Scope scope(
                profiler, "attention.av",
                profile_gemm_metrics(S.rows, HV[group].cols, S.cols));
            P = gemm(S, HV[group]);
        }
        {
            Profiler::Scope scope(
                profiler, "attention.head_copy",
                matrix_copy_metrics(P.rows, P.cols));
            HS.push_back(P);
        }
    }

    // Concat
    Matrix C;
    {
        Profiler::Scope scope(
            profiler, "attention.concat",
            matrix_copy_metrics(hidden.rows, Q.cols));
        C = concat(HS);
    }

    // Qwen2.5 has no attention-output bias.
    Matrix O;
    {
        Profiler::Scope scope(
            profiler, "attention.output_proj",
            profile_gemm_metrics(C.rows, layer.attn_output_weight.rows,
                                 C.cols));
        O = gemmt(C, layer.attn_output_weight);
    }

    // Attention residual.
    Matrix result;
    {
        Profiler::Scope scope(
            profiler, "attention.residual",
            residual_profile_metrics(hidden.rows, hidden.cols));
        result = residual(hidden, O);
    }
    return result;
}

Matrix ffn(const Matrix& hidden,
           const Layer& layer,
           float epsilon,
           Profiler* profiler) {
    Profiler::Scope ffn_total(
        profiler, "ffn.total", ffn_profile_metrics(hidden, layer));

    // Qwen2.5 pre-FFN RMSNorm.
    Matrix normalized;
    {
        ProfileMetrics metrics = matrix_copy_metrics(hidden.rows, hidden.cols);
        metrics += profile_elementwise_metrics(
            hidden.rows * hidden.cols, 5, 3, 1);
        Profiler::Scope scope(profiler, "ffn.rmsnorm", metrics);
        normalized = hidden;
        normalized.rmsnorm(layer.ffn_norm_weight, epsilon);
    }

    // SwiGLU: SiLU(gate) * up. Qwen2.5 has no FFN bias tensors.
    Matrix gate;
    {
        Profiler::Scope scope(
            profiler, "ffn.gate_proj",
            profile_gemm_metrics(normalized.rows,
                                 layer.ffn_gate_weight.rows,
                                 normalized.cols));
        gate = gemmt(normalized, layer.ffn_gate_weight);
    }
    Matrix up;
    {
        Profiler::Scope scope(
            profiler, "ffn.up_proj",
            profile_gemm_metrics(normalized.rows,
                                 layer.ffn_up_weight.rows,
                                 normalized.cols));
        up = gemmt(normalized, layer.ffn_up_weight);
    }
    {
        Profiler::Scope scope(
            profiler, "ffn.swiglu",
            profile_elementwise_metrics(gate.rows * gate.cols, 8, 2, 1));
        gate.swiglu(up);
    }

    Matrix down;
    {
        Profiler::Scope scope(
            profiler, "ffn.down_proj",
            profile_gemm_metrics(gate.rows,
                                 layer.ffn_down_weight.rows,
                                 gate.cols));
        down = gemmt(gate, layer.ffn_down_weight);
    }
    Matrix result;
    {
        Profiler::Scope scope(
            profiler, "ffn.residual",
            residual_profile_metrics(hidden.rows, hidden.cols));
        result = residual(hidden, down);
    }
    return result;
}

namespace {

std::pair<size_t, size_t> validate_kv_attention_shape(const Layer& layer,
                                                       float theta,
                                                       size_t d_rope,
                                                       size_t d_head) {
    if (d_head == 0 || d_rope == 0 || d_rope > d_head || d_rope % 2 != 0) {
        throw std::invalid_argument("invalid KV attention head dimensions");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::invalid_argument("KV attention RoPE theta must be finite and positive");
    }

    const size_t q_dimension = layer.attn_q_weight.rows;
    const size_t k_dimension = layer.attn_k_weight.rows;
    const size_t v_dimension = layer.attn_v_weight.rows;
    if (q_dimension == 0 || k_dimension == 0 || k_dimension != v_dimension ||
        q_dimension % d_head != 0 || k_dimension % d_head != 0 ||
        layer.attn_q_weight.cols != layer.attn_k_weight.cols ||
        layer.attn_q_weight.cols != layer.attn_v_weight.cols) {
        throw std::invalid_argument("invalid KV attention projection shapes");
    }

    const size_t q_head_num = q_dimension / d_head;
    const size_t kv_head_num = k_dimension / d_head;
    if (q_head_num == 0 || kv_head_num == 0 ||
        q_head_num % kv_head_num != 0) {
        throw std::invalid_argument("invalid GQA head configuration");
    }
    return {q_head_num, kv_head_num};
}

void validate_kv_cache(const Matrix& kc,
                       const Matrix& vc,
                       size_t kv_dimension) {
    validate_matrix(kc, "Key cache");
    validate_matrix(vc, "Value cache");
    if (kc.rows != vc.rows || kc.cols != vc.cols) {
        throw std::invalid_argument("Key and value caches must have identical shapes");
    }
    if ((kc.rows != 0 || kc.cols != 0) && kc.cols != kv_dimension) {
        throw std::invalid_argument("Key cache column count does not match KV projection");
    }
}

} // namespace

Matrix prefill(const Matrix& hidden,
               const Layer& layer,
               float epsilon,
               size_t d_rope,
               float theta,
               size_t d_head,
               Matrix& kc,
               Matrix& vc) {
    validate_matrix(hidden, "Prefill hidden input");
    if (hidden.rows == 0 || hidden.cols == 0) {
        throw std::invalid_argument("Prefill hidden input cannot be empty");
    }

    const auto [q_head_num, kv_head_num] =
        validate_kv_attention_shape(layer, theta, d_rope, d_head);
    if (hidden.cols != layer.attn_q_weight.cols) {
        throw std::invalid_argument("Prefill hidden dimension does not match attention input");
    }

    // Attention pre-normalization and projections.
    Matrix A = hidden;
    A.rmsnorm(layer.attn_norm_weight, epsilon);

    // Qwen2.5 defines bias only for the Q/K/V projections.
    Matrix Q = gemmtb(A, layer.attn_q_weight, layer.attn_q_bias);
    kc = gemmtb(A, layer.attn_k_weight, layer.attn_k_bias);
    vc = gemmtb(A, layer.attn_v_weight, layer.attn_v_bias);

    // RoPE is applied independently inside each head. The cache stores the
    // already-rotated K rows so decode never rotates historical keys again.
    std::vector<Matrix> HQ = Q.split_heads(q_head_num);
    std::vector<Matrix> HK = kc.split_heads(kv_head_num);
    std::vector<Matrix> HV = vc.split_heads(kv_head_num);
    for (Matrix& head : HQ) {
        head.rope(0, theta, d_rope);
    }
    for (Matrix& head : HK) {
        head.rope(0, theta, d_rope);
    }
    kc = concat(HK);

    // Causal GQA attention. Every query head shares the corresponding KV
    // head, while prefill still computes all query positions at once.
    std::vector<Matrix> HS;
    const size_t group_size = q_head_num / kv_head_num;
    HS.reserve(q_head_num);
    for (size_t head = 0; head < q_head_num; ++head) {
        const size_t group = head / group_size;
        Matrix scores = gemmts(HQ[head], HK[group], d_head);
        scores.causal_mask();
        scores.softmax();
        HS.push_back(gemm(scores, HV[group]));
    }

    Matrix C = concat(HS);
    Matrix O = gemmt(C, layer.attn_output_weight);
    Matrix X = residual(hidden, O);

    // Qwen2.5 FFN: RMSNorm -> gate/up -> SwiGLU -> down -> residual.
    Matrix F = X;
    F.rmsnorm(layer.ffn_norm_weight, epsilon);
    Matrix G = gemmt(F, layer.ffn_gate_weight);
    Matrix U = gemmt(F, layer.ffn_up_weight);
    G.swiglu(U);
    Matrix D = gemmt(G, layer.ffn_down_weight);
    return residual(X, D);
}

Vector decode(const Vector& hidden,
              const Layer& layer,
              float epsilon,
              size_t d_rope,
              float theta,
              size_t d_head,
              Matrix& kc,
              Matrix& vc) {
    validate_vector(hidden, "Decode hidden input");
    if (hidden.lens == 0) {
        throw std::invalid_argument("Decode hidden input cannot be empty");
    }

    const auto [q_head_num, kv_head_num] =
        validate_kv_attention_shape(layer, theta, d_rope, d_head);
    if (hidden.lens != layer.attn_q_weight.cols) {
        throw std::invalid_argument("Decode hidden dimension does not match attention input");
    }
    validate_kv_cache(kc, vc, layer.attn_k_weight.rows);

    // The current cache length is the absolute position of this new token.
    const size_t position = kc.rows;

    Vector A = hidden;
    A.rmsnorm(layer.attn_norm_weight, epsilon);
    // Qwen2.5 defines bias only for the Q/K/V projections.
    Vector Q = gevmtb(layer.attn_q_weight, A, layer.attn_q_bias);
    Vector K = gevmtb(layer.attn_k_weight, A, layer.attn_k_bias);
    Vector V = gevmtb(layer.attn_v_weight, A, layer.attn_v_bias);

    std::vector<Vector> HQ = Q.split_heads(q_head_num);
    for (Vector& head : HQ) {
        head.rope(position, theta, d_rope);
    }

    // Rotate only the new K row, then append it. Historical cache rows are
    // already rotated by prefill or an earlier decode step.
    std::vector<Vector> new_k_heads = K.split_heads(kv_head_num);
    for (Vector& head : new_k_heads) {
        head.rope(position, theta, d_rope);
    }
    kc.append(concat(new_k_heads));
    vc.append(V);

    const std::vector<Matrix> HK = kc.split_heads(kv_head_num);
    const std::vector<Matrix> HV = vc.split_heads(kv_head_num);

    // Decode has no future positions in the cache, so a causal mask is not
    // needed. The score vector covers exactly the valid prefix [0, position].
    std::vector<Vector> HS;
    const size_t group_size = q_head_num / kv_head_num;
    HS.reserve(q_head_num);
    for (size_t head = 0; head < q_head_num; ++head) {
        const size_t group = head / group_size;
        Vector scores = gevmts(HK[group], HQ[head], d_head);
        scores.softmax();
        HS.push_back(gevm(HV[group], scores));
    }

    Vector C = concat(HS);
    Vector O = gevmt(layer.attn_output_weight, C);
    Vector X = residual(hidden, O);

    Vector F = X;
    F.rmsnorm(layer.ffn_norm_weight, epsilon);
    Vector G = gevmt(layer.ffn_gate_weight, F);
    Vector U = gevmt(layer.ffn_up_weight, F);
    G.swiglu(U);
    Vector D = gevmt(layer.ffn_down_weight, G);
    return residual(X, D);
}

}

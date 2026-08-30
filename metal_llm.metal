#include <metal_stdlib>

using namespace metal;

// All strides are measured in FP32 elements, not bytes.  A transpose flag
// changes how a physical row-major tensor is viewed by the logical product:
// lhs(row, k) = lhs[k, row] when lhs_transposed is set, and similarly for rhs.
struct MetalMatmulParams {
    uint m;
    uint n;
    uint k;
    uint lhs_stride;
    uint rhs_stride;
    uint lhs_transposed;
    uint rhs_transposed;
    uint has_bias;
    float scale;
};

struct MetalGemvParams {
    uint output_size;
    uint input_size;
    uint matrix_stride;
    uint matrix_transposed;
    uint has_bias;
    float scale;
};

struct MetalElementwiseParams {
    uint rows;
    uint cols;
};

struct MetalRmsNormParams {
    uint rows;
    uint cols;
    float epsilon;
};

// Head-wise RoPE for packed Q/K projections. The physical tensor remains
// [rows, head_count * head_dim], but the rotary pair is reset at every head
// boundary. This avoids materializing one GPU buffer per attention head.
struct MetalRopeHeadsParams {
    uint rows;
    uint cols;
    uint head_dim;
    uint rotary_dimension;
    uint position;
    float theta;
};

struct MetalEmbeddingParams {
    uint sequence_length;
    uint embedding_size;
    uint vocabulary_size;
};

struct MetalKVCacheWriteParams {
    uint source_rows;
    uint source_cols;
    uint cache_stride;
    uint position;
};

struct MetalKVCacheQKParams {
    uint query_rows;
    uint key_length;
    uint cache_stride;
    uint query_stride;
    uint head_dim;
    uint cache_head;
    uint query_position;
    float scale;
};

struct MetalKVCacheAVParams {
    uint query_rows;
    uint key_length;
    uint cache_stride;
    uint score_stride;
    uint head_dim;
    uint cache_head;
    uint output_stride;
    uint output_offset;
};

struct MetalArgmaxParams {
    uint length;
};

#define MYLLM_TILE_M 16
#define MYLLM_TILE_N 16
#define MYLLM_TILE_K 16
#define MYLLM_ELEMENTWISE_THREADS 256

inline float load_lhs(device const float * source,
                      constant MetalMatmulParams & params,
                      uint row,
                      uint k) {
    return params.lhs_transposed != 0
        ? source[k * params.lhs_stride + row]
        : source[row * params.lhs_stride + k];
}

inline float load_rhs(device const float * source,
                      constant MetalMatmulParams & params,
                      uint k,
                      uint col) {
    return params.rhs_transposed != 0
        ? source[col * params.rhs_stride + k]
        : source[k * params.rhs_stride + col];
}

// One 16x16 threadgroup computes one output tile.  Each thread owns one
// output element and cooperatively stages both operands through threadgroup
// memory for every K tile.
kernel void metal_gemm_f32(
        device const float * lhs [[buffer(0)]],
        device const float * rhs [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalMatmulParams & params [[buffer(4)]],
        uint3 local_id [[thread_position_in_threadgroup]],
        uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float lhs_tile[MYLLM_TILE_M][MYLLM_TILE_K];
    threadgroup float rhs_tile[MYLLM_TILE_K][MYLLM_TILE_N];

    const uint row = group_id.y * MYLLM_TILE_M + local_id.y;
    const uint col = group_id.x * MYLLM_TILE_N + local_id.x;
    float accumulator = 0.0f;

    for (uint k0 = 0; k0 < params.k; k0 += MYLLM_TILE_K) {
        const uint lhs_k = k0 + local_id.x;
        const uint rhs_k = k0 + local_id.y;

        if (row < params.m && lhs_k < params.k) {
            lhs_tile[local_id.y][local_id.x] =
                load_lhs(lhs, params, row, lhs_k);
        } else {
            lhs_tile[local_id.y][local_id.x] = 0.0f;
        }

        if (rhs_k < params.k && col < params.n) {
            rhs_tile[local_id.y][local_id.x] =
                load_rhs(rhs, params, rhs_k, col);
        } else {
            rhs_tile[local_id.y][local_id.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint remaining_k = params.k - k0;
        const uint tile_k_count = remaining_k < uint(MYLLM_TILE_K)
            ? remaining_k
            : uint(MYLLM_TILE_K);
        for (uint tile_k = 0; tile_k < tile_k_count; ++tile_k) {
            accumulator += lhs_tile[local_id.y][tile_k] *
                           rhs_tile[tile_k][local_id.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < params.m && col < params.n) {
        float value = accumulator * params.scale;
        if (params.has_bias != 0) {
            value += bias[col];
        }
        output[row * params.n + col] = value;
    }
}

// One thread computes one output element.  This is intentionally a separate
// kernel from GEMM: for Batch=1 the output width supplies parallelism while a
// thread performs a contiguous K reduction without a cross-thread reduction.
kernel void metal_gevm_f32(
        device const float * matrix [[buffer(0)]],
        device const float * vector [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalGemvParams & params [[buffer(4)]],
        uint output_index [[thread_position_in_grid]]) {
    if (output_index >= params.output_size) {
        return;
    }

    float accumulator = 0.0f;
    for (uint k = 0; k < params.input_size; ++k) {
        const float weight = params.matrix_transposed != 0
            ? matrix[output_index * params.matrix_stride + k]
            : matrix[k * params.matrix_stride + output_index];
        accumulator += vector[k] * weight;
    }

    float value = accumulator * params.scale;
    if (params.has_bias != 0) {
        value += bias[output_index];
    }
    output[output_index] = value;
}

// One threadgroup handles one row. The fixed 256-thread reduction is shared
// by the matrix and vector RMSNorm paths (vectors use rows == 1).
kernel void metal_rmsnorm_f32(
        device const float * input [[buffer(0)]],
        device const float * gamma [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalRmsNormParams & params [[buffer(3)]],
        uint local_index [[thread_index_in_threadgroup]],
        uint3 group_id [[threadgroup_position_in_grid]]) {
    const uint row = group_id.x;
    if (row >= params.rows) {
        return;
    }

    threadgroup float partial[MYLLM_ELEMENTWISE_THREADS];
    float sum_of_squares = 0.0f;
    const uint row_offset = row * params.cols;
    for (uint col = local_index; col < params.cols;
         col += MYLLM_ELEMENTWISE_THREADS) {
        const float value = input[row_offset + col];
        sum_of_squares += value * value;
    }
    partial[local_index] = sum_of_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = MYLLM_ELEMENTWISE_THREADS / 2; stride > 0;
         stride >>= 1) {
        if (local_index < stride) {
            partial[local_index] += partial[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inverse_rms = rsqrt(
        partial[0] / float(params.cols) + params.epsilon);
    for (uint col = local_index; col < params.cols;
         col += MYLLM_ELEMENTWISE_THREADS) {
        output[row_offset + col] =
            input[row_offset + col] * inverse_rms * gamma[col];
    }
}

// Stable row-wise softmax. A vector is represented as a one-row matrix.
kernel void metal_softmax_f32(
        device const float * input [[buffer(0)]],
        device float * output [[buffer(1)]],
        constant MetalElementwiseParams & params [[buffer(2)]],
        uint local_index [[thread_index_in_threadgroup]],
        uint3 group_id [[threadgroup_position_in_grid]]) {
    const uint row = group_id.x;
    if (row >= params.rows) {
        return;
    }

    threadgroup float partial[MYLLM_ELEMENTWISE_THREADS];
    const uint row_offset = row * params.cols;
    float local_maximum = -INFINITY;
    for (uint col = local_index; col < params.cols;
         col += MYLLM_ELEMENTWISE_THREADS) {
        const float value = input[row_offset + col];
        if (value > local_maximum) {
            local_maximum = value;
        }
    }
    partial[local_index] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = MYLLM_ELEMENTWISE_THREADS / 2; stride > 0;
         stride >>= 1) {
        if (local_index < stride &&
            partial[local_index + stride] > partial[local_index]) {
            partial[local_index] = partial[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float maximum = partial[0];
    float local_sum = 0.0f;
    for (uint col = local_index; col < params.cols;
         col += MYLLM_ELEMENTWISE_THREADS) {
        const float value = input[row_offset + col];
        if (value != -INFINITY) {
            local_sum += exp(value - maximum);
        }
    }
    partial[local_index] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = MYLLM_ELEMENTWISE_THREADS / 2; stride > 0;
         stride >>= 1) {
        if (local_index < stride) {
            partial[local_index] += partial[local_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float denominator = partial[0];
    for (uint col = local_index; col < params.cols;
         col += MYLLM_ELEMENTWISE_THREADS) {
        const float value = input[row_offset + col];
        const float numerator = value == -INFINITY
            ? 0.0f
            : exp(value - maximum);
        output[row_offset + col] = denominator > 0.0f
            ? numerator / denominator
            : 0.0f;
    }
}

// Flat elementwise dispatch: thread `index` owns one [row, ffn_col] element.
// There is no communication between threads. The branch keeps sigmoid stable
// for large negative gate values.
kernel void metal_swiglu_f32(
        device const float * gate [[buffer(0)]],
        device const float * up [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalElementwiseParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index >= total) {
        return;
    }

    const float gate_value = gate[index];
    const float sigmoid = gate_value >= 0.0f
        ? 1.0f / (1.0f + exp(-gate_value))
        : exp(gate_value) / (1.0f + exp(gate_value));
    output[index] = gate_value * sigmoid * up[index];
}

// Flat elementwise dispatch: one thread performs one FP32 residual addition.
kernel void metal_residual_f32(
        device const float * left [[buffer(0)]],
        device const float * right [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalElementwiseParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index < total) {
        output[index] = left[index] + right[index];
    }
}

// Apply RoPE independently inside every packed attention head. One thread
// owns one element, and the output has the same row-major layout as input.
kernel void metal_rope_heads_f32(
        device const float * input [[buffer(0)]],
        device float * output [[buffer(1)]],
        constant MetalRopeHeadsParams & params [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index >= total || params.head_dim == 0) {
        return;
    }

    const uint row = index / params.cols;
    const uint col = index - row * params.cols;
    const uint head_start = (col / params.head_dim) * params.head_dim;
    const uint local_col = col - head_start;
    const uint rotary = params.rotary_dimension == 0
        ? params.head_dim
        : params.rotary_dimension;
    if (local_col >= rotary) {
        output[index] = input[index];
        return;
    }

    const uint rotary_half = rotary / 2;
    const uint pair = local_col < rotary_half
        ? local_col
        : local_col - rotary_half;
    const float frequency = pow(
        params.theta,
        -2.0f * float(pair) / float(rotary));
    const float angle = float(params.position + row) * frequency;
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const uint first_index = row * params.cols + head_start + pair;
    const uint second_index = first_index + rotary_half;
    const float component_a = input[first_index];
    const float component_b = input[second_index];
    output[index] = local_col < rotary_half
        ? component_a * cosine - component_b * sine
        : component_a * sine + component_b * cosine;
}

// Copy projected K and V matrices into their preallocated caches at
// `position`. Both sources are contiguous [source_rows, source_cols], while
// each destination has a fixed row stride equal to the complete KV width.
kernel void metal_kv_cache_write_f32(
        device const float * key_source [[buffer(0)]],
        device const float * value_source [[buffer(1)]],
        device float * key_cache [[buffer(2)]],
        device float * value_cache [[buffer(3)]],
        constant MetalKVCacheWriteParams & params [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.source_rows * params.source_cols;
    if (index >= total) {
        return;
    }

    const uint row = index / params.source_cols;
    const uint col = index - row * params.source_cols;
    const uint destination =
        (params.position + row) * params.cache_stride + col;
    key_cache[destination] = key_source[index];
    value_cache[destination] = value_source[index];
}

// Compute scaled QK scores directly against one KV head in the fixed cache.
// Output is [query_rows, key_length]. One thread owns one score element.
// `query_position` is the absolute position of query row zero, allowing the
// same kernel to serve both a contiguous prefill and one-token decode.
kernel void metal_kv_cache_qk_f32(
        device const float * query [[buffer(0)]],
        device const float * key_cache [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalKVCacheQKParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.query_rows * params.key_length;
    if (index >= total) {
        return;
    }

    const uint query_row = index / params.key_length;
    const uint key_row = index - query_row * params.key_length;
    const uint query_offset = query_row * params.query_stride;
    const uint key_offset = key_row * params.cache_stride +
                            params.cache_head * params.head_dim;

    float accumulator = 0.0f;
    for (uint dim = 0; dim < params.head_dim; ++dim) {
        accumulator += query[query_offset + dim] * key_cache[key_offset + dim];
    }

    const uint absolute_query_position = params.query_position + query_row;
    output[index] = key_row > absolute_query_position
        ? -INFINITY
        : accumulator * params.scale;
}

// Compute attention-value products directly against one KV value head in the
// fixed cache. Scores are supplied after softmax and have shape
// [query_rows, key_length]; output has shape [query_rows, head_dim].
kernel void metal_kv_cache_av_f32(
        device const float * scores [[buffer(0)]],
        device const float * value_cache [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalKVCacheAVParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.query_rows * params.head_dim;
    if (index >= total) {
        return;
    }

    const uint query_row = index / params.head_dim;
    const uint dim = index - query_row * params.head_dim;
    float accumulator = 0.0f;
    for (uint key_row = 0; key_row < params.key_length; ++key_row) {
        const float probability =
            scores[query_row * params.score_stride + key_row];
        const uint value_offset = key_row * params.cache_stride +
                                   params.cache_head * params.head_dim + dim;
        accumulator += probability * value_cache[value_offset];
    }
    const uint output_index = query_row * params.output_stride +
                              params.output_offset + dim;
    output[output_index] = accumulator;
}

// Embedding is a gather expressed as a flat copy. One thread owns one output
// scalar; all embedding columns belonging to one token read the same token id
// and contiguous values from that token's [vocabulary, hidden] weight row.
kernel void metal_embedding_f32(
        device const int * token_ids [[buffer(0)]],
        device const float * embedding [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalEmbeddingParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.sequence_length * params.embedding_size;
    if (index >= total) {
        return;
    }

    const uint position = index / params.embedding_size;
    const uint col = index - position * params.embedding_size;
    const int token = token_ids[position];
    if (token < 0 || uint(token) >= params.vocabulary_size) {
        output[index] = 0.0f;
        return;
    }
    output[index] = embedding[
        uint(token) * params.embedding_size + col];
}

// Greedy selection is kept on the device so a generation step returns only a
// four-byte token id instead of the complete vocabulary logits. This baseline
// implementation intentionally launches one thread, which scans the whole
// vocabulary serially; a production implementation would use a parallel
// multi-stage reduction.
kernel void metal_argmax_f32(
        device const float * input [[buffer(0)]],
        device uint * output [[buffer(1)]],
        constant MetalArgmaxParams & params [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    if (index != 0) {
        return;
    }
    if (params.length == 0) {
        output[0] = 0;
        return;
    }

    uint best_index = 0;
    float best_value = input[0];
    for (uint candidate = 1; candidate < params.length; ++candidate) {
        const float value = input[candidate];
        if (value > best_value) {
            best_value = value;
            best_index = candidate;
        }
    }
    output[0] = best_index;
}

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

struct MetalQuantizedProductParams {
    uint m;
    uint n;
    uint k;
    uint activation_stride;
    uint weight_row_bytes;
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
    uint weight_row_bytes;
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

struct MetalSplitQGateParams {
    uint rows;
    uint head_count;
    uint head_dim;
};

struct MetalBroadcastParams {
    uint rows;
    uint cols;
};

struct MetalRowScaleParams {
    uint rows;
    uint cols;
    uint weight_stride;
    uint weight_column;
};

struct MetalHeadNormParams {
    uint tokens;
    uint head_count;
    uint head_dim;
    uint input_token_stride;
    float epsilon;
};

struct MetalDepthwiseConvParams {
    uint tokens;
    uint channels;
    uint kernel_size;
};

struct MetalGdnParams {
    uint tokens;
    uint key_head_count;
    uint value_head_count;
    uint head_dim;
    uint q_stride;
    uint k_stride;
    uint v_stride;
    uint output_stride;
};

struct MetalTopKParams {
    uint rows;
    uint cols;
    uint k;
};

struct MetalExpertProductParams {
    uint rows;
    uint output_size;
    uint input_size;
    uint activation_stride;
    uint weight_row_bytes;
    uint expert_stride_bytes;
    uint route_stride;
    uint route_index;
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

// GGUF quantized blocks are byte-packed and may not satisfy native struct
// alignment. Read all FP16 values and bit fields explicitly in little-endian
// order so the kernels exactly match the on-disk GGML layouts.
inline ushort load_u16_le(device const uchar * data) {
    return ushort(data[0]) | (ushort(data[1]) << 8);
}

inline uint load_u32_le(device const uchar * data) {
    return uint(data[0]) |
           (uint(data[1]) << 8) |
           (uint(data[2]) << 16) |
           (uint(data[3]) << 24);
}

inline float load_f16_le(device const uchar * data) {
    return float(as_type<half>(load_u16_le(data)));
}

inline int load_i8_value(device const uchar * data) {
    const int value = int(data[0]);
    return value < 128 ? value : value - 256;
}

inline uint2 q4_k_scale_min(device const uchar * scales, uint index) {
    uint scale = 0;
    uint minimum = 0;
    if (index < 4) {
        scale = uint(scales[index] & 63);
        minimum = uint(scales[index + 4] & 63);
    } else {
        scale = uint(scales[index + 4] & 0x0f) |
                (uint(scales[index - 4] >> 6) << 4);
        minimum = uint(scales[index + 4] >> 4) |
                  (uint(scales[index] >> 6) << 4);
    }
    return uint2(scale, minimum);
}

inline float dot_q5_0_f32(device const uchar * row,
                          device const float * activation,
                          uint columns) {
    float accumulator = 0.0f;
    for (uint block_index = 0; block_index < columns / 32;
         ++block_index) {
        device const uchar * block = row + block_index * 22;
        const float d = load_f16_le(block);
        const uint high_bits = load_u32_le(block + 2);
        device const uchar * quants = block + 6;
        const uint activation_offset = block_index * 32;
        for (uint local = 0; local < 16; ++local) {
            const int low_value =
                int(quants[local] & 0x0f) |
                int(((high_bits >> local) & 1) << 4);
            const int high_value =
                int(quants[local] >> 4) |
                int(((high_bits >> (local + 16)) & 1) << 4);
            accumulator += activation[activation_offset + local] *
                           (float(low_value - 16) * d);
            accumulator += activation[activation_offset + local + 16] *
                           (float(high_value - 16) * d);
        }
    }
    return accumulator;
}

inline float dot_q8_0_f32(device const uchar * row,
                          device const float * activation,
                          uint columns) {
    float accumulator = 0.0f;
    for (uint block_index = 0; block_index < columns / 32;
         ++block_index) {
        device const uchar * block = row + block_index * 34;
        const float d = load_f16_le(block);
        const uint activation_offset = block_index * 32;
        for (uint local = 0; local < 32; ++local) {
            accumulator += activation[activation_offset + local] *
                           (float(load_i8_value(block + 2 + local)) * d);
        }
    }
    return accumulator;
}

inline float dot_q4_k_f32(device const uchar * row,
                          device const float * activation,
                          uint columns) {
    float accumulator = 0.0f;
    for (uint block_index = 0; block_index < columns / 256;
         ++block_index) {
        device const uchar * block = row + block_index * 144;
        const float d = load_f16_le(block);
        const float dmin = load_f16_le(block + 2);
        device const uchar * scales = block + 4;
        device const uchar * quants = block + 16;
        const uint activation_offset = block_index * 256;
        for (uint group = 0; group < 4; ++group) {
            device const uchar * packed = quants + group * 32;
            for (uint half_index = 0; half_index < 2; ++half_index) {
                const uint2 scale_min = q4_k_scale_min(
                    scales, group * 2 + half_index);
                const uint group_offset = group * 64 + half_index * 32;
                float quant_dot = 0.0f;
                float activation_sum = 0.0f;
                for (uint local = 0; local < 32; ++local) {
                    const uint quant = half_index == 0
                        ? uint(packed[local] & 0x0f)
                        : uint(packed[local] >> 4);
                    const float activation_value =
                        activation[activation_offset + group_offset + local];
                    quant_dot += activation_value * float(quant);
                    activation_sum += activation_value;
                }
                accumulator += d * float(scale_min.x) * quant_dot -
                               dmin * float(scale_min.y) * activation_sum;
            }
        }
    }
    return accumulator;
}

inline float dot_q4_k_f32_packed_reuse(device const uchar * row,
                                       device const float * activation,
                                       uint columns) {
    // Each packed byte supplies a low- and high-nibble weight for adjacent
    // 32-element activation halves. Consume both while the byte is live, then
    // apply each subgroup's scale and minimum once after its local reduction.
    float accumulator = 0.0f;
    for (uint block_index = 0; block_index < columns / 256;
         ++block_index) {
        device const uchar * block = row + block_index * 144;
        const float d = load_f16_le(block);
        const float dmin = load_f16_le(block + 2);
        device const uchar * scales = block + 4;
        device const uchar * quants = block + 16;
        const uint activation_offset = block_index * 256;
        for (uint group = 0; group < 4; ++group) {
            device const uchar * packed = quants + group * 32;
            const uint2 low_scale_min = q4_k_scale_min(scales, group * 2);
            const uint2 high_scale_min = q4_k_scale_min(
                scales, group * 2 + 1);
            const uint group_offset = activation_offset + group * 64;
            float low_quant_dot = 0.0f;
            float high_quant_dot = 0.0f;
            float low_activation_sum = 0.0f;
            float high_activation_sum = 0.0f;
            for (uint local = 0; local < 32; ++local) {
                const uint packed_value = uint(packed[local]);
                const uint low_quant = packed_value & 0x0f;
                const uint high_quant = packed_value >> 4;
                const float low_activation =
                    activation[group_offset + local];
                const float high_activation =
                    activation[group_offset + local + 32];
                low_quant_dot += low_activation * float(low_quant);
                high_quant_dot += high_activation * float(high_quant);
                low_activation_sum += low_activation;
                high_activation_sum += high_activation;
            }
            accumulator += d * float(low_scale_min.x) * low_quant_dot -
                           dmin * float(low_scale_min.y) * low_activation_sum;
            accumulator += d * float(high_scale_min.x) * high_quant_dot -
                           dmin * float(high_scale_min.y) * high_activation_sum;
        }
    }
    return accumulator;
}

inline float dot_q5_k_f32(device const uchar * row,
                          device const float * activation,
                          uint columns) {
    float accumulator = 0.0f;
    for (uint block_index = 0; block_index < columns / 256;
         ++block_index) {
        device const uchar * block = row + block_index * 176;
        const float d = load_f16_le(block);
        const float dmin = load_f16_le(block + 2);
        device const uchar * scales = block + 4;
        device const uchar * high_bits = block + 16;
        device const uchar * quants = block + 48;
        const uint activation_offset = block_index * 256;

        for (uint group = 0; group < 4; ++group) {
            const uint2 low_scale_min = q4_k_scale_min(scales, group * 2);
            const uint2 high_scale_min = q4_k_scale_min(
                scales, group * 2 + 1);
            const uint low_high_mask = 1u << (group * 2);
            const uint high_high_mask = 2u << (group * 2);
            device const uchar * packed = quants + group * 32;
            const uint group_offset = activation_offset + group * 64;

            float low_quant_dot = 0.0f;
            float high_quant_dot = 0.0f;
            float low_activation_sum = 0.0f;
            float high_activation_sum = 0.0f;
            for (uint local = 0; local < 32; ++local) {
                const uint packed_value = uint(packed[local]);
                const uint low_quant = (packed_value & 0x0f) +
                    ((uint(high_bits[local]) & low_high_mask) != 0 ? 16 : 0);
                const uint high_quant = (packed_value >> 4) +
                    ((uint(high_bits[local]) & high_high_mask) != 0 ? 16 : 0);
                const float low_activation =
                    activation[group_offset + local];
                const float high_activation =
                    activation[group_offset + local + 32];
                low_quant_dot += low_activation * float(low_quant);
                high_quant_dot += high_activation * float(high_quant);
                low_activation_sum += low_activation;
                high_activation_sum += high_activation;
            }
            accumulator += d * float(low_scale_min.x) * low_quant_dot -
                           dmin * float(low_scale_min.y) * low_activation_sum;
            accumulator += d * float(high_scale_min.x) * high_quant_dot -
                           dmin * float(high_scale_min.y) * high_activation_sum;
        }
    }
    return accumulator;
}

inline float dot_q6_k_f32(device const uchar * row,
                          device const float * activation,
                          uint columns) {
    float accumulator = 0.0f;
    for (uint block_index = 0; block_index < columns / 256;
         ++block_index) {
        device const uchar * block = row + block_index * 210;
        device const uchar * lower = block;
        device const uchar * upper = block + 128;
        device const uchar * scales = block + 192;
        const float d = load_f16_le(block + 208);
        const uint activation_offset = block_index * 256;
        for (uint half_index = 0; half_index < 2; ++half_index) {
            device const uchar * lower_half = lower + half_index * 64;
            device const uchar * upper_half = upper + half_index * 32;
            device const uchar * scale_half = scales + half_index * 8;
            const uint half_offset = activation_offset + half_index * 128;
            for (uint scale_index = 0; scale_index < 2; ++scale_index) {
                float quant_dot0 = 0.0f;
                float quant_dot1 = 0.0f;
                float quant_dot2 = 0.0f;
                float quant_dot3 = 0.0f;
                const uint local_offset = scale_index * 16;
                for (uint lane = 0; lane < 16; ++lane) {
                    const uint local = local_offset + lane;
                    const uint upper_value = uint(upper_half[local]);
                    const int q0 = int((lower_half[local] & 0x0f) |
                        ((upper_value & 0x03) << 4)) - 32;
                    const int q1 = int((lower_half[local + 32] & 0x0f) |
                        (((upper_value >> 2) & 0x03) << 4)) - 32;
                    const int q2 = int((lower_half[local] >> 4) |
                        (((upper_value >> 4) & 0x03) << 4)) - 32;
                    const int q3 = int((lower_half[local + 32] >> 4) |
                        (((upper_value >> 6) & 0x03) << 4)) - 32;
                    quant_dot0 += activation[half_offset + local] * float(q0);
                    quant_dot1 += activation[half_offset + local + 32] *
                                  float(q1);
                    quant_dot2 += activation[half_offset + local + 64] *
                                  float(q2);
                    quant_dot3 += activation[half_offset + local + 96] *
                                  float(q3);
                }
                accumulator += d * (
                    float(load_i8_value(scale_half + scale_index)) *
                        quant_dot0 +
                    float(load_i8_value(scale_half + scale_index + 2)) *
                        quant_dot1 +
                    float(load_i8_value(scale_half + scale_index + 4)) *
                        quant_dot2 +
                    float(load_i8_value(scale_half + scale_index + 6)) *
                        quant_dot3);
            }
        }
    }
    return accumulator;
}

inline float q5_0_value(device const uchar * row, uint column) {
    device const uchar * block = row + (column / 32) * 22;
    const uint local = column % 32;
    const uint packed_index = local % 16;
    const uint low = local < 16
        ? uint(block[6 + packed_index] & 0x0f)
        : uint(block[6 + packed_index] >> 4);
    const uint high = ((load_u32_le(block + 2) >> local) & 1) << 4;
    return float(int(low | high) - 16) * load_f16_le(block);
}

inline float q8_0_value(device const uchar * row, uint column) {
    device const uchar * block = row + (column / 32) * 34;
    return float(load_i8_value(block + 2 + (column % 32))) *
           load_f16_le(block);
}

inline float q4_k_value(device const uchar * row, uint column) {
    device const uchar * block = row + (column / 256) * 144;
    const uint local = column % 256;
    const uint group = local / 64;
    const uint half_index = (local % 64) / 32;
    const uint packed_index = local % 32;
    const uint2 scale_min = q4_k_scale_min(
        block + 4, group * 2 + half_index);
    device const uchar * packed = block + 16 + group * 32;
    const uint quant = half_index == 0
        ? uint(packed[packed_index] & 0x0f)
        : uint(packed[packed_index] >> 4);
    return load_f16_le(block) * float(scale_min.x) * float(quant) -
           load_f16_le(block + 2) * float(scale_min.y);
}

inline float q6_k_value(device const uchar * row, uint column) {
    device const uchar * block = row + (column / 256) * 210;
    const uint local = column % 256;
    const uint half_index = local / 128;
    const uint quarter = (local % 128) / 32;
    const uint packed_index = local % 32;
    device const uchar * lower = block + half_index * 64;
    device const uchar * upper = block + 128 + half_index * 32;
    const uint upper_value = uint(upper[packed_index]);
    uint quant = 0;
    if (quarter == 0) {
        quant = uint(lower[packed_index] & 0x0f) |
                ((upper_value & 0x03) << 4);
    } else if (quarter == 1) {
        quant = uint(lower[packed_index + 32] & 0x0f) |
                (((upper_value >> 2) & 0x03) << 4);
    } else if (quarter == 2) {
        quant = uint(lower[packed_index] >> 4) |
                (((upper_value >> 4) & 0x03) << 4);
    } else {
        quant = uint(lower[packed_index + 32] >> 4) |
                (((upper_value >> 6) & 0x03) << 4);
    }
    const uint scale_index = half_index * 8 + quarter * 2 +
                             packed_index / 16;
    return load_f16_le(block + 208) *
           float(load_i8_value(block + 192 + scale_index)) *
           float(int(quant) - 32);
}

// Prefill products use one thread per [token, output-channel] pair. The FP32
// activation row is reused by neighboring output threads through GPU caches;
// each thread streams and decodes one independent GGUF weight row.
kernel void metal_gemm_q5_0_f32(
        device const float * activation [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint2 position [[thread_position_in_grid]]) {
    const uint col = position.x;
    const uint row = position.y;
    if (row >= params.m || col >= params.n) return;
    float value = dot_q5_0_f32(
        weight + col * params.weight_row_bytes,
        activation + row * params.activation_stride, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[col];
    output[row * params.n + col] = value;
}

kernel void metal_gemm_q8_0_f32(
        device const float * activation [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint2 position [[thread_position_in_grid]]) {
    const uint col = position.x;
    const uint row = position.y;
    if (row >= params.m || col >= params.n) return;
    float value = dot_q8_0_f32(
        weight + col * params.weight_row_bytes,
        activation + row * params.activation_stride, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[col];
    output[row * params.n + col] = value;
}

kernel void metal_gemm_q4_k_f32(
        device const float * activation [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint2 position [[thread_position_in_grid]]) {
    const uint col = position.x;
    const uint row = position.y;
    if (row >= params.m || col >= params.n) return;
    float value = dot_q4_k_f32(
        weight + col * params.weight_row_bytes,
        activation + row * params.activation_stride, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[col];
    output[row * params.n + col] = value;
}

kernel void metal_gemm_q5_k_f32(
        device const float * activation [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint2 position [[thread_position_in_grid]]) {
    const uint col = position.x;
    const uint row = position.y;
    if (row >= params.m || col >= params.n) return;
    float value = dot_q5_k_f32(
        weight + col * params.weight_row_bytes,
        activation + row * params.activation_stride, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[col];
    output[row * params.n + col] = value;
}

kernel void metal_gemm_q6_k_f32(
        device const float * activation [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint2 position [[thread_position_in_grid]]) {
    const uint col = position.x;
    const uint row = position.y;
    if (row >= params.m || col >= params.n) return;
    float value = dot_q6_k_f32(
        weight + col * params.weight_row_bytes,
        activation + row * params.activation_stride, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[col];
    output[row * params.n + col] = value;
}

// Decode products use one thread per output channel. Each thread decodes one
// complete quantized weight row and accumulates its dot product in FP32.
kernel void metal_gevm_q5_0_f32(
        device const uchar * weight [[buffer(0)]],
        device const float * activation [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint output_index [[thread_position_in_grid]]) {
    if (output_index >= params.n) return;
    float value = dot_q5_0_f32(
        weight + output_index * params.weight_row_bytes,
        activation, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[output_index];
    output[output_index] = value;
}

kernel void metal_gevm_q8_0_f32(
        device const uchar * weight [[buffer(0)]],
        device const float * activation [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint output_index [[thread_position_in_grid]]) {
    if (output_index >= params.n) return;
    float value = dot_q8_0_f32(
        weight + output_index * params.weight_row_bytes,
        activation, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[output_index];
    output[output_index] = value;
}

kernel void metal_gevm_q4_k_f32(
        device const uchar * weight [[buffer(0)]],
        device const float * activation [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint output_index [[thread_position_in_grid]]) {
    if (output_index >= params.n) return;
    float value = dot_q4_k_f32_packed_reuse(
        weight + output_index * params.weight_row_bytes,
        activation, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[output_index];
    output[output_index] = value;
}

kernel void metal_gevm_q5_k_f32(
        device const uchar * weight [[buffer(0)]],
        device const float * activation [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint output_index [[thread_position_in_grid]]) {
    if (output_index >= params.n) return;
    float value = dot_q5_k_f32(
        weight + output_index * params.weight_row_bytes,
        activation, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[output_index];
    output[output_index] = value;
}

kernel void metal_gevm_q6_k_f32(
        device const uchar * weight [[buffer(0)]],
        device const float * activation [[buffer(1)]],
        device const float * bias [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalQuantizedProductParams & params [[buffer(4)]],
        uint output_index [[thread_position_in_grid]]) {
    if (output_index >= params.n) return;
    float value = dot_q6_k_f32(
        weight + output_index * params.weight_row_bytes,
        activation, params.k) * params.scale;
    if (params.has_bias != 0) value += bias[output_index];
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

kernel void metal_embedding_q5_0_f32(
        device const int * token_ids [[buffer(0)]],
        device const uchar * embedding [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalEmbeddingParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.sequence_length * params.embedding_size;
    if (index >= total) return;
    const uint position = index / params.embedding_size;
    const uint col = index - position * params.embedding_size;
    const int token = token_ids[position];
    if (token < 0 || uint(token) >= params.vocabulary_size) {
        output[index] = 0.0f;
        return;
    }
    output[index] = q5_0_value(
        embedding + uint(token) * params.weight_row_bytes, col);
}

kernel void metal_embedding_q8_0_f32(
        device const int * token_ids [[buffer(0)]],
        device const uchar * embedding [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalEmbeddingParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.sequence_length * params.embedding_size;
    if (index >= total) return;
    const uint position = index / params.embedding_size;
    const uint col = index - position * params.embedding_size;
    const int token = token_ids[position];
    if (token < 0 || uint(token) >= params.vocabulary_size) {
        output[index] = 0.0f;
        return;
    }
    output[index] = q8_0_value(
        embedding + uint(token) * params.weight_row_bytes, col);
}

kernel void metal_embedding_q4_k_f32(
        device const int * token_ids [[buffer(0)]],
        device const uchar * embedding [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalEmbeddingParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.sequence_length * params.embedding_size;
    if (index >= total) return;
    const uint position = index / params.embedding_size;
    const uint col = index - position * params.embedding_size;
    const int token = token_ids[position];
    if (token < 0 || uint(token) >= params.vocabulary_size) {
        output[index] = 0.0f;
        return;
    }
    output[index] = q4_k_value(
        embedding + uint(token) * params.weight_row_bytes, col);
}

kernel void metal_embedding_q6_k_f32(
        device const int * token_ids [[buffer(0)]],
        device const uchar * embedding [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalEmbeddingParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.sequence_length * params.embedding_size;
    if (index >= total) return;
    const uint position = index / params.embedding_size;
    const uint col = index - position * params.embedding_size;
    const int token = token_ids[position];
    if (token < 0 || uint(token) >= params.vocabulary_size) {
        output[index] = 0.0f;
        return;
    }
    output[index] = q6_k_value(
        embedding + uint(token) * params.weight_row_bytes, col);
}

// Qwen3.5/3.6 packs each attention head as [query, output_gate]. Materialize
// two conventional row-major matrices so the following norm, RoPE and gate
// kernels can remain ordinary standalone operators.
kernel void metal_split_interleaved_q_gate_f32(
        device const float * input [[buffer(0)]],
        device float * query [[buffer(1)]],
        device float * gate [[buffer(2)]],
        constant MetalSplitQGateParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint output_cols = params.head_count * params.head_dim;
    const uint total = params.rows * output_cols;
    if (index >= total || params.head_dim == 0) return;
    const uint row = index / output_cols;
    const uint col = index - row * output_cols;
    const uint head = col / params.head_dim;
    const uint local = col - head * params.head_dim;
    const uint input_cols = output_cols * 2;
    const uint source = row * input_cols + head * params.head_dim * 2 + local;
    query[index] = input[source];
    gate[index] = input[source + params.head_dim];
}

kernel void metal_sigmoid_f32(
        device const float * input [[buffer(0)]],
        device float * output [[buffer(1)]],
        constant MetalElementwiseParams & params [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index >= total) return;
    const float value = input[index];
    output[index] = value >= 0.0f
        ? 1.0f / (1.0f + exp(-value))
        : exp(value) / (1.0f + exp(value));
}

kernel void metal_silu_only_f32(
        device const float * input [[buffer(0)]],
        device float * output [[buffer(1)]],
        constant MetalElementwiseParams & params [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index >= total) return;
    const float value = input[index];
    const float sigmoid = value >= 0.0f
        ? 1.0f / (1.0f + exp(-value))
        : exp(value) / (1.0f + exp(value));
    output[index] = value * sigmoid;
}

kernel void metal_softplus_f32(
        device const float * input [[buffer(0)]],
        device float * output [[buffer(1)]],
        constant MetalElementwiseParams & params [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index >= total) return;
    const float value = input[index];
    output[index] = value > 20.0f
        ? value
        : (value < -20.0f ? exp(value) : log(1.0f + exp(value)));
}

kernel void metal_exp_f32(
        device const float * input [[buffer(0)]],
        device float * output [[buffer(1)]],
        constant MetalElementwiseParams & params [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index < total) output[index] = exp(input[index]);
}

kernel void metal_mul_f32(
        device const float * left [[buffer(0)]],
        device const float * right [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalElementwiseParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index < total) output[index] = left[index] * right[index];
}

kernel void metal_add_channel_bias_f32(
        device const float * input [[buffer(0)]],
        device const float * channel [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalBroadcastParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index < total) output[index] = input[index] + channel[index % params.cols];
}

kernel void metal_mul_channel_f32(
        device const float * input [[buffer(0)]],
        device const float * channel [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalBroadcastParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index < total) output[index] = input[index] * channel[index % params.cols];
}

kernel void metal_row_scale_selected_f32(
        device const float * input [[buffer(0)]],
        device const float * row_weights [[buffer(1)]],
        device float * output [[buffer(2)]],
        constant MetalRowScaleParams & params [[buffer(3)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.cols;
    if (index >= total) return;
    const uint row = index / params.cols;
    output[index] = input[index] *
        row_weights[row * params.weight_stride + params.weight_column];
}

// One threadgroup normalizes one [token, head] vector. input_token_stride
// permits Q/K to remain views into the wider convolved QKV matrix.
kernel void metal_l2norm_heads_f32(
        device const float * input [[buffer(0)]],
        device float * output [[buffer(1)]],
        constant MetalHeadNormParams & params [[buffer(2)]],
        uint local_index [[thread_index_in_threadgroup]],
        uint3 group_id [[threadgroup_position_in_grid]]) {
    const uint vector_index = group_id.x;
    const uint vector_count = params.tokens * params.head_count;
    if (vector_index >= vector_count) return;
    const uint token = vector_index / params.head_count;
    const uint head = vector_index - token * params.head_count;
    const uint source = token * params.input_token_stride +
                        head * params.head_dim;
    const uint destination = vector_index * params.head_dim;

    threadgroup float partial[MYLLM_ELEMENTWISE_THREADS];
    float local_sum = 0.0f;
    for (uint dim = local_index; dim < params.head_dim;
         dim += MYLLM_ELEMENTWISE_THREADS) {
        const float value = input[source + dim];
        local_sum += value * value;
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
    const float inverse_norm = rsqrt(partial[0] + params.epsilon);
    for (uint dim = local_index; dim < params.head_dim;
         dim += MYLLM_ELEMENTWISE_THREADS) {
        output[destination + dim] = input[source + dim] * inverse_norm;
    }
}

// The logical convolution input is [old history, projected QKV]. Each channel
// has its own short kernel and therefore maps cleanly to one output thread.
kernel void metal_depthwise_conv1d_causal_f32(
        device const float * input [[buffer(0)]],
        device const float * history [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalDepthwiseConvParams & params [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    const uint total = params.tokens * params.channels;
    if (index >= total || params.kernel_size == 0) return;
    const uint token = index / params.channels;
    const uint channel = index - token * params.channels;
    const uint history_rows = params.kernel_size - 1;
    float accumulator = 0.0f;
    for (uint tap = 0; tap < params.kernel_size; ++tap) {
        const uint concat_row = token + tap;
        const float value = concat_row < history_rows
            ? history[concat_row * params.channels + channel]
            : input[(concat_row - history_rows) * params.channels + channel];
        accumulator += value * weight[channel * params.kernel_size + tap];
    }
    output[index] = accumulator;
}

// One thread owns a channel so all old history values are captured before the
// same buffer is overwritten. This makes the update race-free for decode.
kernel void metal_conv_history_commit_f32(
        device const float * input [[buffer(0)]],
        device float * history [[buffer(1)]],
        constant MetalDepthwiseConvParams & params [[buffer(2)]],
        uint channel [[thread_position_in_grid]]) {
    if (channel >= params.channels || params.kernel_size < 2 ||
        params.kernel_size > 9) return;
    const uint history_rows = params.kernel_size - 1;
    float old_values[8];
    for (uint row = 0; row < history_rows; ++row) {
        old_values[row] = history[row * params.channels + channel];
    }
    for (uint row = 0; row < history_rows; ++row) {
        const uint concat_row = params.tokens + row;
        history[row * params.channels + channel] =
            concat_row < history_rows
                ? old_values[concat_row]
                : input[(concat_row - history_rows) * params.channels +
                        channel];
    }
}

// Stateful DeltaNet scan. A thread owns one state column for one value head,
// while token order remains serial. Qwen3.5/3.6 interleaves the repeated Q/K
// heads, so value head h reads Q/K head (h % key_head_count).
kernel void metal_gdn_recurrence_f32(
        device const float * query [[buffer(0)]],
        device const float * key [[buffer(1)]],
        device const float * value [[buffer(2)]],
        device const float * decay [[buffer(3)]],
        device const float * beta [[buffer(4)]],
        device float * state [[buffer(5)]],
        device float * output [[buffer(6)]],
        constant MetalGdnParams & params [[buffer(7)]],
        uint local_index [[thread_index_in_threadgroup]],
        uint3 group_id [[threadgroup_position_in_grid]]) {
    const uint value_head = group_id.x;
    const uint column = local_index;
    if (value_head >= params.value_head_count ||
        column >= params.head_dim || params.key_head_count == 0) return;
    const uint key_head = value_head % params.key_head_count;
    const uint state_head = value_head * params.head_dim * params.head_dim;
    const float query_scale = rsqrt(float(params.head_dim));

    for (uint token = 0; token < params.tokens; ++token) {
        const uint q_base = token * params.q_stride +
                            key_head * params.head_dim;
        const uint k_base = token * params.k_stride +
                            key_head * params.head_dim;
        const uint v_base = token * params.v_stride +
                            value_head * params.head_dim;
        const float decay_value =
            decay[token * params.value_head_count + value_head];

        float prediction = 0.0f;
        for (uint row = 0; row < params.head_dim; ++row) {
            const uint state_index = state_head + row * params.head_dim + column;
            const float decayed_state = state[state_index] * decay_value;
            state[state_index] = decayed_state;
            prediction += decayed_state * key[k_base + row];
        }

        const float delta = beta[token * params.value_head_count + value_head] *
            (value[v_base + column] - prediction);
        float result = 0.0f;
        for (uint row = 0; row < params.head_dim; ++row) {
            const uint state_index = state_head + row * params.head_dim + column;
            const float updated = state[state_index] +
                key[k_base + row] * delta;
            state[state_index] = updated;
            result += updated * query[q_base + row] * query_scale;
        }
        output[token * params.output_stride +
               value_head * params.head_dim + column] = result;
    }
}

// Router softmax is computed separately. This kernel performs a deterministic
// top-k insertion scan for every row and keeps lower expert ids on exact ties.
kernel void metal_topk_f32(
        device const float * input [[buffer(0)]],
        device uint * indices [[buffer(1)]],
        device float * values [[buffer(2)]],
        constant MetalTopKParams & params [[buffer(3)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= params.rows || params.k == 0 || params.k > 8) return;
    float best_values[8];
    uint best_indices[8];
    for (uint rank = 0; rank < 8; ++rank) {
        best_values[rank] = -INFINITY;
        best_indices[rank] = 0xffffffffu;
    }
    for (uint col = 0; col < params.cols; ++col) {
        const float candidate = input[row * params.cols + col];
        uint insertion = params.k;
        for (uint rank = 0; rank < params.k; ++rank) {
            if (candidate > best_values[rank]) {
                insertion = rank;
                break;
            }
        }
        if (insertion < params.k) {
            for (uint rank = params.k - 1; rank > insertion; --rank) {
                best_values[rank] = best_values[rank - 1];
                best_indices[rank] = best_indices[rank - 1];
            }
            best_values[insertion] = candidate;
            best_indices[insertion] = col;
        }
    }
    for (uint rank = 0; rank < params.k; ++rank) {
        indices[row * params.k + rank] = best_indices[rank];
        values[row * params.k + rank] = best_values[rank];
    }
}

kernel void metal_topk_renorm_f32(
        device float * values [[buffer(0)]],
        constant MetalTopKParams & params [[buffer(1)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= params.rows || params.k == 0) return;
    float sum = 0.0f;
    for (uint rank = 0; rank < params.k; ++rank) {
        sum += values[row * params.k + rank];
    }
    const float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint rank = 0; rank < params.k; ++rank) {
        values[row * params.k + rank] *= inverse;
    }
}

kernel void metal_expert_gemm_q4_k_f32(
        device const float * activation [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const uint * expert_ids [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalExpertProductParams & params [[buffer(4)]],
        uint2 position [[thread_position_in_grid]]) {
    const uint col = position.x;
    const uint row = position.y;
    if (row >= params.rows || col >= params.output_size) return;
    const uint expert = expert_ids[
        row * params.route_stride + params.route_index];
    device const uchar * weight_row = weights +
        expert * params.expert_stride_bytes + col * params.weight_row_bytes;
    output[row * params.output_size + col] = dot_q4_k_f32_packed_reuse(
        weight_row, activation + row * params.activation_stride,
        params.input_size);
}

kernel void metal_expert_gemm_q6_k_f32(
        device const float * activation [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const uint * expert_ids [[buffer(2)]],
        device float * output [[buffer(3)]],
        constant MetalExpertProductParams & params [[buffer(4)]],
        uint2 position [[thread_position_in_grid]]) {
    const uint col = position.x;
    const uint row = position.y;
    if (row >= params.rows || col >= params.output_size) return;
    const uint expert = expert_ids[
        row * params.route_stride + params.route_index];
    device const uchar * weight_row = weights +
        expert * params.expert_stride_bytes + col * params.weight_row_bytes;
    output[row * params.output_size + col] = dot_q6_k_f32(
        weight_row, activation + row * params.activation_stride,
        params.input_size);
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

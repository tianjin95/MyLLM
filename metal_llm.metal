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

#define MYLLM_TILE_M 16
#define MYLLM_TILE_N 16
#define MYLLM_TILE_K 16

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

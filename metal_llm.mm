#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_llm.h"

#include "model.h"
#include "profiler.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mach-o/dyld.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llm {
namespace {

constexpr uint32_t kTileSize = 16;
constexpr uint32_t kDefaultGemvThreads = 256;

struct MetalMatmulParamsHost {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t lhs_stride = 0;
    uint32_t rhs_stride = 0;
    uint32_t lhs_transposed = 0;
    uint32_t rhs_transposed = 0;
    uint32_t has_bias = 0;
    float scale = 1.0f;
};

struct MetalGemvParamsHost {
    uint32_t output_size = 0;
    uint32_t input_size = 0;
    uint32_t matrix_stride = 0;
    uint32_t matrix_transposed = 0;
    uint32_t has_bias = 0;
    float scale = 1.0f;
};

struct MetalElementwiseParamsHost {
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct MetalRmsNormParamsHost {
    uint32_t rows = 0;
    uint32_t cols = 0;
    float epsilon = 0.0f;
};

struct MetalRopeHeadsParamsHost {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t head_dim = 0;
    uint32_t rotary_dimension = 0;
    uint32_t position = 0;
    float theta = 0.0f;
};

struct MetalEmbeddingParamsHost {
    uint32_t sequence_length = 0;
    uint32_t embedding_size = 0;
    uint32_t vocabulary_size = 0;
};

struct MetalKVCacheWriteParamsHost {
    uint32_t source_rows = 0;
    uint32_t source_cols = 0;
    uint32_t cache_stride = 0;
    uint32_t position = 0;
};

struct MetalKVCacheQKParamsHost {
    uint32_t query_rows = 0;
    uint32_t key_length = 0;
    uint32_t cache_stride = 0;
    uint32_t query_stride = 0;
    uint32_t head_dim = 0;
    uint32_t cache_head = 0;
    uint32_t query_position = 0;
    float scale = 1.0f;
};

struct MetalKVCacheAVParamsHost {
    uint32_t query_rows = 0;
    uint32_t key_length = 0;
    uint32_t cache_stride = 0;
    uint32_t score_stride = 0;
    uint32_t head_dim = 0;
    uint32_t cache_head = 0;
    uint32_t output_stride = 0;
    uint32_t output_offset = 0;
};

struct MetalArgmaxParamsHost {
    uint32_t length = 0;
};

static_assert(sizeof(MetalMatmulParamsHost) == 36,
              "Metal matmul parameter layout changed");
static_assert(sizeof(MetalGemvParamsHost) == 24,
              "Metal gemv parameter layout changed");
static_assert(sizeof(MetalElementwiseParamsHost) == 8,
              "Metal elementwise parameter layout changed");
static_assert(sizeof(MetalRmsNormParamsHost) == 12,
              "Metal RMSNorm parameter layout changed");
static_assert(sizeof(MetalRopeHeadsParamsHost) == 24,
              "Metal head-wise RoPE parameter layout changed");
static_assert(sizeof(MetalEmbeddingParamsHost) == 12,
              "Metal embedding parameter layout changed");
static_assert(sizeof(MetalKVCacheWriteParamsHost) == 16,
              "Metal KV-cache write parameter layout changed");
static_assert(sizeof(MetalKVCacheQKParamsHost) == 32,
              "Metal KV-cache QK parameter layout changed");
static_assert(sizeof(MetalKVCacheAVParamsHost) == 32,
              "Metal KV-cache AV parameter layout changed");
static_assert(sizeof(MetalArgmaxParamsHost) == 4,
              "Metal argmax parameter layout changed");

void validate_vector(const Vector& vector, const char* name) {
    if (vector.lens != vector.values.size()) {
        throw std::invalid_argument(std::string(name) +
                                    " length does not match its storage");
    }
}

void validate_matrix(const Matrix& matrix, const char* name) {
    if (matrix.rows != matrix.values.size()) {
        throw std::invalid_argument(std::string(name) +
                                    " row count does not match its storage");
    }
    for (const Vector& row : matrix.values) {
        validate_vector(row, name);
        if (row.lens != matrix.cols) {
            throw std::invalid_argument(std::string(name) +
                                        " column count does not match its rows");
        }
    }
}

size_t checked_elements(size_t rows, size_t cols, const char* name) {
    if (rows != 0 && cols > std::numeric_limits<size_t>::max() / rows) {
        throw std::length_error(std::string(name) + " element count overflows");
    }
    return rows * cols;
}

size_t checked_bytes_for(size_t elements,
                         size_t element_size,
                         const char* name) {
    if (element_size != 0 &&
        elements > std::numeric_limits<size_t>::max() / element_size) {
        throw std::length_error(std::string(name) + " byte count overflows");
    }
    return elements * element_size;
}

size_t checked_bytes(size_t elements, const char* name) {
    return checked_bytes_for(elements, sizeof(float), name);
}

uint32_t checked_uint(size_t value, const char* name) {
    if (value > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::length_error(std::string(name) + " does not fit in Metal uint");
    }
    return static_cast<uint32_t>(value);
}

std::vector<float> flatten_matrix(const Matrix& matrix, const char* name) {
    validate_matrix(matrix, name);
    const size_t elements = checked_elements(matrix.rows, matrix.cols, name);
    std::vector<float> result;
    result.reserve(elements);
    for (const Vector& row : matrix.values) {
        result.insert(result.end(), row.values.begin(), row.values.end());
    }
    return result;
}

std::vector<float> flatten_vector(const Vector& vector, const char* name) {
    validate_vector(vector, name);
    return vector.values;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    return std::string(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string locate_shader(const std::string& requested_path) {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    if (!requested_path.empty()) {
        candidates.emplace_back(requested_path);
    }

    if (const char* environment_path = std::getenv("MYLLM_METAL_SHADER");
        environment_path != nullptr && environment_path[0] != '\0') {
        candidates.emplace_back(environment_path);
    }
    candidates.emplace_back("metal_llm.metal");

    // Running ./chat from another directory should still find the shader next
    // to the executable.  The source-tree candidate below covers development
    // builds where the executable is elsewhere.
    char executable_path[PATH_MAX] = {};
    uint32_t executable_length = sizeof(executable_path);
    if (_NSGetExecutablePath(executable_path, &executable_length) == 0) {
        candidates.emplace_back(fs::path(executable_path).parent_path() /
                                "metal_llm.metal");
    }
    candidates.emplace_back(fs::path(__FILE__).parent_path() / "metal_llm.metal");

    for (const fs::path& candidate : candidates) {
        const std::string source = read_text_file(candidate);
        if (!source.empty()) {
            return source;
        }
    }

    throw std::runtime_error(
        "Cannot find metal_llm.metal; set MYLLM_METAL_SHADER to its path");
}

std::string ns_error_description(NSError* error) {
    if (error == nil) {
        return "unknown Metal error";
    }
    NSString* description = [error localizedDescription];
    if (description == nil || [description UTF8String] == nullptr) {
        return "unknown Metal error";
    }
    return std::string([description UTF8String]);
}

[[noreturn]] void throw_metal_error(const char* operation, NSError* error) {
    throw std::runtime_error(std::string(operation) + ": " +
                             ns_error_description(error));
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          const char* function_name,
                                          const char* operation) {
    NSString* name = [NSString stringWithUTF8String:function_name];
    if (name == nil) {
        throw std::runtime_error(std::string(operation) +
                                 ": invalid Metal function name");
    }
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        throw std::runtime_error(std::string(operation) +
                                 ": kernel function is missing");
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw_metal_error(operation, error);
    }
    return pipeline;
}

ProfileMetrics resident_prefill_metrics(const ModelConfig& model,
                                        size_t sequence_length) {
    ProfileMetrics result = profile_copy_metrics(
        static_cast<uint64_t>(sequence_length) * model.embedding_size *
            sizeof(float),
        1);
    for (size_t layer_index = 0;
         layer_index < model.layer_count;
         ++layer_index) {
        const size_t q_dimension =
            model.attention_head_count * model.head_size;
        const size_t kv_dimension = model.kv_head_count * model.head_size;
        const size_t feed_forward = model.feed_forward_size;
        const size_t q_heads = model.head_size == 0
            ? 0 : q_dimension / model.head_size;

        result += profile_elementwise_metrics(
            sequence_length * model.embedding_size, 5, 3, 1);
        result += profile_gemmb_metrics(
            sequence_length, q_dimension, model.embedding_size);
        result += profile_gemmb_metrics(
            sequence_length, kv_dimension, model.embedding_size);
        result += profile_gemmb_metrics(
            sequence_length, kv_dimension, model.embedding_size);
        result += profile_elementwise_metrics(
            sequence_length * (q_dimension + kv_dimension), 6, 1, 1);
        result += profile_copy_metrics(
            static_cast<uint64_t>(sequence_length) * kv_dimension *
                sizeof(float) * 2, 1);

        for (size_t head = 0; head < q_heads; ++head) {
            ProfileMetrics qk = profile_gemm_metrics(
                sequence_length, sequence_length, model.head_size);
            qk.flops += static_cast<uint64_t>(sequence_length) *
                        sequence_length;
            result += qk;
            result += profile_elementwise_metrics(
                sequence_length * sequence_length, 4, 3, 2);
            result += profile_gemm_metrics(
                sequence_length, model.head_size, sequence_length);
        }
        result += profile_gemm_metrics(
            sequence_length, model.embedding_size, q_dimension);
        result += profile_copy_metrics(
            static_cast<uint64_t>(sequence_length) * model.embedding_size *
                sizeof(float), 1);

        result += profile_elementwise_metrics(
            sequence_length * model.embedding_size, 5, 3, 1);
        result += profile_gemm_metrics(
            sequence_length, feed_forward, model.embedding_size);
        result += profile_gemm_metrics(
            sequence_length, feed_forward, model.embedding_size);
        result += profile_elementwise_metrics(
            sequence_length * feed_forward, 8, 2, 1);
        result += profile_gemm_metrics(
            sequence_length, model.embedding_size, feed_forward);
        result += profile_copy_metrics(
            static_cast<uint64_t>(sequence_length) * model.embedding_size *
                sizeof(float), 1);
    }
    result += profile_elementwise_metrics(
        sequence_length * model.embedding_size, 5, 2, 1);
    result += profile_gemv_metrics(
        model.embedding_size, model.vocabulary_size);
    result += profile_elementwise_metrics(
        model.vocabulary_size, 1, 1, 0);
    return result;
}

ProfileMetrics resident_decode_metrics(const ModelConfig& model,
                                       size_t key_length) {
    ProfileMetrics result = profile_copy_metrics(
        static_cast<uint64_t>(model.embedding_size) * sizeof(float), 1);
    for (size_t layer_index = 0;
         layer_index < model.layer_count;
         ++layer_index) {
        const size_t q_dimension =
            model.attention_head_count * model.head_size;
        const size_t kv_dimension = model.kv_head_count * model.head_size;
        const size_t feed_forward = model.feed_forward_size;
        const size_t q_heads = model.head_size == 0
            ? 0 : q_dimension / model.head_size;

        result += profile_elementwise_metrics(
            model.embedding_size, 5, 3, 1);
        result += profile_gemv_metrics(
            model.embedding_size, q_dimension);
        result += profile_gemv_metrics(
            model.embedding_size, kv_dimension);
        result += profile_gemv_metrics(
            model.embedding_size, kv_dimension);
        result += profile_elementwise_metrics(
            q_dimension + kv_dimension, 6, 1, 1);
        result += profile_copy_metrics(
            static_cast<uint64_t>(kv_dimension) * sizeof(float) * 2, 1);
        for (size_t head = 0; head < q_heads; ++head) {
            result += profile_gemv_metrics(
                model.head_size, key_length);
            result += profile_elementwise_metrics(
                key_length, 4, 3, 2);
            result += profile_gemv_metrics(
                key_length, model.head_size);
        }
        result += profile_gemv_metrics(
            q_dimension, model.embedding_size);
        result += profile_copy_metrics(
            static_cast<uint64_t>(model.embedding_size) * sizeof(float), 1);
        result += profile_elementwise_metrics(
            model.embedding_size, 5, 3, 1);
        result += profile_gemv_metrics(
            model.embedding_size, feed_forward);
        result += profile_gemv_metrics(
            model.embedding_size, feed_forward);
        result += profile_elementwise_metrics(
            feed_forward, 8, 2, 1);
        result += profile_gemv_metrics(
            feed_forward, model.embedding_size);
        result += profile_copy_metrics(
            static_cast<uint64_t>(model.embedding_size) * sizeof(float), 1);
    }
    result += profile_elementwise_metrics(
        model.embedding_size, 5, 2, 1);
    result += profile_gemv_metrics(
        model.embedding_size, model.vocabulary_size);
    result += profile_elementwise_metrics(
        model.vocabulary_size, 1, 1, 0);
    return result;
}

} // namespace

struct MetalLLM::Impl {
    // Device-side tensor descriptors used by the asynchronous graph path.
    // All tensors are contiguous row-major FP32 allocations unless an
    // explicit element offset is supplied for a row/vector view.
    struct DeviceMatrix {
        id<MTLBuffer> buffer = nil;
        size_t rows = 0;
        size_t cols = 0;
        size_t stride = 0;
        size_t offset_elements = 0;
    };

    struct DeviceVector {
        id<MTLBuffer> buffer = nil;
        size_t length = 0;
        size_t offset_elements = 0;
    };

    // These descriptors own the model buffers for the whole lifetime of the
    // backend. They are built once during construction and are the only model
    // weights used by prefill/decode.
    struct PreparedLayer {
        DeviceVector attn_norm_weight;
        DeviceMatrix attn_q_weight;
        DeviceVector attn_q_bias;
        DeviceMatrix attn_k_weight;
        DeviceVector attn_k_bias;
        DeviceMatrix attn_v_weight;
        DeviceVector attn_v_bias;
        DeviceMatrix attn_output_weight;
        DeviceVector ffn_norm_weight;
        DeviceMatrix ffn_gate_weight;
        DeviceMatrix ffn_down_weight;
        DeviceMatrix ffn_up_weight;
    };

    struct PreparedModel {
        DeviceMatrix token_embedding_weight;
        DeviceVector output_norm_weight;
        DeviceMatrix output_weight;
        std::vector<PreparedLayer> layers;
        bool ready = false;
    };

    struct KVCacheLayer {
        id<MTLBuffer> key = nil;
        id<MTLBuffer> value = nil;
        size_t kv_dimension = 0;
    };

    struct AsyncExecution {
        id<MTLCommandBuffer> command = nil;
        struct TemporaryBuffer {
            id<MTLBuffer> buffer = nil;
            size_t capacity_bytes = 0;
            bool reusable = false;
        };

        // Retain every temporary buffer until the command has completed. The
        // caller returns them to Impl::temporary_pool after consuming any
        // CPU-visible result.
        std::vector<TemporaryBuffer> temporary_buffers;
    };

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> gemm_pipeline = nil;
    id<MTLComputePipelineState> gevm_pipeline = nil;
    id<MTLComputePipelineState> embedding_pipeline = nil;
    id<MTLComputePipelineState> rmsnorm_pipeline = nil;
    id<MTLComputePipelineState> softmax_pipeline = nil;
    id<MTLComputePipelineState> swiglu_pipeline = nil;
    id<MTLComputePipelineState> residual_pipeline = nil;
    id<MTLComputePipelineState> rope_heads_pipeline = nil;
    id<MTLComputePipelineState> kv_cache_write_pipeline = nil;
    id<MTLComputePipelineState> kv_cache_qk_pipeline = nil;
    id<MTLComputePipelineState> kv_cache_av_pipeline = nil;
    id<MTLComputePipelineState> argmax_pipeline = nil;
    id<MTLBuffer> dummy_buffer = nil;
    std::string device_name;
    ModelConfig config;
    PreparedModel prepared;
    std::vector<KVCacheLayer> kv_cache;
    size_t max_sequence = 0;
    size_t sequence_length = 0;
    std::unique_ptr<Profiler> profiler;
    mutable std::mutex temporary_pool_mutex;
    mutable std::unordered_map<size_t, std::vector<id<MTLBuffer>>>
        temporary_pool;
    mutable size_t temporary_pool_bytes = 0;

    static constexpr size_t kMaxReusableTemporaryBytes = 16 * 1024 * 1024;
    static constexpr size_t kMaxTemporaryPoolBytes = 256 * 1024 * 1024;

    void require_available() const {
        if (device == nil || gemm_pipeline == nil || gevm_pipeline == nil ||
            embedding_pipeline == nil || rmsnorm_pipeline == nil ||
            softmax_pipeline == nil || swiglu_pipeline == nil ||
            residual_pipeline == nil || rope_heads_pipeline == nil ||
            kv_cache_write_pipeline == nil || kv_cache_qk_pipeline == nil ||
            kv_cache_av_pipeline == nil || argmax_pipeline == nil) {
            const std::string reason = device_name.empty()
                ? "Metal backend is not initialized"
                : device_name;
            throw std::runtime_error("Metal backend is unavailable: " + reason);
        }
    }

    explicit Impl(const std::string& shader_path) {
        NSUInteger discovered_device_count = 0;
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                // Some macOS sessions expose the GPU through the all-devices
                // API even when the default-device helper is unavailable (for
                // example, a non-WindowServer or locked console session).
                NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
                discovered_device_count = [devices count];
                if ([devices count] != 0) {
                    device = [devices objectAtIndex:0];
                }
            }
            if (device == nil) {
                // A CPU-only or headless process is a valid deployment
                // environment for MyLLM. Keep the backend object in an
                // explicit unavailable state so callers can use
                // available() and select the CPU reference implementation.
                device_name =
                    "unavailable (no Metal device; discovered " +
                    std::to_string(static_cast<size_t>(discovered_device_count)) +
                    ")";
                return;
            }
            NSString* name = [device name];
            if (name != nil && [name UTF8String] != nullptr) {
                device_name = std::string([name UTF8String]);
            } else {
                device_name = "Metal device";
            }

            queue = [device newCommandQueue];
            if (queue == nil) {
                throw std::runtime_error("MTLDevice could not create a command queue");
            }
            dummy_buffer = [device newBufferWithLength:sizeof(float)
                                               options:MTLResourceStorageModeShared];
            if (dummy_buffer == nil) {
                throw std::runtime_error("MTLDevice could not create a dummy buffer");
            }
            *static_cast<float*>([dummy_buffer contents]) = 0.0f;

            const std::string source = locate_shader(shader_path);
            NSString* source_string = [[NSString alloc]
                initWithBytes:source.data()
                       length:source.size()
                     encoding:NSUTF8StringEncoding];
            if (source_string == nil) {
                throw std::runtime_error("metal_llm.metal is not valid UTF-8");
            }

            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            NSError* error = nil;
            id<MTLLibrary> library =
                [device newLibraryWithSource:source_string
                                     options:options
                                       error:&error];
            if (library == nil) {
                throw_metal_error("MTLDevice newLibraryWithSource", error);
            }

            gemm_pipeline = make_pipeline(
                device, library, "metal_gemm_f32", "create GEMM pipeline");
            gevm_pipeline = make_pipeline(
                device, library, "metal_gevm_f32", "create GEVM pipeline");
            embedding_pipeline = make_pipeline(
                device, library, "metal_embedding_f32",
                "create embedding pipeline");
            rmsnorm_pipeline = make_pipeline(
                device, library, "metal_rmsnorm_f32",
                "create RMSNorm pipeline");
            softmax_pipeline = make_pipeline(
                device, library, "metal_softmax_f32",
                "create softmax pipeline");
            swiglu_pipeline = make_pipeline(
                device, library, "metal_swiglu_f32",
                "create SwiGLU pipeline");
            residual_pipeline = make_pipeline(
                device, library, "metal_residual_f32",
                "create residual pipeline");
            rope_heads_pipeline = make_pipeline(
                device, library, "metal_rope_heads_f32",
                "create head-wise RoPE pipeline");
            kv_cache_write_pipeline = make_pipeline(
                device, library, "metal_kv_cache_write_f32",
                "create KV-cache write pipeline");
            kv_cache_qk_pipeline = make_pipeline(
                device, library, "metal_kv_cache_qk_f32",
                "create KV-cache QK pipeline");
            kv_cache_av_pipeline = make_pipeline(
                device, library, "metal_kv_cache_av_f32",
                "create KV-cache AV pipeline");
            argmax_pipeline = make_pipeline(
                device, library, "metal_argmax_f32",
                "create argmax pipeline");
            if ([gemm_pipeline maxTotalThreadsPerThreadgroup] <
                    static_cast<NSUInteger>(kTileSize * kTileSize)) {
                throw std::runtime_error(
                    "Metal device supports fewer than 256 GEMM threads per group");
            }
            if ([rmsnorm_pipeline maxTotalThreadsPerThreadgroup] <
                    static_cast<NSUInteger>(kDefaultGemvThreads) ||
                [softmax_pipeline maxTotalThreadsPerThreadgroup] <
                    static_cast<NSUInteger>(kDefaultGemvThreads)) {
                throw std::runtime_error(
                    "Metal device supports fewer than 256 reduction threads per group");
            }
        }
    }

    id<MTLBuffer> make_buffer(const void* data, size_t bytes) const {
        if (bytes == 0) {
            return nil;
        }
        id<MTLBuffer> buffer =
            [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw std::runtime_error("Metal failed to allocate a shared buffer");
        }
        if (data != nullptr) {
            std::memcpy([buffer contents], data, bytes);
        }
        return buffer;
    }

    DeviceMatrix upload_matrix(const Matrix& matrix, const char* name) const {
        const std::vector<float> flat = flatten_matrix(matrix, name);
        const size_t elements = checked_elements(
            matrix.rows, matrix.cols, name);
        id<MTLBuffer> buffer = make_buffer(
            flat.data(), checked_bytes(elements, name));
        return {buffer, matrix.rows, matrix.cols, matrix.cols, 0};
    }

    DeviceVector upload_vector(const Vector& vector, const char* name) const {
        const std::vector<float> flat = flatten_vector(vector, name);
        id<MTLBuffer> buffer = make_buffer(
            flat.data(), checked_bytes(flat.size(), name));
        return {buffer, vector.lens, 0};
    }

    void require_prepared() const {
        if (!prepared.ready || config.layer_count == 0 ||
            prepared.layers.size() != config.layer_count ||
            kv_cache.size() != config.layer_count || max_sequence == 0) {
            throw std::runtime_error(
                "Metal model weights or KV cache are not initialized");
        }
    }

    NSUInteger flat_threads(id<MTLComputePipelineState> pipeline) const {
        const NSUInteger maximum =
            [pipeline maxTotalThreadsPerThreadgroup];
        return std::min<NSUInteger>(
            kDefaultGemvThreads, std::max<NSUInteger>(1, maximum));
    }

    void dispatch_flat(id<MTLComputeCommandEncoder> encoder,
                       id<MTLComputePipelineState> pipeline,
                       size_t element_count) const {
        const NSUInteger threads = flat_threads(pipeline);
        const NSUInteger count = static_cast<NSUInteger>(element_count);
        const MTLSize threadgroups = MTLSizeMake(
            (count + threads - 1) / threads, 1, 1);
        [encoder dispatchThreadgroups:threadgroups
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }

    void dispatch_rows(id<MTLComputeCommandEncoder> encoder,
                       size_t row_count) const {
        [encoder dispatchThreadgroups:MTLSizeMake(
            static_cast<NSUInteger>(row_count), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(
                     kDefaultGemvThreads, 1, 1)];
    }

    AsyncExecution begin_async(const char* operation) const {
        AsyncExecution execution;
        execution.command = [queue commandBuffer];
        if (execution.command == nil) {
            throw std::runtime_error(std::string(
                "Metal failed to create ") + operation + " command buffer");
        }
        return execution;
    }

    size_t temporary_bucket_size(size_t bytes) const {
        constexpr size_t kMinimumBucket = 256;
        size_t bucket = kMinimumBucket;
        while (bucket < bytes) {
            if (bucket > std::numeric_limits<size_t>::max() / 2) {
                return bytes;
            }
            bucket *= 2;
        }
        return bucket;
    }

    id<MTLBuffer> acquire_temporary_buffer(AsyncExecution& execution,
                                           size_t bytes,
                                           const char* operation) const {
        if (bytes == 0) {
            throw std::invalid_argument(std::string(operation) +
                                        " buffer cannot be empty");
        }
        const size_t capacity = bytes <= kMaxReusableTemporaryBytes
            ? temporary_bucket_size(bytes)
            : bytes;
        const bool reusable = capacity <= kMaxReusableTemporaryBytes;
        id<MTLBuffer> buffer = nil;
        if (reusable) {
            std::lock_guard<std::mutex> lock(temporary_pool_mutex);
            auto bucket = temporary_pool.find(capacity);
            if (bucket != temporary_pool.end() && !bucket->second.empty()) {
                buffer = bucket->second.back();
                bucket->second.pop_back();
                temporary_pool_bytes -= capacity;
            }
        }
        if (buffer == nil) {
            buffer = make_buffer(nullptr, capacity);
        }
        if (buffer == nil) {
            throw std::runtime_error(std::string(
                "Metal failed to allocate ") + operation + " buffer");
        }
        execution.temporary_buffers.push_back(
            {buffer, capacity, reusable});
        return buffer;
    }

    void recycle_temporary_buffers(AsyncExecution& execution) const {
        std::lock_guard<std::mutex> lock(temporary_pool_mutex);
        for (const AsyncExecution::TemporaryBuffer& temporary :
             execution.temporary_buffers) {
            if (temporary.reusable &&
                temporary_pool_bytes <= kMaxTemporaryPoolBytes -
                    temporary.capacity_bytes) {
                temporary_pool[temporary.capacity_bytes].push_back(
                    temporary.buffer);
                temporary_pool_bytes += temporary.capacity_bytes;
            }
        }
        execution.temporary_buffers.clear();
    }

    id<MTLBuffer> async_buffer(AsyncExecution& execution,
                               const void* data,
                               size_t bytes,
                               const char* operation) const {
        id<MTLBuffer> buffer = acquire_temporary_buffer(
            execution, bytes, operation);
        if (data != nullptr) {
            std::memcpy([buffer contents], data, bytes);
        }
        return buffer;
    }

    id<MTLComputeCommandEncoder> async_encoder(
        AsyncExecution& execution,
        id<MTLComputePipelineState> pipeline,
        const char* operation) const {
        id<MTLComputeCommandEncoder> encoder =
            [execution.command computeCommandEncoder];
        if (encoder == nil) {
            throw std::runtime_error(std::string(
                "Metal failed to create ") + operation + " encoder");
        }
        [encoder setComputePipelineState:pipeline];
        return encoder;
    }

    void finish_async(AsyncExecution& execution, const char* operation) const {
        [execution.command commit];
        [execution.command waitUntilCompleted];
        check_command(execution.command, operation);
    }

    DeviceMatrix allocate_device_matrix(AsyncExecution& execution,
                                        size_t rows,
                                        size_t cols,
                                        const char* operation) const {
        const size_t elements = checked_elements(rows, cols, operation);
        id<MTLBuffer> buffer = async_buffer(
            execution, nullptr, checked_bytes(elements, operation), operation);
        return {buffer, rows, cols, cols, 0};
    }

    DeviceVector allocate_device_vector(AsyncExecution& execution,
                                        size_t length,
                                        const char* operation) const {
        id<MTLBuffer> buffer = async_buffer(
            execution, nullptr, checked_bytes(length, operation), operation);
        return {buffer, length, 0};
    }

    id<MTLBuffer> async_token_ids(AsyncExecution& execution,
                                  const std::vector<int32_t>& token_ids) const {
        if (token_ids.empty()) {
            throw std::invalid_argument(
                "Metal resident execution requires at least one token");
        }
        return async_buffer(
            execution, token_ids.data(),
            checked_bytes_for(token_ids.size(), sizeof(int32_t),
                              "Metal token ids"),
            "token ids");
    }

    DeviceMatrix encode_embedding(AsyncExecution& execution,
                                  id<MTLBuffer> token_ids,
                                  const DeviceMatrix& embedding,
                                  size_t sequence_length,
                                  size_t vocabulary_size) const {
        if (embedding.rows != vocabulary_size || embedding.cols == 0) {
            throw std::invalid_argument(
                "Metal resident embedding weight shape is invalid");
        }
        DeviceMatrix output = allocate_device_matrix(
            execution, sequence_length, embedding.cols,
            "Metal resident embedding");
        MetalEmbeddingParamsHost params;
        params.sequence_length = checked_uint(
            sequence_length, "Resident embedding sequence length");
        params.embedding_size = checked_uint(
            embedding.cols, "Resident embedding size");
        params.vocabulary_size = checked_uint(
            vocabulary_size, "Resident embedding vocabulary size");

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, embedding_pipeline, "resident embedding");
        [encoder setBuffer:token_ids offset:0 atIndex:0];
        [encoder setBuffer:embedding.buffer
                   offset:checked_bytes_for(embedding.offset_elements,
                                            sizeof(float),
                                            "Resident embedding offset")
                  atIndex:1];
        [encoder setBuffer:output.buffer offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_flat(encoder, embedding_pipeline,
                      checked_elements(sequence_length, embedding.cols,
                                       "Resident embedding dispatch"));
        [encoder endEncoding];
        return output;
    }

    DeviceMatrix encode_matrix_product(AsyncExecution& execution,
                                       const DeviceMatrix& left,
                                       const DeviceMatrix& right,
                                       bool left_transposed,
                                       bool right_transposed,
                                       const DeviceVector* bias,
                                       float scale,
                                       const char* operation) const {
        if (!std::isfinite(scale)) {
            throw std::invalid_argument(
                "Metal resident GEMM scale must be finite");
        }
        const size_t m = left_transposed ? left.cols : left.rows;
        const size_t left_k = left_transposed ? left.rows : left.cols;
        const size_t right_k = right_transposed ? right.cols : right.rows;
        const size_t n = right_transposed ? right.rows : right.cols;
        if (left_k != right_k) {
            throw std::invalid_argument(
                "Metal resident GEMM inner dimensions do not match");
        }
        if (bias != nullptr && bias->length != n) {
            throw std::invalid_argument(
                "Metal resident GEMM bias dimension does not match output");
        }

        DeviceMatrix output = allocate_device_matrix(
            execution, m, n, operation);
        if (m == 0 || n == 0) {
            return output;
        }

        MetalMatmulParamsHost params;
        params.m = checked_uint(m, "Resident GEMM M");
        params.n = checked_uint(n, "Resident GEMM N");
        params.k = checked_uint(left_k, "Resident GEMM K");
        params.lhs_stride = checked_uint(left.stride,
                                         "Resident GEMM lhs stride");
        params.rhs_stride = checked_uint(right.stride,
                                         "Resident GEMM rhs stride");
        params.lhs_transposed = left_transposed ? 1U : 0U;
        params.rhs_transposed = right_transposed ? 1U : 0U;
        params.has_bias = bias == nullptr ? 0U : 1U;
        params.scale = scale;

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, gemm_pipeline, operation);
        [encoder setBuffer:left.buffer
                   offset:checked_bytes_for(left.offset_elements,
                                            sizeof(float),
                                            "Resident GEMM lhs offset")
                  atIndex:0];
        [encoder setBuffer:right.buffer
                   offset:checked_bytes_for(right.offset_elements,
                                            sizeof(float),
                                            "Resident GEMM rhs offset")
                  atIndex:1];
        id<MTLBuffer> bias_buffer = bias == nullptr
            ? dummy_buffer
            : bias->buffer;
        const size_t bias_offset = bias == nullptr ? 0 : bias->offset_elements;
        [encoder setBuffer:bias_buffer
                   offset:checked_bytes_for(bias_offset, sizeof(float),
                                            "Resident GEMM bias offset")
                  atIndex:2];
        [encoder setBuffer:output.buffer offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];

        const MTLSize threadgroups = MTLSizeMake(
            (static_cast<NSUInteger>(n) + kTileSize - 1) / kTileSize,
            (static_cast<NSUInteger>(m) + kTileSize - 1) / kTileSize, 1);
        [encoder dispatchThreadgroups:threadgroups
                 threadsPerThreadgroup:MTLSizeMake(kTileSize, kTileSize, 1)];
        [encoder endEncoding];
        return output;
    }

    DeviceVector encode_vector_product(AsyncExecution& execution,
                                       const DeviceMatrix& matrix,
                                       const DeviceVector& input,
                                       bool matrix_transposed,
                                       const DeviceVector* bias,
                                       float scale,
                                       const char* operation) const {
        if (!std::isfinite(scale)) {
            throw std::invalid_argument(
                "Metal resident GEVM scale must be finite");
        }
        const size_t input_size = matrix_transposed
            ? matrix.cols : matrix.rows;
        const size_t output_size = matrix_transposed
            ? matrix.rows : matrix.cols;
        if (input.length != input_size) {
            throw std::invalid_argument(
                "Metal resident GEVM inner dimensions do not match");
        }
        if (bias != nullptr && bias->length != output_size) {
            throw std::invalid_argument(
                "Metal resident GEVM bias dimension does not match output");
        }

        DeviceVector output = allocate_device_vector(
            execution, output_size, operation);
        if (output_size == 0) {
            return output;
        }

        MetalGemvParamsHost params;
        params.output_size = checked_uint(output_size, "Resident GEVM output");
        params.input_size = checked_uint(input_size, "Resident GEVM input");
        params.matrix_stride = checked_uint(
            matrix.stride, "Resident GEVM matrix stride");
        params.matrix_transposed = matrix_transposed ? 1U : 0U;
        params.has_bias = bias == nullptr ? 0U : 1U;
        params.scale = scale;

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, gevm_pipeline, operation);
        [encoder setBuffer:matrix.buffer
                   offset:checked_bytes_for(matrix.offset_elements,
                                            sizeof(float),
                                            "Resident GEVM matrix offset")
                  atIndex:0];
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident GEVM input offset")
                  atIndex:1];
        id<MTLBuffer> bias_buffer = bias == nullptr
            ? dummy_buffer
            : bias->buffer;
        const size_t bias_offset = bias == nullptr ? 0 : bias->offset_elements;
        [encoder setBuffer:bias_buffer
                   offset:checked_bytes_for(bias_offset, sizeof(float),
                                            "Resident GEVM bias offset")
                  atIndex:2];
        [encoder setBuffer:output.buffer offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        const NSUInteger threads = std::min<NSUInteger>(
            kDefaultGemvThreads,
            std::max<NSUInteger>(1, [gevm_pipeline maxTotalThreadsPerThreadgroup]));
        [encoder dispatchThreadgroups:MTLSizeMake(
            (static_cast<NSUInteger>(output_size) + threads - 1) / threads,
            1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        return output;
    }

    DeviceMatrix encode_rmsnorm_matrix(AsyncExecution& execution,
                                       const DeviceMatrix& input,
                                       const DeviceVector& gamma,
                                       float epsilon,
                                       const char* operation) const {
        if (gamma.length != input.cols || input.rows == 0 || input.cols == 0) {
            throw std::invalid_argument(
                "Metal resident RMSNorm matrix dimensions are invalid");
        }
        DeviceMatrix output = allocate_device_matrix(
            execution, input.rows, input.cols, operation);
        MetalRmsNormParamsHost params;
        params.rows = checked_uint(input.rows, "Resident RMSNorm rows");
        params.cols = checked_uint(input.cols, "Resident RMSNorm columns");
        params.epsilon = epsilon;

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, rmsnorm_pipeline, operation);
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident RMSNorm input offset")
                  atIndex:0];
        [encoder setBuffer:gamma.buffer
                   offset:checked_bytes_for(gamma.offset_elements,
                                            sizeof(float),
                                            "Resident RMSNorm gamma offset")
                  atIndex:1];
        [encoder setBuffer:output.buffer offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_rows(encoder, input.rows);
        [encoder endEncoding];
        return output;
    }

    DeviceVector encode_rmsnorm_vector(AsyncExecution& execution,
                                       const DeviceVector& input,
                                       const DeviceVector& gamma,
                                       float epsilon,
                                       const char* operation) const {
        if (input.length == 0 || gamma.length != input.length) {
            throw std::invalid_argument(
                "Metal resident RMSNorm vector dimensions are invalid");
        }
        DeviceVector output = allocate_device_vector(
            execution, input.length, operation);
        MetalRmsNormParamsHost params;
        params.rows = 1;
        params.cols = checked_uint(input.length, "Resident RMSNorm length");
        params.epsilon = epsilon;

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, rmsnorm_pipeline, operation);
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident RMSNorm input offset")
                  atIndex:0];
        [encoder setBuffer:gamma.buffer
                   offset:checked_bytes_for(gamma.offset_elements,
                                            sizeof(float),
                                            "Resident RMSNorm gamma offset")
                  atIndex:1];
        [encoder setBuffer:output.buffer offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_rows(encoder, 1);
        [encoder endEncoding];
        return output;
    }

    DeviceMatrix encode_softmax_matrix(AsyncExecution& execution,
                                       const DeviceMatrix& input,
                                       const char* operation) const {
        if (input.rows == 0 || input.cols == 0) {
            throw std::invalid_argument(
                "Metal resident Softmax matrix cannot be empty");
        }
        DeviceMatrix output = allocate_device_matrix(
            execution, input.rows, input.cols, operation);
        MetalElementwiseParamsHost params;
        params.rows = checked_uint(input.rows, "Resident Softmax rows");
        params.cols = checked_uint(input.cols, "Resident Softmax columns");

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, softmax_pipeline, operation);
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident Softmax input offset")
                  atIndex:0];
        [encoder setBuffer:output.buffer offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_rows(encoder, input.rows);
        [encoder endEncoding];
        return output;
    }

    DeviceVector encode_softmax_vector(AsyncExecution& execution,
                                       const DeviceVector& input,
                                       const char* operation) const {
        if (input.length == 0) {
            throw std::invalid_argument(
                "Metal resident Softmax vector cannot be empty");
        }
        DeviceVector output = allocate_device_vector(
            execution, input.length, operation);
        MetalElementwiseParamsHost params;
        params.rows = 1;
        params.cols = checked_uint(input.length, "Resident Softmax length");

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, softmax_pipeline, operation);
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident Softmax input offset")
                  atIndex:0];
        [encoder setBuffer:output.buffer offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_rows(encoder, 1);
        [encoder endEncoding];
        return output;
    }

    DeviceMatrix encode_binary_matrix(AsyncExecution& execution,
                                      const DeviceMatrix& left,
                                      const DeviceMatrix& right,
                                      id<MTLComputePipelineState> pipeline,
                                      const char* operation) const {
        if (left.rows != right.rows || left.cols != right.cols ||
            left.rows == 0 || left.cols == 0) {
            throw std::invalid_argument(
                "Metal resident binary matrix dimensions are invalid");
        }
        DeviceMatrix output = allocate_device_matrix(
            execution, left.rows, left.cols, operation);
        MetalElementwiseParamsHost params;
        params.rows = checked_uint(left.rows, "Resident binary rows");
        params.cols = checked_uint(left.cols, "Resident binary columns");
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, pipeline, operation);
        [encoder setBuffer:left.buffer
                   offset:checked_bytes_for(left.offset_elements,
                                            sizeof(float),
                                            "Resident binary left offset")
                  atIndex:0];
        [encoder setBuffer:right.buffer
                   offset:checked_bytes_for(right.offset_elements,
                                            sizeof(float),
                                            "Resident binary right offset")
                  atIndex:1];
        [encoder setBuffer:output.buffer offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_flat(encoder, pipeline,
                      checked_elements(left.rows, left.cols,
                                       "Resident binary dispatch"));
        [encoder endEncoding];
        return output;
    }

    DeviceVector encode_binary_vector(AsyncExecution& execution,
                                      const DeviceVector& left,
                                      const DeviceVector& right,
                                      id<MTLComputePipelineState> pipeline,
                                      const char* operation) const {
        if (left.length != right.length || left.length == 0) {
            throw std::invalid_argument(
                "Metal resident binary vector dimensions are invalid");
        }
        DeviceVector output = allocate_device_vector(
            execution, left.length, operation);
        MetalElementwiseParamsHost params;
        params.rows = 1;
        params.cols = checked_uint(left.length, "Resident binary length");
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, pipeline, operation);
        [encoder setBuffer:left.buffer
                   offset:checked_bytes_for(left.offset_elements,
                                            sizeof(float),
                                            "Resident binary left offset")
                  atIndex:0];
        [encoder setBuffer:right.buffer
                   offset:checked_bytes_for(right.offset_elements,
                                            sizeof(float),
                                            "Resident binary right offset")
                  atIndex:1];
        [encoder setBuffer:output.buffer offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_flat(encoder, pipeline, left.length);
        [encoder endEncoding];
        return output;
    }

    DeviceMatrix encode_rope_heads_matrix(AsyncExecution& execution,
                                          const DeviceMatrix& input,
                                          size_t head_dim,
                                          size_t rotary_dimension,
                                          size_t position,
                                          float theta,
                                          const char* operation) const {
        if (input.rows == 0 || input.cols == 0 || head_dim == 0 ||
            input.cols % head_dim != 0 || rotary_dimension == 0 ||
            rotary_dimension > head_dim || rotary_dimension % 2 != 0 ||
            !std::isfinite(theta) || theta <= 0.0f) {
            throw std::invalid_argument(
                "Metal resident head-wise RoPE matrix dimensions are invalid");
        }
        DeviceMatrix output = allocate_device_matrix(
            execution, input.rows, input.cols, operation);
        MetalRopeHeadsParamsHost params;
        params.rows = checked_uint(input.rows, "Resident RoPE rows");
        params.cols = checked_uint(input.cols, "Resident RoPE columns");
        params.head_dim = checked_uint(head_dim, "Resident RoPE head dimension");
        params.rotary_dimension = checked_uint(
            rotary_dimension, "Resident RoPE rotary dimension");
        params.position = checked_uint(position, "Resident RoPE position");
        params.theta = theta;
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, rope_heads_pipeline, operation);
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident RoPE input offset")
                  atIndex:0];
        [encoder setBuffer:output.buffer offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_flat(encoder, rope_heads_pipeline,
                      checked_elements(input.rows, input.cols,
                                       "Resident RoPE dispatch"));
        [encoder endEncoding];
        return output;
    }

    DeviceVector encode_rope_heads_vector(AsyncExecution& execution,
                                          const DeviceVector& input,
                                          size_t head_dim,
                                          size_t rotary_dimension,
                                          size_t position,
                                          float theta,
                                          const char* operation) const {
        if (input.length == 0 || head_dim == 0 ||
            input.length % head_dim != 0 || rotary_dimension == 0 ||
            rotary_dimension > head_dim || rotary_dimension % 2 != 0 ||
            !std::isfinite(theta) || theta <= 0.0f) {
            throw std::invalid_argument(
                "Metal resident head-wise RoPE vector dimensions are invalid");
        }
        DeviceVector output = allocate_device_vector(
            execution, input.length, operation);
        MetalRopeHeadsParamsHost params;
        params.rows = 1;
        params.cols = checked_uint(input.length, "Resident RoPE length");
        params.head_dim = checked_uint(head_dim, "Resident RoPE head dimension");
        params.rotary_dimension = checked_uint(
            rotary_dimension, "Resident RoPE rotary dimension");
        params.position = checked_uint(position, "Resident RoPE position");
        params.theta = theta;
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, rope_heads_pipeline, operation);
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident RoPE input offset")
                  atIndex:0];
        [encoder setBuffer:output.buffer offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_flat(encoder, rope_heads_pipeline, input.length);
        [encoder endEncoding];
        return output;
    }

    void encode_kv_cache_write(AsyncExecution& execution,
                               const DeviceMatrix& key_source,
                               const DeviceMatrix& value_source,
                               id<MTLBuffer> key_cache,
                               id<MTLBuffer> value_cache,
                               size_t position,
                               const char* operation) const {
        if (key_source.rows != value_source.rows ||
            key_source.cols != value_source.cols || key_source.rows == 0 ||
            key_source.cols == 0 || key_cache == nil || value_cache == nil) {
            throw std::invalid_argument(
                "Metal resident KV-cache write dimensions are invalid");
        }

        MetalKVCacheWriteParamsHost params;
        params.source_rows = checked_uint(
            key_source.rows, "Resident KV write rows");
        params.source_cols = checked_uint(
            key_source.cols, "Resident KV write columns");
        params.cache_stride = checked_uint(
            key_source.cols, "Resident KV write cache stride");
        params.position = checked_uint(
            position, "Resident KV write position");

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, kv_cache_write_pipeline, operation);
        [encoder setBuffer:key_source.buffer
                   offset:checked_bytes_for(key_source.offset_elements,
                                            sizeof(float),
                                            "Resident KV key offset")
                  atIndex:0];
        [encoder setBuffer:value_source.buffer
                   offset:checked_bytes_for(value_source.offset_elements,
                                            sizeof(float),
                                            "Resident KV value offset")
                  atIndex:1];
        [encoder setBuffer:key_cache offset:0 atIndex:2];
        [encoder setBuffer:value_cache offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        dispatch_flat(encoder, kv_cache_write_pipeline,
                      checked_elements(key_source.rows, key_source.cols,
                                       "Resident KV write dispatch"));
        [encoder endEncoding];
    }

    DeviceMatrix encode_kv_cache_qk(AsyncExecution& execution,
                                    id<MTLBuffer> query,
                                    size_t query_offset_elements,
                                    size_t query_rows,
                                    size_t key_length,
                                    id<MTLBuffer> key_cache,
                                    size_t cache_stride,
                                    size_t query_stride,
                                    size_t head_dim,
                                    size_t cache_head,
                                    size_t query_position,
                                    float scale,
                                    const char* operation) const {
        if (query == nil || key_cache == nil || query_rows == 0 ||
            key_length == 0 || head_dim == 0) {
            throw std::invalid_argument(
                "Metal resident KV-cache QK dimensions are invalid");
        }

        DeviceMatrix output = allocate_device_matrix(
            execution, query_rows, key_length, operation);
        MetalKVCacheQKParamsHost params;
        params.query_rows = checked_uint(
            query_rows, "Resident QK query rows");
        params.key_length = checked_uint(
            key_length, "Resident QK key length");
        params.cache_stride = checked_uint(
            cache_stride, "Resident QK cache stride");
        params.query_stride = checked_uint(
            query_stride, "Resident QK query stride");
        params.head_dim = checked_uint(
            head_dim, "Resident QK head dimension");
        params.cache_head = checked_uint(
            cache_head, "Resident QK cache head");
        params.query_position = checked_uint(
            query_position, "Resident QK query position");
        params.scale = scale;

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, kv_cache_qk_pipeline, operation);
        [encoder setBuffer:query
                   offset:checked_bytes_for(query_offset_elements,
                                            sizeof(float),
                                            "Resident QK query offset")
                  atIndex:0];
        [encoder setBuffer:key_cache offset:0 atIndex:1];
        [encoder setBuffer:output.buffer offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_flat(encoder, kv_cache_qk_pipeline,
                      checked_elements(query_rows, key_length,
                                       "Resident QK dispatch"));
        [encoder endEncoding];
        return output;
    }

    void encode_kv_cache_av(AsyncExecution& execution,
                            id<MTLBuffer> scores,
                            id<MTLBuffer> value_cache,
                            id<MTLBuffer> output,
                            size_t query_rows,
                            size_t key_length,
                            size_t cache_stride,
                            size_t score_stride,
                            size_t head_dim,
                            size_t cache_head,
                            size_t output_stride,
                            size_t output_offset,
                            const char* operation) const {
        if (scores == nil || value_cache == nil || output == nil ||
            query_rows == 0 || key_length == 0 || head_dim == 0 ||
            output_stride == 0) {
            throw std::invalid_argument(
                "Metal resident KV-cache AV dimensions are invalid");
        }

        MetalKVCacheAVParamsHost params;
        params.query_rows = checked_uint(
            query_rows, "Resident AV query rows");
        params.key_length = checked_uint(
            key_length, "Resident AV key length");
        params.cache_stride = checked_uint(
            cache_stride, "Resident AV cache stride");
        params.score_stride = checked_uint(
            score_stride, "Resident AV score stride");
        params.head_dim = checked_uint(
            head_dim, "Resident AV head dimension");
        params.cache_head = checked_uint(
            cache_head, "Resident AV cache head");
        params.output_stride = checked_uint(
            output_stride, "Resident AV output stride");
        params.output_offset = checked_uint(
            output_offset, "Resident AV output offset");

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, kv_cache_av_pipeline, operation);
        [encoder setBuffer:scores offset:0 atIndex:0];
        [encoder setBuffer:value_cache offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_flat(encoder, kv_cache_av_pipeline,
                      checked_elements(query_rows, head_dim,
                                       "Resident AV dispatch"));
        [encoder endEncoding];
    }

    id<MTLBuffer> encode_argmax(AsyncExecution& execution,
                                id<MTLBuffer> input,
                                size_t length,
                                const char* operation) const {
        if (input == nil || length == 0) {
            throw std::invalid_argument(
                "Metal resident argmax input cannot be empty");
        }
        id<MTLBuffer> output = async_buffer(
            execution, nullptr, sizeof(uint32_t), operation);
        MetalArgmaxParamsHost params;
        params.length = checked_uint(length, "Resident argmax length");
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, argmax_pipeline, operation);
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
        return output;
    }

    void check_command(id<MTLCommandBuffer> command,
                       const char* operation) const {
        NSError* error = [command error];
        if (error != nil) {
            throw_metal_error(operation, error);
        }
        const MTLCommandBufferStatus status = [command status];
        if (status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error(std::string(operation) +
                                     ": command buffer did not complete");
        }
    }

    DeviceMatrix encode_prefill_layer(AsyncExecution& execution,
                                      const DeviceMatrix& hidden,
                                      const PreparedLayer& layer,
                                      float epsilon,
                                      size_t d_rope,
                                      float theta,
                                      size_t d_head,
                                      id<MTLBuffer> key_cache,
                                      id<MTLBuffer> value_cache,
                                      size_t cache_stride,
                                      size_t position) const;

    DeviceVector encode_decode_layer(AsyncExecution& execution,
                                     const DeviceVector& hidden,
                                     const PreparedLayer& layer,
                                     float epsilon,
                                     size_t d_rope,
                                     float theta,
                                     size_t d_head,
                                     id<MTLBuffer> key_cache,
                                     id<MTLBuffer> value_cache,
                                     size_t cache_stride,
                                     size_t position) const;

    void upload_model(const std::vector<Layer>& layers,
                      const Matrix& output_weight,
                      const Matrix& token_embedding_weight,
                      const Vector& output_norm_weight);
    void allocate_kv_cache(size_t capacity);
    void load_model(const std::string& gguf_path, size_t capacity);
};

MetalLLM::Impl::DeviceMatrix MetalLLM::Impl::encode_prefill_layer(
    AsyncExecution& execution,
    const DeviceMatrix& hidden,
    const PreparedLayer& layer,
    float epsilon,
    size_t d_rope,
    float theta,
    size_t d_head,
    id<MTLBuffer> key_cache,
    id<MTLBuffer> value_cache,
    size_t cache_stride,
    size_t position) const {
    const size_t q_dimension = layer.attn_q_weight.rows;
    const size_t kv_dimension = layer.attn_k_weight.rows;
    const size_t q_head_count = q_dimension / d_head;
    const size_t kv_head_count = kv_dimension / d_head;
    if (hidden.rows == 0 || hidden.cols == 0 || d_head == 0 ||
        q_dimension == 0 || kv_dimension == 0 ||
        q_dimension % d_head != 0 || kv_dimension % d_head != 0 ||
        q_head_count == 0 || kv_head_count == 0 ||
        q_head_count % kv_head_count != 0 || cache_stride != kv_dimension) {
        throw std::invalid_argument(
            "Metal resident prefill layer dimensions are invalid");
    }
    const size_t key_length = position + hidden.rows;
    if (key_length < position) {
        throw std::length_error(
            "Metal resident prefill key length overflows");
    }

    const DeviceVector& attn_norm = layer.attn_norm_weight;
    const DeviceMatrix normalized = encode_rmsnorm_matrix(
        execution, hidden, attn_norm, epsilon, "resident prefill attention norm");

    const DeviceMatrix& q_weight = layer.attn_q_weight;
    const DeviceVector& q_bias = layer.attn_q_bias;
    const DeviceMatrix& k_weight = layer.attn_k_weight;
    const DeviceVector& k_bias = layer.attn_k_bias;
    const DeviceMatrix& v_weight = layer.attn_v_weight;
    const DeviceVector& v_bias = layer.attn_v_bias;

    const DeviceMatrix query = encode_matrix_product(
        execution, normalized, q_weight, false, true, &q_bias, 1.0f,
        "resident prefill Q projection");
    const DeviceMatrix key = encode_matrix_product(
        execution, normalized, k_weight, false, true, &k_bias, 1.0f,
        "resident prefill K projection");
    const DeviceMatrix value = encode_matrix_product(
        execution, normalized, v_weight, false, true, &v_bias, 1.0f,
        "resident prefill V projection");

    const DeviceMatrix rotated_query = encode_rope_heads_matrix(
        execution, query, d_head, d_rope, position, theta,
        "resident prefill Q RoPE");
    const DeviceMatrix rotated_key = encode_rope_heads_matrix(
        execution, key, d_head, d_rope, position, theta,
        "resident prefill K RoPE");
    encode_kv_cache_write(
        execution, rotated_key, value, key_cache, value_cache, position,
        "resident prefill KV write");

    // AV writes each head directly into its final packed [sequence, Q]
    // layout. This removes both per-head split buffers and the later concat.
    const DeviceMatrix attention_output = allocate_device_matrix(
        execution, hidden.rows, q_dimension,
        "resident prefill attention output");
    const size_t group_size = q_head_count / kv_head_count;
    for (size_t head = 0; head < q_head_count; ++head) {
        const size_t group = head / group_size;
        const size_t head_offset = checked_elements(
            head, d_head, "Resident prefill Q head offset");
        const DeviceMatrix scores = encode_kv_cache_qk(
            execution, rotated_query.buffer, head_offset, hidden.rows,
            key_length, key_cache, cache_stride, q_dimension, d_head, group,
            position, 1.0f / std::sqrt(static_cast<float>(d_head)),
            "resident prefill QK");
        const DeviceMatrix probabilities = encode_softmax_matrix(
            execution, scores, "resident prefill Softmax");
        encode_kv_cache_av(
            execution, probabilities.buffer, value_cache,
            attention_output.buffer, hidden.rows, key_length, cache_stride,
            key_length, d_head, group, q_dimension, head_offset,
            "resident prefill AV");
    }

    const DeviceMatrix& output_weight = layer.attn_output_weight;
    const DeviceMatrix attention_projected = encode_matrix_product(
        execution, attention_output, output_weight, false, true, nullptr, 1.0f,
        "resident prefill attention output projection");
    const DeviceMatrix after_attention = encode_binary_matrix(
        execution, hidden, attention_projected, residual_pipeline,
        "resident prefill attention residual");

    const DeviceVector& ffn_norm = layer.ffn_norm_weight;
    const DeviceMatrix ffn_input = encode_rmsnorm_matrix(
        execution, after_attention, ffn_norm, epsilon,
        "resident prefill FFN norm");
    const DeviceMatrix& gate_weight = layer.ffn_gate_weight;
    const DeviceMatrix& up_weight = layer.ffn_up_weight;
    const DeviceMatrix& down_weight = layer.ffn_down_weight;
    const DeviceMatrix gate = encode_matrix_product(
        execution, ffn_input, gate_weight, false, true, nullptr, 1.0f,
        "resident prefill FFN gate projection");
    const DeviceMatrix up = encode_matrix_product(
        execution, ffn_input, up_weight, false, true, nullptr, 1.0f,
        "resident prefill FFN up projection");
    const DeviceMatrix activated = encode_binary_matrix(
        execution, gate, up, swiglu_pipeline, "resident prefill SwiGLU");
    const DeviceMatrix down = encode_matrix_product(
        execution, activated, down_weight, false, true, nullptr, 1.0f,
        "resident prefill FFN down projection");
    return encode_binary_matrix(
        execution, after_attention, down, residual_pipeline,
        "resident prefill FFN residual");
}

MetalLLM::Impl::DeviceVector MetalLLM::Impl::encode_decode_layer(
    AsyncExecution& execution,
    const DeviceVector& hidden,
    const PreparedLayer& layer,
    float epsilon,
    size_t d_rope,
    float theta,
    size_t d_head,
    id<MTLBuffer> key_cache,
    id<MTLBuffer> value_cache,
    size_t cache_stride,
    size_t position) const {
    const size_t q_dimension = layer.attn_q_weight.rows;
    const size_t kv_dimension = layer.attn_k_weight.rows;
    const size_t q_head_count = q_dimension / d_head;
    const size_t kv_head_count = kv_dimension / d_head;
    if (hidden.length == 0 || d_head == 0 || q_dimension == 0 ||
        kv_dimension == 0 || q_dimension % d_head != 0 ||
        kv_dimension % d_head != 0 || q_head_count == 0 ||
        kv_head_count == 0 || q_head_count % kv_head_count != 0 ||
        cache_stride != kv_dimension) {
        throw std::invalid_argument(
            "Metal resident decode layer dimensions are invalid");
    }
    if (position == std::numeric_limits<size_t>::max()) {
        throw std::length_error(
            "Metal resident decode position overflows");
    }
    const size_t key_length = position + 1;

    const DeviceVector& attn_norm = layer.attn_norm_weight;
    const DeviceVector normalized = encode_rmsnorm_vector(
        execution, hidden, attn_norm, epsilon, "resident decode attention norm");

    const DeviceMatrix& q_weight = layer.attn_q_weight;
    const DeviceVector& q_bias = layer.attn_q_bias;
    const DeviceMatrix& k_weight = layer.attn_k_weight;
    const DeviceVector& k_bias = layer.attn_k_bias;
    const DeviceMatrix& v_weight = layer.attn_v_weight;
    const DeviceVector& v_bias = layer.attn_v_bias;

    const DeviceVector query = encode_vector_product(
        execution, q_weight, normalized, true, &q_bias, 1.0f,
        "resident decode Q projection");
    const DeviceVector key = encode_vector_product(
        execution, k_weight, normalized, true, &k_bias, 1.0f,
        "resident decode K projection");
    const DeviceVector value = encode_vector_product(
        execution, v_weight, normalized, true, &v_bias, 1.0f,
        "resident decode V projection");
    const DeviceVector rotated_query = encode_rope_heads_vector(
        execution, query, d_head, d_rope, position, theta,
        "resident decode Q RoPE");
    const DeviceVector rotated_key = encode_rope_heads_vector(
        execution, key, d_head, d_rope, position, theta,
        "resident decode K RoPE");

    const DeviceMatrix key_source = {
        rotated_key.buffer, 1, rotated_key.length, rotated_key.length,
        rotated_key.offset_elements};
    const DeviceMatrix value_source = {
        value.buffer, 1, value.length, value.length, value.offset_elements};
    encode_kv_cache_write(
        execution, key_source, value_source, key_cache, value_cache, position,
        "resident decode KV write");

    const DeviceVector attention_output = allocate_device_vector(
        execution, q_dimension, "resident decode attention output");
    const size_t group_size = q_head_count / kv_head_count;
    for (size_t head = 0; head < q_head_count; ++head) {
        const size_t group = head / group_size;
        const size_t head_offset = checked_elements(
            head, d_head, "Resident decode Q head offset");
        const DeviceMatrix scores = encode_kv_cache_qk(
            execution, rotated_query.buffer, head_offset, 1, key_length,
            key_cache, cache_stride, q_dimension, d_head, group, position,
            1.0f / std::sqrt(static_cast<float>(d_head)),
            "resident decode QK");
        const DeviceVector score_vector = {
            scores.buffer, key_length, scores.offset_elements};
        const DeviceVector probabilities = encode_softmax_vector(
            execution, score_vector, "resident decode Softmax");
        encode_kv_cache_av(
            execution, probabilities.buffer, value_cache,
            attention_output.buffer, 1, key_length, cache_stride, key_length,
            d_head, group, q_dimension, head_offset, "resident decode AV");
    }

    const DeviceMatrix& output_weight = layer.attn_output_weight;
    const DeviceVector attention_projected = encode_vector_product(
        execution, output_weight, attention_output, true, nullptr, 1.0f,
        "resident decode attention output projection");
    const DeviceVector after_attention = encode_binary_vector(
        execution, hidden, attention_projected, residual_pipeline,
        "resident decode attention residual");

    const DeviceVector& ffn_norm = layer.ffn_norm_weight;
    const DeviceVector ffn_input = encode_rmsnorm_vector(
        execution, after_attention, ffn_norm, epsilon,
        "resident decode FFN norm");
    const DeviceMatrix& gate_weight = layer.ffn_gate_weight;
    const DeviceMatrix& up_weight = layer.ffn_up_weight;
    const DeviceMatrix& down_weight = layer.ffn_down_weight;
    const DeviceVector gate = encode_vector_product(
        execution, gate_weight, ffn_input, true, nullptr, 1.0f,
        "resident decode FFN gate projection");
    const DeviceVector up = encode_vector_product(
        execution, up_weight, ffn_input, true, nullptr, 1.0f,
        "resident decode FFN up projection");
    const DeviceVector activated = encode_binary_vector(
        execution, gate, up, swiglu_pipeline, "resident decode SwiGLU");
    const DeviceVector down = encode_vector_product(
        execution, down_weight, activated, true, nullptr, 1.0f,
        "resident decode FFN down projection");
    return encode_binary_vector(
        execution, after_attention, down, residual_pipeline,
        "resident decode FFN residual");
}

void MetalLLM::Impl::upload_model(
    const std::vector<Layer>& layers,
    const Matrix& output_weight,
    const Matrix& token_embedding_weight,
    const Vector& output_norm_weight) {
    require_available();
    if (layers.empty()) {
        throw std::invalid_argument(
            "Metal model preparation requires at least one layer");
    }

    @autoreleasepool {
        PreparedModel next;

        // These maps matter when a caller supplies tied host tensor objects.
        // They exist only during initialization; inference uses only the
        // descriptors retained in PreparedModel.
        std::unordered_map<const Matrix*, DeviceMatrix> matrices;
        std::unordered_map<const Vector*, DeviceVector> vectors;
        const auto upload_matrix_once =
            [&](const Matrix& matrix, const char* name) -> DeviceMatrix {
                const auto found = matrices.find(&matrix);
                if (found != matrices.end()) {
                    return found->second;
                }
                const DeviceMatrix uploaded = upload_matrix(matrix, name);
                matrices.emplace(&matrix, uploaded);
                return uploaded;
            };
        const auto upload_vector_once =
            [&](const Vector& vector, const char* name) -> DeviceVector {
                const auto found = vectors.find(&vector);
                if (found != vectors.end()) {
                    return found->second;
                }
                const DeviceVector uploaded = upload_vector(vector, name);
                vectors.emplace(&vector, uploaded);
                return uploaded;
            };

        next.output_weight = upload_matrix_once(
            output_weight, "Metal output weight");
        next.token_embedding_weight = upload_matrix_once(
            token_embedding_weight, "Metal token embedding weight");
        next.output_norm_weight = upload_vector_once(
            output_norm_weight, "Metal output norm weight");
        next.layers.reserve(layers.size());
        for (const Layer& layer : layers) {
            PreparedLayer prepared_layer;
            prepared_layer.attn_norm_weight = upload_vector_once(
                layer.attn_norm_weight, "Metal attention norm weight");
            prepared_layer.attn_q_weight = upload_matrix_once(
                layer.attn_q_weight, "Metal Q projection weight");
            prepared_layer.attn_q_bias = upload_vector_once(
                layer.attn_q_bias, "Metal Q projection bias");
            prepared_layer.attn_k_weight = upload_matrix_once(
                layer.attn_k_weight, "Metal K projection weight");
            prepared_layer.attn_k_bias = upload_vector_once(
                layer.attn_k_bias, "Metal K projection bias");
            prepared_layer.attn_v_weight = upload_matrix_once(
                layer.attn_v_weight, "Metal V projection weight");
            prepared_layer.attn_v_bias = upload_vector_once(
                layer.attn_v_bias, "Metal V projection bias");
            prepared_layer.attn_output_weight = upload_matrix_once(
                layer.attn_output_weight, "Metal attention output weight");
            prepared_layer.ffn_norm_weight = upload_vector_once(
                layer.ffn_norm_weight, "Metal FFN norm weight");
            prepared_layer.ffn_gate_weight = upload_matrix_once(
                layer.ffn_gate_weight, "Metal FFN gate weight");
            prepared_layer.ffn_down_weight = upload_matrix_once(
                layer.ffn_down_weight, "Metal FFN down weight");
            prepared_layer.ffn_up_weight = upload_matrix_once(
                layer.ffn_up_weight, "Metal FFN up weight");
            next.layers.push_back(std::move(prepared_layer));
        }
        next.ready = true;
        prepared = std::move(next);
    }
}

void MetalLLM::Impl::allocate_kv_cache(size_t capacity) {
    require_available();
    if (!prepared.ready || prepared.layers.empty()) {
        throw std::runtime_error(
            "Metal weights must be uploaded before allocating KV cache");
    }
    if (capacity == 0) {
        throw std::invalid_argument(
            "Metal KV-cache maximum sequence must be greater than zero");
    }
    checked_uint(capacity, "Metal KV-cache maximum sequence");

    std::vector<KVCacheLayer> next;
    next.reserve(prepared.layers.size());
    @autoreleasepool {
        for (const PreparedLayer& layer : prepared.layers) {
            const size_t kv_dimension = layer.attn_k_weight.rows;
            if (kv_dimension == 0 ||
                kv_dimension != layer.attn_v_weight.rows) {
                throw std::invalid_argument(
                    "Metal KV-cache K/V projection dimensions do not match");
            }
            const size_t elements = checked_elements(
                capacity, kv_dimension, "Metal KV-cache allocation");
            const size_t bytes = checked_bytes(
                elements, "Metal KV-cache allocation");
            KVCacheLayer storage;
            storage.key = [device
                newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
            storage.value = [device
                newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
            if (storage.key == nil || storage.value == nil) {
                throw std::runtime_error(
                    "Metal failed to allocate private KV-cache buffers");
            }
            storage.kv_dimension = kv_dimension;
            next.push_back(storage);
        }
    }
    kv_cache = std::move(next);
    max_sequence = capacity;
    sequence_length = 0;
}

void MetalLLM::Impl::load_model(const std::string& gguf_path,
                                size_t capacity) {
    require_available();
    ModelFile model_file(gguf_path);
    ModelConfig loaded_config = model_file.config();
    if (loaded_config.layer_count == 0 ||
        loaded_config.embedding_size == 0 ||
        loaded_config.feed_forward_size == 0 ||
        loaded_config.attention_head_count == 0 ||
        loaded_config.kv_head_count == 0 ||
        loaded_config.head_size == 0 ||
        loaded_config.vocabulary_size == 0 ||
        loaded_config.context_length == 0) {
        throw std::runtime_error("Metal model metadata is incomplete");
    }
    if (capacity == 0) {
        capacity = loaded_config.context_length;
    }
    if (capacity > loaded_config.context_length) {
        throw std::invalid_argument(
            "Metal maximum sequence exceeds the model context length");
    }

    // CPU tensors below are temporary staging objects. upload_model() copies
    // every value into a persistent MTLBuffer; all staging memory is released
    // when this function returns.
    Matrix token_embedding_weight = model_file.load_token_embedding();
    Vector output_norm_weight = model_file.load_output_norm();
    Matrix output_weight = model_file.load_output_weight();
    std::vector<Layer> layers;
    layers.reserve(loaded_config.layer_count);
    for (size_t layer_index = 0;
         layer_index < loaded_config.layer_count;
         ++layer_index) {
        layers.emplace_back(model_file, layer_index);
    }
    model_file.validate_all_tensors_loaded();

    config = std::move(loaded_config);
    upload_model(
        layers, output_weight, token_embedding_weight, output_norm_weight);
    allocate_kv_cache(capacity);
}

MetalLLM::MetalLLM(const std::string& gguf_path,
                   std::size_t max_sequence,
                   const std::string& shader_path)
    : impl_(std::make_unique<Impl>(shader_path)) {
    impl_->load_model(gguf_path, max_sequence);
}

MetalLLM::~MetalLLM() = default;

MetalLLM::MetalLLM(MetalLLM&&) noexcept = default;

MetalLLM& MetalLLM::operator=(MetalLLM&&) noexcept = default;

bool MetalLLM::available() const noexcept {
    return impl_ != nullptr && impl_->device != nil &&
           impl_->gemm_pipeline != nil && impl_->gevm_pipeline != nil &&
           impl_->embedding_pipeline != nil &&
           impl_->rmsnorm_pipeline != nil && impl_->softmax_pipeline != nil &&
           impl_->swiglu_pipeline != nil && impl_->residual_pipeline != nil &&
           impl_->rope_heads_pipeline != nil &&
           impl_->kv_cache_write_pipeline != nil &&
           impl_->kv_cache_qk_pipeline != nil &&
           impl_->kv_cache_av_pipeline != nil &&
           impl_->argmax_pipeline != nil;
}

const std::string& MetalLLM::device_name() const noexcept {
    static const std::string empty;
    return impl_ == nullptr ? empty : impl_->device_name;
}

void MetalLLM::reset() {
    if (impl_ == nullptr) {
        throw std::runtime_error("Metal model is not initialized");
    }
    impl_->require_available();
    impl_->require_prepared();
    // Old bytes do not need clearing: every attention kernel is bounded by the
    // logical sequence length and prefill overwrites the new prefix from row 0.
    impl_->sequence_length = 0;
}

int32_t MetalLLM::prefill(const std::vector<int32_t>& token_ids) {
    if (impl_ == nullptr) {
        throw std::runtime_error("Metal model is not initialized");
    }
    impl_->require_available();
    impl_->require_prepared();
    if (token_ids.empty()) {
        throw std::invalid_argument(
            "Metal resident prefill requires at least one token id");
    }
    if (impl_->sequence_length != 0) {
        throw std::invalid_argument(
            "Metal prefill requires an empty conversation; call reset first");
    }
    if (token_ids.size() > impl_->max_sequence) {
        throw std::length_error(
            "Metal resident prefill exceeds the allocated sequence capacity");
    }
    for (const int32_t token_id : token_ids) {
        if (token_id < 0 ||
            static_cast<size_t>(token_id) >= impl_->config.vocabulary_size) {
            throw std::out_of_range(
                "Metal prefill token id is outside the vocabulary");
        }
    }

    Profiler::ForwardScope forward_profile(
        impl_->profiler.get(), token_ids.size());
    @autoreleasepool {
        Profiler::Scope resident_scope(
            impl_->profiler.get(), "prefill.gpu_resident",
            resident_prefill_metrics(impl_->config, token_ids.size()));
        Impl::AsyncExecution execution = impl_->begin_async(
            "resident prefill");
        const Impl::DeviceMatrix& embedding_weight =
            impl_->prepared.token_embedding_weight;
        const id<MTLBuffer> token_buffer = impl_->async_token_ids(
            execution, token_ids);
        Impl::DeviceMatrix hidden = impl_->encode_embedding(
            execution, token_buffer, embedding_weight, token_ids.size(),
            impl_->config.vocabulary_size);

        for (size_t layer_index = 0;
             layer_index < impl_->prepared.layers.size();
             ++layer_index) {
            const Impl::PreparedLayer& layer =
                impl_->prepared.layers[layer_index];
            const Impl::KVCacheLayer& storage =
                impl_->kv_cache[layer_index];
            hidden = impl_->encode_prefill_layer(
                execution, hidden, layer, impl_->config.norm_epsilon,
                impl_->config.rotary_dimension, impl_->config.rope_theta,
                impl_->config.head_size,
                storage.key, storage.value, storage.kv_dimension, 0);
        }

        const Impl::DeviceVector& output_norm =
            impl_->prepared.output_norm_weight;
        const Impl::DeviceMatrix final_norm = impl_->encode_rmsnorm_matrix(
            execution, hidden,
            output_norm, impl_->config.norm_epsilon,
            "resident prefill final RMSNorm");
        const size_t last_row_offset = checked_elements(
            final_norm.rows - 1, final_norm.stride,
            "Resident prefill final row offset");
        const Impl::DeviceVector last_hidden = {
            final_norm.buffer, final_norm.cols,
            final_norm.offset_elements + last_row_offset};
        const Impl::DeviceMatrix& output_weight =
            impl_->prepared.output_weight;
        const Impl::DeviceVector logits = impl_->encode_vector_product(
            execution, output_weight, last_hidden, true, nullptr, 1.0f,
            "resident prefill LM head");
        const id<MTLBuffer> token_output = impl_->encode_argmax(
            execution, logits.buffer, logits.length,
            "resident prefill argmax");

        impl_->finish_async(execution, "Metal resident prefill");
        const uint32_t token = *static_cast<const uint32_t*>(
            [token_output contents]);
        // Read the token before making the argmax buffer available to another
        // execution. The command is complete, but this CPU load still needs the
        // buffer to remain exclusively owned by this execution.
        impl_->recycle_temporary_buffers(execution);
        if (token > static_cast<uint32_t>(
                        std::numeric_limits<int32_t>::max())) {
            throw std::overflow_error(
                "Metal resident prefill token id does not fit in int32_t");
        }
        impl_->sequence_length = token_ids.size();
        return static_cast<int32_t>(token);
    }
}

int32_t MetalLLM::decode(int32_t token_id) {
    if (impl_ == nullptr) {
        throw std::runtime_error("Metal model is not initialized");
    }
    impl_->require_available();
    impl_->require_prepared();
    if (impl_->sequence_length == 0) {
        throw std::invalid_argument(
            "Metal decode requires a completed prefill");
    }
    if (token_id < 0 ||
        static_cast<size_t>(token_id) >= impl_->config.vocabulary_size) {
        throw std::out_of_range(
            "Metal resident decode token id is outside the vocabulary");
    }
    const size_t position = impl_->sequence_length;
    if (position >= impl_->max_sequence) {
        throw std::length_error(
            "Metal resident decode exceeds the allocated sequence capacity");
    }

    Profiler::ForwardScope forward_profile(
        impl_->profiler.get(), position + 1);
    @autoreleasepool {
        Profiler::Scope resident_scope(
            impl_->profiler.get(), "decode.gpu_resident",
            resident_decode_metrics(impl_->config, position + 1));
        Impl::AsyncExecution execution = impl_->begin_async(
            "resident decode");
        const Impl::DeviceMatrix& embedding_weight =
            impl_->prepared.token_embedding_weight;
        const std::vector<int32_t> one_token{token_id};
        const id<MTLBuffer> token_buffer = impl_->async_token_ids(
            execution, one_token);
        const Impl::DeviceMatrix embedded = impl_->encode_embedding(
            execution, token_buffer, embedding_weight, 1,
            impl_->config.vocabulary_size);
        Impl::DeviceVector hidden = {
            embedded.buffer, embedded.cols, embedded.offset_elements};

        for (size_t layer_index = 0;
             layer_index < impl_->prepared.layers.size();
             ++layer_index) {
            const Impl::PreparedLayer& layer =
                impl_->prepared.layers[layer_index];
            const Impl::KVCacheLayer& storage =
                impl_->kv_cache[layer_index];
            hidden = impl_->encode_decode_layer(
                execution, hidden, layer, impl_->config.norm_epsilon,
                impl_->config.rotary_dimension, impl_->config.rope_theta,
                impl_->config.head_size,
                storage.key, storage.value, storage.kv_dimension, position);
        }

        const Impl::DeviceVector& output_norm =
            impl_->prepared.output_norm_weight;
        const Impl::DeviceVector final_norm = impl_->encode_rmsnorm_vector(
            execution, hidden, output_norm, impl_->config.norm_epsilon,
            "resident decode final RMSNorm");
        const Impl::DeviceMatrix& output_weight =
            impl_->prepared.output_weight;
        const Impl::DeviceVector logits = impl_->encode_vector_product(
            execution, output_weight, final_norm, true, nullptr, 1.0f,
            "resident decode LM head");
        const id<MTLBuffer> token_output = impl_->encode_argmax(
            execution, logits.buffer, logits.length,
            "resident decode argmax");

        impl_->finish_async(execution, "Metal resident decode");
        const uint32_t token = *static_cast<const uint32_t*>(
            [token_output contents]);
        // See the corresponding prefill path: consume the result before
        // returning temporary storage to the shared pool.
        impl_->recycle_temporary_buffers(execution);
        if (token > static_cast<uint32_t>(
                        std::numeric_limits<int32_t>::max())) {
            throw std::overflow_error(
                "Metal resident decode token id does not fit in int32_t");
        }
        impl_->sequence_length = position + 1;
        return static_cast<int32_t>(token);
    }
}

void MetalLLM::enable_profiling(const std::string& csv_path) {
    if (impl_ == nullptr) {
        throw std::runtime_error("Metal model is not initialized");
    }
    impl_->profiler = std::make_unique<Profiler>(csv_path);
}

void MetalLLM::disable_profiling() {
    if (impl_ != nullptr) {
        impl_->profiler.reset();
    }
}

const ModelConfig& MetalLLM::config() const {
    if (impl_ == nullptr) {
        throw std::runtime_error("Metal model is not initialized");
    }
    return impl_->config;
}

std::size_t MetalLLM::position() const noexcept {
    return impl_ == nullptr ? 0 : impl_->sequence_length;
}

std::size_t MetalLLM::max_sequence() const noexcept {
    return impl_ == nullptr ? 0 : impl_->max_sequence;
}

bool MetalLLM::uses_gpu() const noexcept {
    return available();
}

} // namespace llm

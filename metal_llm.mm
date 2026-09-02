#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_llm.h"

#include "profiler.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mach/mach_time.h>
#include <mach-o/dyld.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llm {
namespace {

constexpr uint32_t kTileSize = 16;
constexpr uint32_t kDefaultGemvThreads = 256;
constexpr size_t kCounterSamplesPerBuffer = 4096;
constexpr size_t kNoProfileIndex = std::numeric_limits<size_t>::max();

uint64_t monotonic_time_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

const mach_timebase_info_data_t& mach_timebase() {
    static const mach_timebase_info_data_t value = [] {
        mach_timebase_info_data_t info{};
        mach_timebase_info(&info);
        return info;
    }();
    return value;
}

double mach_ticks_to_ns(uint64_t ticks) {
    const mach_timebase_info_data_t& info = mach_timebase();
    return static_cast<double>(ticks) * static_cast<double>(info.numer) /
           static_cast<double>(info.denom);
}

double mach_host_time_seconds() {
    return mach_ticks_to_ns(mach_absolute_time()) / 1.0e9;
}

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

// Quantized projection weights always keep GGUF's native [output, input]
// layout.  Strides are therefore expressed in bytes rather than elements.
struct MetalQuantizedProductParamsHost {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t activation_stride = 0;
    uint32_t weight_row_bytes = 0;
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
    uint32_t weight_row_bytes = 0;
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
static_assert(sizeof(MetalQuantizedProductParamsHost) == 28,
              "Metal quantized product parameter layout changed");
static_assert(sizeof(MetalElementwiseParamsHost) == 8,
              "Metal elementwise parameter layout changed");
static_assert(sizeof(MetalRmsNormParamsHost) == 12,
              "Metal RMSNorm parameter layout changed");
static_assert(sizeof(MetalRopeHeadsParamsHost) == 24,
              "Metal head-wise RoPE parameter layout changed");
static_assert(sizeof(MetalEmbeddingParamsHost) == 16,
              "Metal embedding parameter layout changed");
static_assert(sizeof(MetalKVCacheWriteParamsHost) == 16,
              "Metal KV-cache write parameter layout changed");
static_assert(sizeof(MetalKVCacheQKParamsHost) == 32,
              "Metal KV-cache QK parameter layout changed");
static_assert(sizeof(MetalKVCacheAVParamsHost) == 32,
              "Metal KV-cache AV parameter layout changed");
static_assert(sizeof(MetalArgmaxParamsHost) == 4,
              "Metal argmax parameter layout changed");

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

size_t checked_add_size(size_t left, size_t right, const char* name) {
    if (right > std::numeric_limits<size_t>::max() - left) {
        throw std::length_error(std::string(name) + " size overflows");
    }
    return left + right;
}

size_t align_up_size(size_t value, size_t alignment, const char* name) {
    if (alignment == 0) {
        throw std::invalid_argument(std::string(name) + " alignment is zero");
    }
    const size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return checked_add_size(value, alignment - remainder, name);
}

uint32_t checked_uint(size_t value, const char* name) {
    if (value > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::length_error(std::string(name) + " does not fit in Metal uint");
    }
    return static_cast<uint32_t>(value);
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

ProfileMetrics resident_prefill_metrics(const MetalModelConfig& model,
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

ProfileMetrics resident_decode_metrics(const MetalModelConfig& model,
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
    enum class DeviceDataType : uint8_t {
        F32,
    };

    // Device-side tensor descriptors used by the asynchronous graph path.
    // Activations are currently FP32. Keeping their type in the descriptor
    // and arena plan makes offsets byte-correct when more activation formats
    // are introduced without changing operator boundaries.
    struct DeviceMatrix {
        id<MTLBuffer> buffer = nil;
        size_t rows = 0;
        size_t cols = 0;
        size_t stride = 0;
        size_t offset_elements = 0;
        DeviceDataType type = DeviceDataType::F32;
    };

    struct DeviceVector {
        id<MTLBuffer> buffer = nil;
        size_t length = 0;
        size_t offset_elements = 0;
        DeviceDataType type = DeviceDataType::F32;
    };

    // Persistent model matrix. row_bytes is the physical GGUF row stride;
    // unlike DeviceMatrix it is not assumed to contain FP32 elements.
    struct QuantizedMatrix {
        id<MTLBuffer> buffer = nil;
        MetalGgmlType type = MetalGgmlType::F32;
        size_t rows = 0;
        size_t cols = 0;
        size_t row_bytes = 0;
        size_t offset_bytes = 0;
    };

    // Each execution phase uses one statically planned region map inside the
    // active arena. Prefill plans rows and scores from the actual prompt
    // length; decode plans one activation row and a [1, max_sequence] score
    // vector. All values stored in these slots remain FP32.
    enum class ActivationSlot : uint8_t {
        HiddenA,
        HiddenB,
        Norm,
        Query,
        RotatedQuery,
        Key,
        RotatedKey,
        Value,
        AttentionOutput,
        AttentionScores,
        Projection,
        Gate,
        Up,
        Logits,
        Count,
    };

    struct ArenaRegion {
        size_t offset_bytes = 0;
        size_t capacity_bytes = 0;
        DeviceDataType type = DeviceDataType::F32;
    };

    struct ActivationArenaPlan {
        std::array<ArenaRegion,
                   static_cast<size_t>(ActivationSlot::Count)> regions{};
        size_t bytes = 0;
    };

    enum class ActivationArenaPhase : uint8_t {
        None,
        Prefill,
        Decode,
    };

    // These descriptors own the model buffers for the whole lifetime of the
    // backend. They are built once during construction and are the only model
    // weights used by prefill/decode.
    struct PreparedLayer {
        DeviceVector attn_norm_weight;
        QuantizedMatrix attn_q_weight;
        DeviceVector attn_q_bias;
        QuantizedMatrix attn_k_weight;
        DeviceVector attn_k_bias;
        QuantizedMatrix attn_v_weight;
        DeviceVector attn_v_bias;
        QuantizedMatrix attn_output_weight;
        DeviceVector ffn_norm_weight;
        QuantizedMatrix ffn_gate_weight;
        QuantizedMatrix ffn_down_weight;
        QuantizedMatrix ffn_up_weight;
    };

    struct PreparedModel {
        QuantizedMatrix token_embedding_weight;
        DeviceVector output_norm_weight;
        QuantizedMatrix output_weight;
        std::vector<PreparedLayer> layers;
        size_t weight_bytes = 0;
        bool ready = false;
    };

    struct KVCacheLayer {
        id<MTLBuffer> key = nil;
        id<MTLBuffer> value = nil;
        size_t kv_dimension = 0;
    };

    struct KernelWorkload {
        ProfileMetrics metrics;
        uint64_t weight_bytes = 0;
        uint64_t shader_read_bytes = 0;
        size_t m = 0;
        size_t n = 0;
        size_t k = 0;
        std::string value_type;
    };

    struct DispatchGeometry {
        std::string type;
        MTLSize grid = MTLSizeMake(0, 0, 0);
        MTLSize threads = MTLSizeMake(0, 0, 0);
        uint64_t threadgroups = 0;
        uint64_t dispatched_threads = 0;
    };

    struct CallbackTimes {
        std::atomic<uint64_t> scheduled_ns{0};
        std::atomic<uint64_t> completed_ns{0};
    };

    enum class CounterSamplingMode : uint8_t {
        None,
        DispatchBoundary,
        StageBoundary,
    };

    struct AsyncExecution {
        id<MTLCommandBuffer> command = nil;

        // These are only CPU-visible staging/results. All tensor activations
        // are views into activation_arena and are not stored here.
        std::vector<id<MTLBuffer>> io_buffers;
        std::string phase;
        size_t sequence_tokens = 0;
        size_t command_index = 0;
        size_t current_layer = kNoProfileIndex;
        size_t current_head = kNoProfileIndex;
        uint64_t begin_ns = 0;
        uint64_t encode_begin_ns = 0;
        uint64_t command_create_ns = 0;
        uint64_t kernel_count = 0;
        uint64_t threadgroup_count = 0;
        uint64_t dispatched_threads = 0;
        uint64_t single_threadgroup_kernels = 0;
        std::shared_ptr<CallbackTimes> callback_times;

        std::vector<id<MTLCounterSampleBuffer>> counter_sample_buffers;
        size_t counter_sample_capacity = 0;
        NSUInteger next_sample = 0;
        std::string counter_status;
        std::vector<MetalKernelProfileRecord> kernels;
        size_t active_kernel = kNoProfileIndex;
        uint64_t active_kernel_cpu_start_ns = 0;
    };

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> gemm_pipeline = nil;
    id<MTLComputePipelineState> gevm_pipeline = nil;
    id<MTLComputePipelineState> embedding_pipeline = nil;
    id<MTLComputePipelineState> gemm_q5_0_pipeline = nil;
    id<MTLComputePipelineState> gemm_q8_0_pipeline = nil;
    id<MTLComputePipelineState> gemm_q4_k_pipeline = nil;
    id<MTLComputePipelineState> gemm_q6_k_pipeline = nil;
    id<MTLComputePipelineState> gevm_q5_0_pipeline = nil;
    id<MTLComputePipelineState> gevm_q8_0_pipeline = nil;
    id<MTLComputePipelineState> gevm_q4_k_pipeline = nil;
    id<MTLComputePipelineState> gevm_q6_k_pipeline = nil;
    id<MTLComputePipelineState> embedding_q5_0_pipeline = nil;
    id<MTLComputePipelineState> embedding_q8_0_pipeline = nil;
    id<MTLComputePipelineState> embedding_q4_k_pipeline = nil;
    id<MTLComputePipelineState> embedding_q6_k_pipeline = nil;
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
    MetalModelConfig config;
    PreparedModel prepared;
    std::vector<KVCacheLayer> kv_cache;
    id<MTLBuffer> activation_arena = nil;
    ActivationArenaPlan activation_plan;
    ActivationArenaPhase activation_phase = ActivationArenaPhase::None;
    size_t turn_peak_activation_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t max_sequence = 0;
    size_t sequence_length = 0;
    std::unique_ptr<Profiler> profiler;
    std::unique_ptr<MetalProfiler> metal_profiler;
    id<MTLCounterSet> timestamp_counter_set = nil;
    CounterSamplingMode counter_sampling_mode = CounterSamplingMode::None;
    std::string timestamp_counter_status = "not requested";
    static constexpr size_t kArenaAlignment = 256;

    void require_available() const {
        if (device == nil || gemm_pipeline == nil || gevm_pipeline == nil ||
            embedding_pipeline == nil || gemm_q5_0_pipeline == nil ||
            gemm_q8_0_pipeline == nil || gemm_q4_k_pipeline == nil ||
            gemm_q6_k_pipeline == nil || gevm_q5_0_pipeline == nil ||
            gevm_q8_0_pipeline == nil || gevm_q4_k_pipeline == nil ||
            gevm_q6_k_pipeline == nil || embedding_q5_0_pipeline == nil ||
            embedding_q8_0_pipeline == nil ||
            embedding_q4_k_pipeline == nil ||
            embedding_q6_k_pipeline == nil || rmsnorm_pipeline == nil ||
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
            gemm_q5_0_pipeline = make_pipeline(
                device, library, "metal_gemm_q5_0_f32",
                "create Q5_0 GEMM pipeline");
            gemm_q8_0_pipeline = make_pipeline(
                device, library, "metal_gemm_q8_0_f32",
                "create Q8_0 GEMM pipeline");
            gemm_q4_k_pipeline = make_pipeline(
                device, library, "metal_gemm_q4_k_f32",
                "create Q4_K GEMM pipeline");
            gemm_q6_k_pipeline = make_pipeline(
                device, library, "metal_gemm_q6_k_f32",
                "create Q6_K GEMM pipeline");
            gevm_q5_0_pipeline = make_pipeline(
                device, library, "metal_gevm_q5_0_f32",
                "create Q5_0 GEVM pipeline");
            gevm_q8_0_pipeline = make_pipeline(
                device, library, "metal_gevm_q8_0_f32",
                "create Q8_0 GEVM pipeline");
            gevm_q4_k_pipeline = make_pipeline(
                device, library, "metal_gevm_q4_k_f32",
                "create Q4_K GEVM pipeline");
            gevm_q6_k_pipeline = make_pipeline(
                device, library, "metal_gevm_q6_k_f32",
                "create Q6_K GEVM pipeline");
            embedding_q5_0_pipeline = make_pipeline(
                device, library, "metal_embedding_q5_0_f32",
                "create Q5_0 embedding pipeline");
            embedding_q8_0_pipeline = make_pipeline(
                device, library, "metal_embedding_q8_0_f32",
                "create Q8_0 embedding pipeline");
            embedding_q4_k_pipeline = make_pipeline(
                device, library, "metal_embedding_q4_k_f32",
                "create Q4_K embedding pipeline");
            embedding_q6_k_pipeline = make_pipeline(
                device, library, "metal_embedding_q6_k_f32",
                "create Q6_K embedding pipeline");
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

    QuantizedMatrix upload_matrix(const RawTensor& tensor,
                                  const char* name) const {
        const size_t expected_row_bytes = metal_ggml_row_bytes(
            tensor.type, tensor.cols);
        const size_t expected_bytes = checked_bytes_for(
            tensor.rows, expected_row_bytes, name);
        if (tensor.rows == 0 || tensor.cols == 0 ||
            tensor.row_bytes != expected_row_bytes ||
            tensor.data.size() != expected_bytes) {
            throw std::invalid_argument(std::string(name) +
                                        " raw tensor storage is invalid");
        }
        switch (tensor.type) {
        case MetalGgmlType::F32:
        case MetalGgmlType::Q5_0:
        case MetalGgmlType::Q8_0:
        case MetalGgmlType::Q4_K:
        case MetalGgmlType::Q6_K:
            break;
        default:
            throw std::runtime_error(
                std::string(name) + " uses unsupported Metal matrix type " +
                metal_ggml_type_name(tensor.type));
        }
        id<MTLBuffer> buffer = make_buffer(
            tensor.data.data(), tensor.data.size());
        return {buffer, tensor.type, tensor.rows, tensor.cols,
                tensor.row_bytes, 0};
    }

    DeviceVector upload_vector(const RawTensor& tensor,
                               const char* name) const {
        if (tensor.type != MetalGgmlType::F32 || tensor.rows != 1 ||
            tensor.row_bytes != checked_bytes(tensor.cols, name) ||
            tensor.data.size() != tensor.row_bytes) {
            throw std::invalid_argument(
                std::string(name) + " must be one contiguous F32 GGUF row");
        }
        id<MTLBuffer> buffer = make_buffer(
            tensor.data.data(), tensor.data.size());
        return {buffer, tensor.cols, 0, DeviceDataType::F32};
    }

    void require_prepared() const {
        if (!prepared.ready || config.layer_count == 0 ||
            prepared.layers.size() != config.layer_count ||
            kv_cache.size() != config.layer_count ||
            max_sequence == 0) {
            throw std::runtime_error(
                "Metal model weights or KV cache are not initialized");
        }
    }

    static size_t slot_index(ActivationSlot slot) {
        return static_cast<size_t>(slot);
    }

    static size_t device_type_size(DeviceDataType type) {
        switch (type) {
        case DeviceDataType::F32:
            return sizeof(float);
        }
        throw std::runtime_error("unsupported Metal activation data type");
    }

    ActivationArenaPlan make_activation_plan(size_t row_capacity,
                                             size_t score_rows,
                                             size_t score_cols) const {
        if (row_capacity == 0 || score_rows == 0 || score_cols == 0) {
            throw std::invalid_argument(
                "Metal activation arena dimensions must be greater than zero");
        }

        const size_t hidden = config.embedding_size;
        const size_t query = config.attention_head_count * config.head_size;
        const size_t key_value = config.kv_head_count * config.head_size;
        const size_t feed_forward = config.feed_forward_size;
        const size_t vocabulary = config.vocabulary_size;
        if (hidden == 0 || query == 0 || key_value == 0 || feed_forward == 0 ||
            vocabulary == 0) {
            throw std::runtime_error(
                "Metal activation arena cannot be planned for incomplete model metadata");
        }

        ActivationArenaPlan plan;
        const auto add_region = [&](ActivationSlot slot,
                                    size_t elements,
                                    const char* name,
                                    DeviceDataType type = DeviceDataType::F32) {
            const size_t bytes = checked_bytes_for(
                elements, device_type_size(type), name);
            const size_t offset = align_up_size(
                plan.bytes, kArenaAlignment, name);
            plan.regions[slot_index(slot)] = {offset, bytes, type};
            plan.bytes = checked_add_size(offset, bytes, name);
        };

        // Hidden/norm/projection-sized slots are shared by prefill and decode.
        // Norm is intentionally reused for attention norm, FFN norm, and the
        // final norm because those lifetimes never overlap.
        add_region(ActivationSlot::HiddenA,
                   checked_elements(row_capacity, hidden,
                                    "Metal arena hidden A"),
                   "Metal arena hidden A");
        add_region(ActivationSlot::HiddenB,
                   checked_elements(row_capacity, hidden,
                                    "Metal arena hidden B"),
                   "Metal arena hidden B");
        add_region(ActivationSlot::Norm,
                   checked_elements(row_capacity, hidden, "Metal arena norm"),
                   "Metal arena norm");
        add_region(ActivationSlot::Query,
                   checked_elements(row_capacity, query, "Metal arena query"),
                   "Metal arena query");
        add_region(ActivationSlot::RotatedQuery,
                   checked_elements(row_capacity, query,
                                    "Metal arena rotated query"),
                   "Metal arena rotated query");
        add_region(ActivationSlot::Key,
                   checked_elements(row_capacity, key_value,
                                    "Metal arena key"),
                   "Metal arena key");
        add_region(ActivationSlot::RotatedKey,
                   checked_elements(row_capacity, key_value,
                                    "Metal arena rotated key"),
                   "Metal arena rotated key");
        add_region(ActivationSlot::Value,
                   checked_elements(row_capacity, key_value,
                                    "Metal arena value"),
                   "Metal arena value");
        add_region(ActivationSlot::AttentionOutput,
                   checked_elements(row_capacity, query,
                                    "Metal arena attention output"),
                   "Metal arena attention output");

        // One score/probability region is reused by every head and layer.
        // Prefill supplies [prompt_tokens, prompt_tokens], while decode supplies
        // [1, max_sequence]. Softmax overwrites this region in place.
        add_region(ActivationSlot::AttentionScores,
                   checked_elements(score_rows, score_cols,
                                    "Metal arena attention scores"),
                   "Metal arena attention scores");
        add_region(ActivationSlot::Projection,
                   checked_elements(row_capacity, hidden,
                                    "Metal arena projection"),
                   "Metal arena projection");
        add_region(ActivationSlot::Gate,
                   checked_elements(row_capacity, feed_forward,
                                    "Metal arena gate"),
                   "Metal arena gate");
        add_region(ActivationSlot::Up,
                   checked_elements(row_capacity, feed_forward,
                                    "Metal arena up"),
                   "Metal arena up");
        add_region(ActivationSlot::Logits, vocabulary,
                   "Metal arena logits");
        plan.bytes = align_up_size(plan.bytes, kArenaAlignment,
                                   "Metal activation arena");
        return plan;
    }

    ActivationArenaPlan make_prefill_activation_plan(
        size_t prompt_tokens) const {
        return make_activation_plan(
            prompt_tokens, prompt_tokens, prompt_tokens);
    }

    ActivationArenaPlan make_decode_activation_plan() const {
        if (max_sequence == 0) {
            throw std::runtime_error(
                "Metal decode arena requires a maximum sequence length");
        }
        return make_activation_plan(1, 1, max_sequence);
    }

    const ArenaRegion& arena_region(ActivationSlot slot) const {
        return activation_plan.regions[slot_index(slot)];
    }

    DeviceMatrix arena_matrix(ActivationSlot slot,
                              size_t rows,
                              size_t cols,
                              const char* operation) const {
        if (activation_arena == nil) {
            throw std::runtime_error(
                "Metal activation arena is not initialized");
        }
        const ArenaRegion& region = arena_region(slot);
        if (region.type != DeviceDataType::F32) {
            throw std::runtime_error(std::string(operation) +
                                     " requires an FP32 arena slot");
        }
        const size_t element_size = device_type_size(region.type);
        const size_t bytes = checked_bytes_for(
            checked_elements(rows, cols, operation), element_size, operation);
        if (bytes > region.capacity_bytes) {
            throw std::length_error(std::string(operation) +
                                    " exceeds its activation arena slot");
        }
        return {activation_arena, rows, cols, cols,
                region.offset_bytes / element_size, region.type};
    }

    DeviceVector arena_vector(ActivationSlot slot,
                              size_t length,
                              const char* operation) const {
        if (activation_arena == nil) {
            throw std::runtime_error(
                "Metal activation arena is not initialized");
        }
        const ArenaRegion& region = arena_region(slot);
        if (region.type != DeviceDataType::F32) {
            throw std::runtime_error(std::string(operation) +
                                     " requires an FP32 arena slot");
        }
        const size_t element_size = device_type_size(region.type);
        const size_t bytes = checked_bytes_for(
            length, element_size, operation);
        if (bytes > region.capacity_bytes) {
            throw std::length_error(std::string(operation) +
                                    " exceeds its activation arena slot");
        }
        return {activation_arena, length,
                region.offset_bytes / element_size, region.type};
    }

    KernelWorkload product_workload(const QuantizedMatrix& matrix,
                                    size_t m,
                                    size_t n,
                                    size_t k,
                                    bool has_bias) const {
        KernelWorkload workload;
        workload.m = m;
        workload.n = n;
        workload.k = k;
        workload.value_type = metal_ggml_type_name(matrix.type);

        const size_t outputs = checked_elements(
            m, n, "Metal profile product outputs");
        const size_t multiply_accumulates = checked_elements(
            outputs, k, "Metal profile product operations");
        workload.metrics.flops = static_cast<uint64_t>(checked_bytes_for(
            multiply_accumulates, 2, "Metal profile product FLOPs"));
        if (has_bias) {
            workload.metrics.flops += static_cast<uint64_t>(outputs);
        }

        const size_t activation_bytes = checked_bytes(
            checked_elements(m, k, "Metal profile activation elements"),
            "Metal profile activation bytes");
        const size_t weight_bytes = checked_bytes_for(
            n, matrix.row_bytes, "Metal profile weight bytes");
        const size_t bias_bytes = has_bias
            ? checked_bytes(n, "Metal profile bias bytes") : 0;
        const size_t output_bytes = checked_bytes(
            outputs, "Metal profile product output bytes");
        workload.metrics.read_bytes = static_cast<uint64_t>(checked_add_size(
            checked_add_size(activation_bytes, weight_bytes,
                             "Metal profile product reads"),
            bias_bytes, "Metal profile product reads"));
        workload.metrics.write_bytes = static_cast<uint64_t>(output_bytes);
        workload.metrics.temporary_bytes = static_cast<uint64_t>(output_bytes);
        workload.weight_bytes = static_cast<uint64_t>(weight_bytes);

        // This is the load demand expressed by the current one-output-per-
        // thread shader before accounting for GPU caches. It intentionally
        // differs from minimum tensor traffic: every output thread requests
        // the activation vector, and prefill requests each weight row for
        // every input row.
        const size_t shader_weight_bytes = checked_bytes_for(
            m, weight_bytes, "Metal profile shader weight reads");
        const size_t shader_activation_bytes = checked_bytes(
            multiply_accumulates,
            "Metal profile shader activation reads");
        const size_t shader_bias_bytes = has_bias
            ? checked_bytes(outputs, "Metal profile shader bias reads") : 0;
        workload.shader_read_bytes = static_cast<uint64_t>(checked_add_size(
            checked_add_size(shader_weight_bytes, shader_activation_bytes,
                             "Metal profile shader reads"),
            shader_bias_bytes, "Metal profile shader reads"));
        return workload;
    }

    KernelWorkload elementwise_workload(size_t rows,
                                        size_t cols,
                                        uint64_t flops_per_element,
                                        uint64_t read_arrays,
                                        uint64_t write_arrays) const {
        KernelWorkload workload;
        workload.m = rows;
        workload.n = cols;
        workload.value_type = "F32";
        const size_t elements = checked_elements(
            rows, cols, "Metal profile elementwise elements");
        workload.metrics = profile_elementwise_metrics(
            elements, flops_per_element, read_arrays, write_arrays);
        workload.shader_read_bytes = workload.metrics.read_bytes;
        return workload;
    }

    size_t expected_forward_kernel_count() const {
        const size_t heads = config.attention_head_count;
        const size_t per_layer = checked_add_size(
            15, checked_elements(3, heads, "Metal profile head kernels"),
            "Metal profile layer kernels");
        return checked_add_size(
            4, checked_elements(config.layer_count, per_layer,
                                "Metal profile forward kernels"),
            "Metal profile forward kernels");
    }

    std::string pipeline_name(
        id<MTLComputePipelineState> pipeline) const {
        if (pipeline == gemm_pipeline) return "metal_gemm_f32";
        if (pipeline == gevm_pipeline) return "metal_gevm_f32";
        if (pipeline == embedding_pipeline) return "metal_embedding_f32";
        if (pipeline == gemm_q5_0_pipeline) return "metal_gemm_q5_0_f32";
        if (pipeline == gemm_q8_0_pipeline) return "metal_gemm_q8_0_f32";
        if (pipeline == gemm_q4_k_pipeline) return "metal_gemm_q4_k_f32";
        if (pipeline == gemm_q6_k_pipeline) return "metal_gemm_q6_k_f32";
        if (pipeline == gevm_q5_0_pipeline) return "metal_gevm_q5_0_f32";
        if (pipeline == gevm_q8_0_pipeline) return "metal_gevm_q8_0_f32";
        if (pipeline == gevm_q4_k_pipeline) return "metal_gevm_q4_k_f32";
        if (pipeline == gevm_q6_k_pipeline) return "metal_gevm_q6_k_f32";
        if (pipeline == embedding_q5_0_pipeline) {
            return "metal_embedding_q5_0_f32";
        }
        if (pipeline == embedding_q8_0_pipeline) {
            return "metal_embedding_q8_0_f32";
        }
        if (pipeline == embedding_q4_k_pipeline) {
            return "metal_embedding_q4_k_f32";
        }
        if (pipeline == embedding_q6_k_pipeline) {
            return "metal_embedding_q6_k_f32";
        }
        if (pipeline == rmsnorm_pipeline) return "metal_rmsnorm_f32";
        if (pipeline == softmax_pipeline) return "metal_softmax_f32";
        if (pipeline == swiglu_pipeline) return "metal_swiglu_f32";
        if (pipeline == residual_pipeline) return "metal_residual_f32";
        if (pipeline == rope_heads_pipeline) return "metal_rope_heads_f32";
        if (pipeline == kv_cache_write_pipeline) {
            return "metal_kv_cache_write_f32";
        }
        if (pipeline == kv_cache_qk_pipeline) return "metal_kv_cache_qk_f32";
        if (pipeline == kv_cache_av_pipeline) return "metal_kv_cache_av_f32";
        if (pipeline == argmax_pipeline) return "metal_argmax_f32";
        return "unknown";
    }

    NSUInteger flat_threads(id<MTLComputePipelineState> pipeline) const {
        const NSUInteger maximum =
            [pipeline maxTotalThreadsPerThreadgroup];
        return std::min<NSUInteger>(
            kDefaultGemvThreads, std::max<NSUInteger>(1, maximum));
    }

    DispatchGeometry dispatch_threadgroups(
        id<MTLComputeCommandEncoder> encoder,
        MTLSize threadgroups,
        MTLSize threads) const {
        [encoder dispatchThreadgroups:threadgroups
                 threadsPerThreadgroup:threads];
        DispatchGeometry geometry;
        geometry.type = "threadgroups";
        geometry.grid = threadgroups;
        geometry.threads = threads;
        geometry.threadgroups = static_cast<uint64_t>(threadgroups.width) *
            threadgroups.height * threadgroups.depth;
        geometry.dispatched_threads = geometry.threadgroups *
            static_cast<uint64_t>(threads.width) * threads.height *
            threads.depth;
        return geometry;
    }

    DispatchGeometry dispatch_threads(
        id<MTLComputeCommandEncoder> encoder,
        MTLSize grid,
        MTLSize threads) const {
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
        DispatchGeometry geometry;
        geometry.type = "threads";
        geometry.grid = grid;
        geometry.threads = threads;
        const uint64_t total_threads = static_cast<uint64_t>(grid.width) *
            grid.height * grid.depth;
        const uint64_t threads_per_group =
            static_cast<uint64_t>(threads.width) * threads.height *
            threads.depth;
        geometry.threadgroups = threads_per_group == 0 ? 0 :
            (total_threads + threads_per_group - 1) / threads_per_group;
        geometry.dispatched_threads = total_threads;
        return geometry;
    }

    DispatchGeometry dispatch_flat(id<MTLComputeCommandEncoder> encoder,
                                   id<MTLComputePipelineState> pipeline,
                                   size_t element_count) const {
        const NSUInteger threads = flat_threads(pipeline);
        const NSUInteger count = static_cast<NSUInteger>(element_count);
        const MTLSize threadgroups = MTLSizeMake(
            (count + threads - 1) / threads, 1, 1);
        return dispatch_threadgroups(
            encoder, threadgroups, MTLSizeMake(threads, 1, 1));
    }

    DispatchGeometry dispatch_rows(id<MTLComputeCommandEncoder> encoder,
                                   size_t row_count) const {
        return dispatch_threadgroups(
            encoder,
            MTLSizeMake(static_cast<NSUInteger>(row_count), 1, 1),
            MTLSizeMake(kDefaultGemvThreads, 1, 1));
    }

    AsyncExecution begin_async(const char* phase,
                               size_t sequence_tokens) const {
        // The following encode_* calls only record work into this command
        // buffer. They return buffer descriptors immediately; no kernel runs
        // and no CPU wait occurs until finish_async() commits the full graph.
        AsyncExecution execution;
        execution.phase = phase;
        execution.sequence_tokens = sequence_tokens;
        execution.begin_ns = monotonic_time_ns();
        const uint64_t create_begin_ns = execution.begin_ns;
        execution.command = [queue commandBuffer];
        execution.command_create_ns = monotonic_time_ns() - create_begin_ns;
        if (execution.command == nil) {
            throw std::runtime_error(std::string(
                "Metal failed to create ") + phase + " command buffer");
        }
        NSString* command_label = [NSString stringWithUTF8String:phase];
        if (command_label != nil) {
            [execution.command setLabel:command_label];
        }
        execution.encode_begin_ns = monotonic_time_ns();

        if (metal_profiler != nullptr) {
            execution.command_index =
                metal_profiler->acquire_command_index();
        }
        const bool detailed = metal_profiler != nullptr &&
            metal_profiler->detailed_kernel_timestamps();
        if (detailed) {
            const size_t kernel_capacity = expected_forward_kernel_count();
            execution.kernels.reserve(kernel_capacity);
            execution.counter_status = timestamp_counter_status;
            if (timestamp_counter_set != nil) {
                const size_t sample_capacity = checked_elements(
                    kernel_capacity, 2, "Metal counter sample capacity");
                const size_t buffer_count =
                    (sample_capacity + kCounterSamplesPerBuffer - 1) /
                    kCounterSamplesPerBuffer;
                execution.counter_sample_buffers.reserve(buffer_count);
                size_t remaining = sample_capacity;
                for (size_t buffer_index = 0;
                     buffer_index < buffer_count; ++buffer_index) {
                    const size_t buffer_samples = std::min(
                        remaining, kCounterSamplesPerBuffer);
                    MTLCounterSampleBufferDescriptor* descriptor =
                        [[MTLCounterSampleBufferDescriptor alloc] init];
                    descriptor.counterSet = timestamp_counter_set;
                    descriptor.storageMode = MTLStorageModeShared;
                    descriptor.sampleCount = static_cast<NSUInteger>(
                        buffer_samples);
                    descriptor.label = [NSString stringWithFormat:
                        @"MyLLM kernel timestamps %zu", buffer_index];
                    NSError* error = nil;
                    id<MTLCounterSampleBuffer> sample_buffer =
                        [device newCounterSampleBufferWithDescriptor:descriptor
                                                               error:&error];
                    if (sample_buffer == nil) {
                        execution.counter_status =
                            "sample buffer " +
                            std::to_string(buffer_index) +
                            " unavailable: " + ns_error_description(error);
                        execution.counter_sample_buffers.clear();
                        break;
                    }
                    execution.counter_sample_buffers.push_back(sample_buffer);
                    execution.counter_sample_capacity += buffer_samples;
                    remaining -= buffer_samples;
                }
                if (execution.counter_sample_capacity == sample_capacity) {
                    execution.counter_status =
                        counter_sampling_mode ==
                                CounterSamplingMode::StageBoundary
                            ? "active (stage-boundary sampling)"
                            : "active (dispatch-boundary barrier sampling)";
                }
            }
        }
        return execution;
    }

    id<MTLBuffer> async_buffer(AsyncExecution& execution,
                               const void* data,
                               size_t bytes,
                               const char* operation) const {
        if (bytes == 0) {
            throw std::invalid_argument(std::string(operation) +
                                        " buffer cannot be empty");
        }
        id<MTLBuffer> buffer = make_buffer(nullptr, bytes);
        if (buffer == nil) {
            throw std::runtime_error(std::string(
                "Metal failed to allocate ") + operation + " buffer");
        }
        execution.io_buffers.push_back(buffer);
        if (data != nullptr) {
            std::memcpy([buffer contents], data, bytes);
        }
        return buffer;
    }

    id<MTLComputeCommandEncoder> async_encoder(
        AsyncExecution& execution,
        id<MTLComputePipelineState> pipeline,
        const char* operation,
        KernelWorkload workload) const {
        const uint64_t cpu_begin_ns = monotonic_time_ns();
        const bool detailed = metal_profiler != nullptr &&
            metal_profiler->detailed_kernel_timestamps();
        MetalKernelProfileRecord record;
        bool has_record = false;
        if (detailed) {
            record.order = execution.kernels.size();
            if (execution.current_layer != kNoProfileIndex) {
                record.layer_index = execution.current_layer;
            }
            if (execution.current_head != kNoProfileIndex) {
                record.head_index = execution.current_head;
            }
            record.operation = operation;
            record.pipeline = pipeline_name(pipeline);
            record.value_type = std::move(workload.value_type);
            record.m = workload.m;
            record.n = workload.n;
            record.k = workload.k;
            record.metrics = workload.metrics;
            record.weight_bytes = workload.weight_bytes;
            record.shader_read_bytes = workload.shader_read_bytes;
            if (!execution.counter_sample_buffers.empty() &&
                execution.next_sample + 1 <
                    execution.counter_sample_capacity) {
                record.start_sample_index = execution.next_sample++;
                record.end_sample_index = execution.next_sample++;
            }
            has_record = true;
        }

        id<MTLComputeCommandEncoder> encoder = nil;
        if (has_record && !execution.counter_sample_buffers.empty() &&
            counter_sampling_mode == CounterSamplingMode::StageBoundary &&
            record.start_sample_index != kNoProfileIndex) {
            const size_t buffer_index =
                record.start_sample_index / kCounterSamplesPerBuffer;
            const NSUInteger local_start = static_cast<NSUInteger>(
                record.start_sample_index % kCounterSamplesPerBuffer);
            const NSUInteger local_end = static_cast<NSUInteger>(
                record.end_sample_index % kCounterSamplesPerBuffer);
            MTLComputePassDescriptor* pass =
                [MTLComputePassDescriptor computePassDescriptor];
            MTLComputePassSampleBufferAttachmentDescriptor* attachment =
                [pass.sampleBufferAttachments objectAtIndexedSubscript:0];
            attachment.sampleBuffer =
                execution.counter_sample_buffers[buffer_index];
            attachment.startOfEncoderSampleIndex = local_start;
            attachment.endOfEncoderSampleIndex = local_end;
            encoder = [execution.command
                computeCommandEncoderWithDescriptor:pass];
        } else {
            encoder = [execution.command computeCommandEncoder];
        }
        if (encoder == nil) {
            throw std::runtime_error(std::string(
                "Metal failed to create ") + operation + " encoder");
        }
        NSString* encoder_label = [NSString stringWithUTF8String:operation];
        if (encoder_label != nil) {
            [encoder setLabel:encoder_label];
        }
        [encoder setComputePipelineState:pipeline];
        ++execution.kernel_count;

        if (has_record) {
            if (!execution.counter_sample_buffers.empty() &&
                counter_sampling_mode ==
                    CounterSamplingMode::DispatchBoundary &&
                record.start_sample_index != kNoProfileIndex) {
                const size_t buffer_index =
                    record.start_sample_index / kCounterSamplesPerBuffer;
                const NSUInteger local_index = static_cast<NSUInteger>(
                    record.start_sample_index % kCounterSamplesPerBuffer);
                [encoder sampleCountersInBuffer:
                             execution.counter_sample_buffers[buffer_index]
                                  atSampleIndex:local_index
                                    withBarrier:YES];
            }
            execution.kernels.push_back(std::move(record));
            execution.active_kernel = execution.kernels.size() - 1;
            execution.active_kernel_cpu_start_ns = cpu_begin_ns;
        }
        return encoder;
    }

    void end_async_encoder(AsyncExecution& execution,
                           id<MTLComputeCommandEncoder> encoder,
                           const DispatchGeometry& geometry) const {
        execution.threadgroup_count += geometry.threadgroups;
        execution.dispatched_threads += geometry.dispatched_threads;
        if (geometry.threadgroups == 1) {
            ++execution.single_threadgroup_kernels;
        }

        if (execution.active_kernel != kNoProfileIndex) {
            MetalKernelProfileRecord& record =
                execution.kernels[execution.active_kernel];
            record.dispatch_type = geometry.type;
            record.grid_x = geometry.grid.width;
            record.grid_y = geometry.grid.height;
            record.grid_z = geometry.grid.depth;
            record.threads_x = geometry.threads.width;
            record.threads_y = geometry.threads.height;
            record.threads_z = geometry.threads.depth;
            record.threadgroups = geometry.threadgroups;
            record.dispatched_threads = geometry.dispatched_threads;
            if (!execution.counter_sample_buffers.empty() &&
                counter_sampling_mode ==
                    CounterSamplingMode::DispatchBoundary &&
                record.end_sample_index != kNoProfileIndex) {
                const size_t buffer_index =
                    record.end_sample_index / kCounterSamplesPerBuffer;
                const NSUInteger local_index = static_cast<NSUInteger>(
                    record.end_sample_index % kCounterSamplesPerBuffer);
                [encoder sampleCountersInBuffer:
                             execution.counter_sample_buffers[buffer_index]
                                  atSampleIndex:local_index
                                    withBarrier:YES];
            }
        }
        [encoder endEncoding];
        if (execution.active_kernel != kNoProfileIndex) {
            MetalKernelProfileRecord& record =
                execution.kernels[execution.active_kernel];
            record.cpu_encode_ns = monotonic_time_ns() -
                execution.active_kernel_cpu_start_ns;
            execution.active_kernel = kNoProfileIndex;
        }
    }

    void finish_async(AsyncExecution& execution, const char* operation) const {
        if (metal_profiler == nullptr) {
            [execution.command commit];
            [execution.command waitUntilCompleted];
            check_command(execution.command, operation);
            return;
        }

        const uint64_t encode_end_ns = monotonic_time_ns();
        execution.callback_times = std::make_shared<CallbackTimes>();
        const std::shared_ptr<CallbackTimes> callback_times =
            execution.callback_times;
        [execution.command addScheduledHandler:^(id<MTLCommandBuffer>) {
            callback_times->scheduled_ns.store(
                monotonic_time_ns(), std::memory_order_relaxed);
        }];
        [execution.command addCompletedHandler:^(id<MTLCommandBuffer>) {
            callback_times->completed_ns.store(
                monotonic_time_ns(), std::memory_order_relaxed);
        }];

        const uint64_t commit_begin_ns = monotonic_time_ns();
        const double commit_host_seconds = mach_host_time_seconds();
        [execution.command commit];
        const uint64_t commit_end_ns = monotonic_time_ns();
        const uint64_t wait_begin_ns = commit_end_ns;
        [execution.command waitUntilCompleted];
        const uint64_t wait_end_ns = monotonic_time_ns();
        const double wait_end_host_seconds = mach_host_time_seconds();
        check_command(execution.command, operation);

        MetalCommandProfileRecord command;
        command.command_index = execution.command_index;
        command.phase = execution.phase;
        command.sequence_tokens = execution.sequence_tokens;
        command.kernel_count = execution.kernel_count;
        command.threadgroup_count = execution.threadgroup_count;
        command.dispatched_threads = execution.dispatched_threads;
        command.single_threadgroup_kernels =
            execution.single_threadgroup_kernels;
        command.counter_requested =
            metal_profiler->detailed_kernel_timestamps();
        command.counter_status = execution.counter_status.empty()
            ? timestamp_counter_status : execution.counter_status;
        command.sample_count = execution.next_sample;
        command.cpu_command_create_ns = execution.command_create_ns;
        command.cpu_encode_ns = encode_end_ns - execution.encode_begin_ns;
        command.cpu_commit_ns = commit_end_ns - commit_begin_ns;
        command.cpu_wait_ns = wait_end_ns - wait_begin_ns;
        command.cpu_total_ns = wait_end_ns - execution.begin_ns;

        const uint64_t scheduled_ns = callback_times->scheduled_ns.load(
            std::memory_order_relaxed);
        if (scheduled_ns >= commit_begin_ns) {
            command.commit_to_scheduled_callback_ms =
                static_cast<double>(scheduled_ns - commit_begin_ns) / 1.0e6;
        }
        const uint64_t completed_ns = callback_times->completed_ns.load(
            std::memory_order_relaxed);
        if (completed_ns >= commit_begin_ns) {
            command.commit_to_completed_callback_ms =
                static_cast<double>(completed_ns - commit_begin_ns) / 1.0e6;
        }
        if (completed_ns != 0 && wait_end_ns >= completed_ns) {
            command.completed_callback_to_wait_return_ms =
                static_cast<double>(wait_end_ns - completed_ns) / 1.0e6;
        }

        const double gpu_start = [execution.command GPUStartTime];
        const double gpu_end = [execution.command GPUEndTime];
        const double kernel_start = [execution.command kernelStartTime];
        const double kernel_end = [execution.command kernelEndTime];
        if (gpu_start > 0.0) {
            command.commit_to_gpu_start_ms =
                (gpu_start - commit_host_seconds) * 1.0e3;
        }
        if (kernel_start >= gpu_start && gpu_start > 0.0) {
            command.gpu_start_to_kernel_start_ms =
                (kernel_start - gpu_start) * 1.0e3;
        }
        if (kernel_start >= gpu_start && kernel_end >= kernel_start &&
            kernel_end <= gpu_end && gpu_start > 0.0) {
            command.kernel_window_ms =
                (kernel_end - kernel_start) * 1.0e3;
            command.kernel_end_to_gpu_end_ms =
                (gpu_end - kernel_end) * 1.0e3;
        }
        if (gpu_end >= gpu_start && gpu_start > 0.0) {
            command.gpu_duration_ms = (gpu_end - gpu_start) * 1.0e3;
            command.gpu_end_to_wait_return_ms =
                (wait_end_host_seconds - gpu_end) * 1.0e3;
        }

        const uint64_t resolve_begin_ns = monotonic_time_ns();
        if (!execution.counter_sample_buffers.empty() &&
            execution.next_sample != 0) {
            // MTLCounterResultTimestamp is expressed in nanoseconds. Do not
            // calibrate it with sampleTimestamps(), whose gpuTimestamp uses a
            // separate device-clock tick domain on Apple GPUs.
            command.timestamp_ns_per_tick = 1.0;

            std::vector<uint64_t> timestamps(
                execution.next_sample, MTLCounterErrorValue);
            bool resolve_succeeded = true;
            size_t resolved_samples = 0;
            for (size_t buffer_index = 0;
                 buffer_index < execution.counter_sample_buffers.size() &&
                 resolved_samples < execution.next_sample;
                 ++buffer_index) {
                id<MTLCounterSampleBuffer> sample_buffer =
                    execution.counter_sample_buffers[buffer_index];
                const size_t sample_count = std::min<size_t>(
                    [sample_buffer sampleCount],
                    execution.next_sample - resolved_samples);
                NSData* resolved = [sample_buffer resolveCounterRange:
                    NSMakeRange(0, sample_count)];
                const size_t required_bytes = checked_bytes_for(
                    sample_count, sizeof(MTLCounterResultTimestamp),
                    "Metal resolved timestamp bytes");
                if (resolved == nil || [resolved length] < required_bytes) {
                    command.counter_status =
                        "counter buffer " + std::to_string(buffer_index) +
                        " resolve failed";
                    resolve_succeeded = false;
                    break;
                }
                const auto* buffer_timestamps =
                    static_cast<const MTLCounterResultTimestamp*>(
                        [resolved bytes]);
                for (size_t index = 0; index < sample_count; ++index) {
                    timestamps[resolved_samples + index] =
                        buffer_timestamps[index].timestamp;
                }
                resolved_samples += sample_count;
            }
            if (resolve_succeeded &&
                resolved_samples == execution.next_sample) {
                size_t valid_kernel_count = 0;
                for (MetalKernelProfileRecord& kernel : execution.kernels) {
                    if (kernel.start_sample_index >= execution.next_sample ||
                        kernel.end_sample_index >= execution.next_sample) {
                        continue;
                    }
                    const uint64_t start =
                        timestamps[kernel.start_sample_index];
                    const uint64_t end =
                        timestamps[kernel.end_sample_index];
                    if (start == MTLCounterErrorValue ||
                        end == MTLCounterErrorValue || end < start) {
                        continue;
                    }
                    kernel.start_timestamp = start;
                    kernel.end_timestamp = end;
                    kernel.timestamp_valid = true;
                    ++valid_kernel_count;
                }
                command.counter_active = valid_kernel_count != 0;
                if (valid_kernel_count != execution.kernels.size()) {
                    command.counter_status =
                        "partial timestamps: " +
                        std::to_string(valid_kernel_count) + "/" +
                        std::to_string(execution.kernels.size());
                }
            } else if (resolve_succeeded) {
                command.counter_status = "not all counter buffers resolved";
            }
        }
        command.cpu_counter_resolve_ns =
            monotonic_time_ns() - resolve_begin_ns;
        metal_profiler->write(std::move(command), execution.kernels);
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

    id<MTLComputePipelineState> matrix_product_pipeline(
        MetalGgmlType type) const {
        switch (type) {
        case MetalGgmlType::F32: return gemm_pipeline;
        case MetalGgmlType::Q5_0: return gemm_q5_0_pipeline;
        case MetalGgmlType::Q8_0: return gemm_q8_0_pipeline;
        case MetalGgmlType::Q4_K: return gemm_q4_k_pipeline;
        case MetalGgmlType::Q6_K: return gemm_q6_k_pipeline;
        default:
            throw std::runtime_error(
                std::string("Metal GEMM does not support weight type ") +
                metal_ggml_type_name(type));
        }
    }

    id<MTLComputePipelineState> vector_product_pipeline(
        MetalGgmlType type) const {
        switch (type) {
        case MetalGgmlType::F32: return gevm_pipeline;
        case MetalGgmlType::Q5_0: return gevm_q5_0_pipeline;
        case MetalGgmlType::Q8_0: return gevm_q8_0_pipeline;
        case MetalGgmlType::Q4_K: return gevm_q4_k_pipeline;
        case MetalGgmlType::Q6_K: return gevm_q6_k_pipeline;
        default:
            throw std::runtime_error(
                std::string("Metal GEVM does not support weight type ") +
                metal_ggml_type_name(type));
        }
    }

    id<MTLComputePipelineState> embedding_pipeline_for(
        MetalGgmlType type) const {
        switch (type) {
        case MetalGgmlType::F32: return embedding_pipeline;
        case MetalGgmlType::Q5_0: return embedding_q5_0_pipeline;
        case MetalGgmlType::Q8_0: return embedding_q8_0_pipeline;
        case MetalGgmlType::Q4_K: return embedding_q4_k_pipeline;
        case MetalGgmlType::Q6_K: return embedding_q6_k_pipeline;
        default:
            throw std::runtime_error(
                std::string("Metal embedding does not support weight type ") +
                metal_ggml_type_name(type));
        }
    }

    DeviceMatrix encode_embedding(AsyncExecution& execution,
                                  id<MTLBuffer> token_ids,
                                  const QuantizedMatrix& embedding,
                                  size_t sequence_length,
                                  size_t vocabulary_size,
                                  ActivationSlot output_slot,
                                  const char* operation) const {
        if (embedding.rows != vocabulary_size || embedding.cols == 0) {
            throw std::invalid_argument(
                "Metal resident embedding weight shape is invalid");
        }
        DeviceMatrix output = arena_matrix(
            output_slot, sequence_length, embedding.cols, operation);
        MetalEmbeddingParamsHost params;
        params.sequence_length = checked_uint(
            sequence_length, "Resident embedding sequence length");
        params.embedding_size = checked_uint(
            embedding.cols, "Resident embedding size");
        params.vocabulary_size = checked_uint(
            vocabulary_size, "Resident embedding vocabulary size");
        params.weight_row_bytes = checked_uint(
            embedding.row_bytes, "Resident embedding row bytes");

        id<MTLComputePipelineState> pipeline =
            embedding_pipeline_for(embedding.type);

        KernelWorkload workload;
        workload.m = sequence_length;
        workload.n = embedding.cols;
        workload.value_type = metal_ggml_type_name(embedding.type);
        const size_t gathered_weight_bytes = checked_bytes_for(
            sequence_length, embedding.row_bytes,
            "Metal profile embedding weight bytes");
        const size_t token_bytes = checked_bytes_for(
            sequence_length, sizeof(int32_t),
            "Metal profile embedding token bytes");
        const size_t output_bytes = checked_bytes(
            checked_elements(sequence_length, embedding.cols,
                             "Metal profile embedding elements"),
            "Metal profile embedding output bytes");
        workload.metrics.read_bytes = static_cast<uint64_t>(checked_add_size(
            gathered_weight_bytes, token_bytes,
            "Metal profile embedding reads"));
        workload.metrics.write_bytes = output_bytes;
        workload.metrics.temporary_bytes = output_bytes;
        workload.weight_bytes = gathered_weight_bytes;
        workload.shader_read_bytes = workload.metrics.read_bytes;

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, pipeline, operation, std::move(workload));
        [encoder setBuffer:token_ids offset:0 atIndex:0];
        [encoder setBuffer:embedding.buffer
                   offset:embedding.offset_bytes
                  atIndex:1];
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident embedding output offset")
                  atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        const DispatchGeometry geometry = dispatch_flat(
            encoder, pipeline,
            checked_elements(sequence_length, embedding.cols,
                             "Resident embedding dispatch"));
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceMatrix encode_matrix_product(AsyncExecution& execution,
                                       const DeviceMatrix& left,
                                       const QuantizedMatrix& right,
                                       bool left_transposed,
                                       bool right_transposed,
                                       const DeviceVector* bias,
                                       float scale,
                                       ActivationSlot output_slot,
                                       const char* operation) const {
        if (!std::isfinite(scale)) {
            throw std::invalid_argument(
                "Metal resident GEMM scale must be finite");
        }
        if (left.type != DeviceDataType::F32) {
            throw std::invalid_argument(
                "Metal resident GEMM activation must be FP32");
        }
        if (right.type != MetalGgmlType::F32 &&
            (left_transposed || !right_transposed)) {
            throw std::invalid_argument(
                "Metal quantized GEMM requires activation * weight^T");
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

        DeviceMatrix output = arena_matrix(output_slot, m, n, operation);
        if (m == 0 || n == 0) {
            return output;
        }

        id<MTLComputePipelineState> pipeline =
            matrix_product_pipeline(right.type);
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, pipeline, operation,
            product_workload(right, m, n, left_k, bias != nullptr));
        [encoder setBuffer:left.buffer
                   offset:checked_bytes_for(left.offset_elements,
                                            sizeof(float),
                                            "Resident GEMM lhs offset")
                  atIndex:0];
        [encoder setBuffer:right.buffer
                   offset:right.offset_bytes
                  atIndex:1];
        id<MTLBuffer> bias_buffer = bias == nullptr
            ? dummy_buffer
            : bias->buffer;
        const size_t bias_offset = bias == nullptr ? 0 : bias->offset_elements;
        [encoder setBuffer:bias_buffer
                   offset:checked_bytes_for(bias_offset, sizeof(float),
                                            "Resident GEMM bias offset")
                  atIndex:2];
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident GEMM output offset")
                  atIndex:3];
        if (right.type == MetalGgmlType::F32) {
            if (right.row_bytes % sizeof(float) != 0) {
                throw std::invalid_argument(
                    "Metal F32 weight row stride is not float aligned");
            }
            MetalMatmulParamsHost params;
            params.m = checked_uint(m, "Resident GEMM M");
            params.n = checked_uint(n, "Resident GEMM N");
            params.k = checked_uint(left_k, "Resident GEMM K");
            params.lhs_stride = checked_uint(
                left.stride, "Resident GEMM lhs stride");
            params.rhs_stride = checked_uint(
                right.row_bytes / sizeof(float),
                "Resident GEMM rhs stride");
            params.lhs_transposed = left_transposed ? 1U : 0U;
            params.rhs_transposed = right_transposed ? 1U : 0U;
            params.has_bias = bias == nullptr ? 0U : 1U;
            params.scale = scale;
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
        } else {
            MetalQuantizedProductParamsHost params;
            params.m = checked_uint(m, "Resident quantized GEMM M");
            params.n = checked_uint(n, "Resident quantized GEMM N");
            params.k = checked_uint(left_k, "Resident quantized GEMM K");
            params.activation_stride = checked_uint(
                left.stride, "Resident quantized GEMM activation stride");
            params.weight_row_bytes = checked_uint(
                right.row_bytes, "Resident quantized GEMM row bytes");
            params.has_bias = bias == nullptr ? 0U : 1U;
            params.scale = scale;
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
        }

        const MTLSize threadgroups = MTLSizeMake(
            (static_cast<NSUInteger>(n) + kTileSize - 1) / kTileSize,
            (static_cast<NSUInteger>(m) + kTileSize - 1) / kTileSize, 1);
        const DispatchGeometry geometry = dispatch_threadgroups(
            encoder, threadgroups,
            MTLSizeMake(kTileSize, kTileSize, 1));
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceVector encode_vector_product(AsyncExecution& execution,
                                       const QuantizedMatrix& matrix,
                                       const DeviceVector& input,
                                       bool matrix_transposed,
                                       const DeviceVector* bias,
                                       float scale,
                                       ActivationSlot output_slot,
                                       const char* operation) const {
        if (!std::isfinite(scale)) {
            throw std::invalid_argument(
                "Metal resident GEVM scale must be finite");
        }
        if (input.type != DeviceDataType::F32) {
            throw std::invalid_argument(
                "Metal resident GEVM activation must be FP32");
        }
        if (matrix.type != MetalGgmlType::F32 && !matrix_transposed) {
            throw std::invalid_argument(
                "Metal quantized GEVM requires weight * activation");
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

        DeviceVector output = arena_vector(output_slot, output_size, operation);
        if (output_size == 0) {
            return output;
        }

        id<MTLComputePipelineState> pipeline =
            vector_product_pipeline(matrix.type);
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, pipeline, operation,
            product_workload(
                matrix, 1, output_size, input_size, bias != nullptr));
        [encoder setBuffer:matrix.buffer
                   offset:matrix.offset_bytes
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
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident GEVM output offset")
                  atIndex:3];
        if (matrix.type == MetalGgmlType::F32) {
            if (matrix.row_bytes % sizeof(float) != 0) {
                throw std::invalid_argument(
                    "Metal F32 weight row stride is not float aligned");
            }
            MetalGemvParamsHost params;
            params.output_size = checked_uint(
                output_size, "Resident GEVM output");
            params.input_size = checked_uint(
                input_size, "Resident GEVM input");
            params.matrix_stride = checked_uint(
                matrix.row_bytes / sizeof(float),
                "Resident GEVM matrix stride");
            params.matrix_transposed = matrix_transposed ? 1U : 0U;
            params.has_bias = bias == nullptr ? 0U : 1U;
            params.scale = scale;
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
        } else {
            MetalQuantizedProductParamsHost params;
            params.m = 1;
            params.n = checked_uint(
                output_size, "Resident quantized GEVM output");
            params.k = checked_uint(
                input_size, "Resident quantized GEVM input");
            params.activation_stride = checked_uint(
                input_size, "Resident quantized GEVM activation stride");
            params.weight_row_bytes = checked_uint(
                matrix.row_bytes, "Resident quantized GEVM row bytes");
            params.has_bias = bias == nullptr ? 0U : 1U;
            params.scale = scale;
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
        }
        const NSUInteger threads = std::min<NSUInteger>(
            kDefaultGemvThreads,
            std::max<NSUInteger>(
                1, [pipeline maxTotalThreadsPerThreadgroup]));
        const DispatchGeometry geometry = dispatch_threadgroups(
            encoder,
            MTLSizeMake((static_cast<NSUInteger>(output_size) +
                         threads - 1) / threads, 1, 1),
            MTLSizeMake(threads, 1, 1));
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceMatrix encode_rmsnorm_matrix(AsyncExecution& execution,
                                       const DeviceMatrix& input,
                                       const DeviceVector& gamma,
                                       float epsilon,
                                       ActivationSlot output_slot,
                                       const char* operation) const {
        if (gamma.length != input.cols || input.rows == 0 || input.cols == 0) {
            throw std::invalid_argument(
                "Metal resident RMSNorm matrix dimensions are invalid");
        }
        DeviceMatrix output = arena_matrix(
            output_slot, input.rows, input.cols, operation);
        MetalRmsNormParamsHost params;
        params.rows = checked_uint(input.rows, "Resident RMSNorm rows");
        params.cols = checked_uint(input.cols, "Resident RMSNorm columns");
        params.epsilon = epsilon;

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, rmsnorm_pipeline, operation,
            elementwise_workload(input.rows, input.cols, 5, 2, 1));
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
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident RMSNorm output offset")
                  atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        const DispatchGeometry geometry = dispatch_rows(encoder, input.rows);
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceVector encode_rmsnorm_vector(AsyncExecution& execution,
                                       const DeviceVector& input,
                                       const DeviceVector& gamma,
                                       float epsilon,
                                       ActivationSlot output_slot,
                                       const char* operation) const {
        if (input.length == 0 || gamma.length != input.length) {
            throw std::invalid_argument(
                "Metal resident RMSNorm vector dimensions are invalid");
        }
        DeviceVector output = arena_vector(
            output_slot, input.length, operation);
        MetalRmsNormParamsHost params;
        params.rows = 1;
        params.cols = checked_uint(input.length, "Resident RMSNorm length");
        params.epsilon = epsilon;

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, rmsnorm_pipeline, operation,
            elementwise_workload(1, input.length, 5, 2, 1));
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
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident RMSNorm vector output offset")
                  atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        const DispatchGeometry geometry = dispatch_rows(encoder, 1);
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceMatrix encode_softmax_matrix(AsyncExecution& execution,
                                       const DeviceMatrix& input,
                                       ActivationSlot output_slot,
                                       const char* operation) const {
        if (input.rows == 0 || input.cols == 0) {
            throw std::invalid_argument(
                "Metal resident Softmax matrix cannot be empty");
        }
        DeviceMatrix output = arena_matrix(
            output_slot, input.rows, input.cols, operation);
        MetalElementwiseParamsHost params;
        params.rows = checked_uint(input.rows, "Resident Softmax rows");
        params.cols = checked_uint(input.cols, "Resident Softmax columns");

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, softmax_pipeline, operation,
            elementwise_workload(input.rows, input.cols, 4, 3, 2));
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident Softmax input offset")
                  atIndex:0];
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident Softmax output offset")
                  atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        const DispatchGeometry geometry = dispatch_rows(encoder, input.rows);
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceVector encode_softmax_vector(AsyncExecution& execution,
                                       const DeviceVector& input,
                                       ActivationSlot output_slot,
                                       const char* operation) const {
        if (input.length == 0) {
            throw std::invalid_argument(
                "Metal resident Softmax vector cannot be empty");
        }
        DeviceVector output = arena_vector(
            output_slot, input.length, operation);
        MetalElementwiseParamsHost params;
        params.rows = 1;
        params.cols = checked_uint(input.length, "Resident Softmax length");

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, softmax_pipeline, operation,
            elementwise_workload(1, input.length, 4, 3, 2));
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident Softmax input offset")
                  atIndex:0];
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident Softmax vector output offset")
                  atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        const DispatchGeometry geometry = dispatch_rows(encoder, 1);
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceMatrix encode_binary_matrix(AsyncExecution& execution,
                                      const DeviceMatrix& left,
                                      const DeviceMatrix& right,
                                      id<MTLComputePipelineState> pipeline,
                                      ActivationSlot output_slot,
                                      const char* operation) const {
        if (left.rows != right.rows || left.cols != right.cols ||
            left.rows == 0 || left.cols == 0) {
            throw std::invalid_argument(
                "Metal resident binary matrix dimensions are invalid");
        }
        DeviceMatrix output = arena_matrix(
            output_slot, left.rows, left.cols, operation);
        MetalElementwiseParamsHost params;
        params.rows = checked_uint(left.rows, "Resident binary rows");
        params.cols = checked_uint(left.cols, "Resident binary columns");
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, pipeline, operation,
            elementwise_workload(
                left.rows, left.cols,
                pipeline == swiglu_pipeline ? 8 : 1, 2, 1));
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
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident binary output offset")
                  atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        const DispatchGeometry geometry = dispatch_flat(
            encoder, pipeline,
            checked_elements(left.rows, left.cols,
                             "Resident binary dispatch"));
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceVector encode_binary_vector(AsyncExecution& execution,
                                      const DeviceVector& left,
                                      const DeviceVector& right,
                                      id<MTLComputePipelineState> pipeline,
                                      ActivationSlot output_slot,
                                      const char* operation) const {
        if (left.length != right.length || left.length == 0) {
            throw std::invalid_argument(
                "Metal resident binary vector dimensions are invalid");
        }
        DeviceVector output = arena_vector(
            output_slot, left.length, operation);
        MetalElementwiseParamsHost params;
        params.rows = 1;
        params.cols = checked_uint(left.length, "Resident binary length");
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, pipeline, operation,
            elementwise_workload(
                1, left.length,
                pipeline == swiglu_pipeline ? 8 : 1, 2, 1));
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
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident binary vector output offset")
                  atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        const DispatchGeometry geometry = dispatch_flat(
            encoder, pipeline, left.length);
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceMatrix encode_rope_heads_matrix(AsyncExecution& execution,
                                          const DeviceMatrix& input,
                                          size_t head_dim,
                                          size_t rotary_dimension,
                                          size_t position,
                                          float theta,
                                          ActivationSlot output_slot,
                                          const char* operation) const {
        if (input.rows == 0 || input.cols == 0 || head_dim == 0 ||
            input.cols % head_dim != 0 || rotary_dimension == 0 ||
            rotary_dimension > head_dim || rotary_dimension % 2 != 0 ||
            !std::isfinite(theta) || theta <= 0.0f) {
            throw std::invalid_argument(
                "Metal resident head-wise RoPE matrix dimensions are invalid");
        }
        DeviceMatrix output = arena_matrix(
            output_slot, input.rows, input.cols, operation);
        MetalRopeHeadsParamsHost params;
        params.rows = checked_uint(input.rows, "Resident RoPE rows");
        params.cols = checked_uint(input.cols, "Resident RoPE columns");
        params.head_dim = checked_uint(head_dim, "Resident RoPE head dimension");
        params.rotary_dimension = checked_uint(
            rotary_dimension, "Resident RoPE rotary dimension");
        params.position = checked_uint(position, "Resident RoPE position");
        params.theta = theta;
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, rope_heads_pipeline, operation,
            elementwise_workload(input.rows, input.cols, 6, 1, 1));
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident RoPE input offset")
                  atIndex:0];
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident RoPE output offset")
                  atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        const DispatchGeometry geometry = dispatch_flat(
            encoder, rope_heads_pipeline,
            checked_elements(input.rows, input.cols,
                             "Resident RoPE dispatch"));
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    DeviceVector encode_rope_heads_vector(AsyncExecution& execution,
                                          const DeviceVector& input,
                                          size_t head_dim,
                                          size_t rotary_dimension,
                                          size_t position,
                                          float theta,
                                          ActivationSlot output_slot,
                                          const char* operation) const {
        if (input.length == 0 || head_dim == 0 ||
            input.length % head_dim != 0 || rotary_dimension == 0 ||
            rotary_dimension > head_dim || rotary_dimension % 2 != 0 ||
            !std::isfinite(theta) || theta <= 0.0f) {
            throw std::invalid_argument(
                "Metal resident head-wise RoPE vector dimensions are invalid");
        }
        DeviceVector output = arena_vector(
            output_slot, input.length, operation);
        MetalRopeHeadsParamsHost params;
        params.rows = 1;
        params.cols = checked_uint(input.length, "Resident RoPE length");
        params.head_dim = checked_uint(head_dim, "Resident RoPE head dimension");
        params.rotary_dimension = checked_uint(
            rotary_dimension, "Resident RoPE rotary dimension");
        params.position = checked_uint(position, "Resident RoPE position");
        params.theta = theta;
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, rope_heads_pipeline, operation,
            elementwise_workload(1, input.length, 6, 1, 1));
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident RoPE input offset")
                  atIndex:0];
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident RoPE vector output offset")
                  atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        const DispatchGeometry geometry = dispatch_flat(
            encoder, rope_heads_pipeline, input.length);
        end_async_encoder(execution, encoder, geometry);
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
            execution, kv_cache_write_pipeline, operation,
            elementwise_workload(
                key_source.rows, key_source.cols, 0, 2, 2));
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
        const DispatchGeometry geometry = dispatch_flat(
            encoder, kv_cache_write_pipeline,
            checked_elements(key_source.rows, key_source.cols,
                             "Resident KV write dispatch"));
        end_async_encoder(execution, encoder, geometry);
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
                                    ActivationSlot output_slot,
                                    const char* operation) const {
        if (query == nil || key_cache == nil || query_rows == 0 ||
            key_length == 0 || head_dim == 0) {
            throw std::invalid_argument(
                "Metal resident KV-cache QK dimensions are invalid");
        }

        DeviceMatrix output = arena_matrix(
            output_slot, query_rows, key_length, operation);
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

        KernelWorkload workload;
        workload.m = query_rows;
        workload.n = key_length;
        workload.k = head_dim;
        workload.value_type = "F32";
        workload.metrics = profile_gemm_metrics(
            query_rows, key_length, head_dim);
        workload.metrics.flops += static_cast<uint64_t>(checked_elements(
            query_rows, key_length, "Metal profile QK scale operations"));
        workload.shader_read_bytes = static_cast<uint64_t>(checked_bytes_for(
            checked_bytes(
                checked_elements(
                    checked_elements(query_rows, key_length,
                                     "Metal profile QK outputs"),
                    head_dim, "Metal profile QK products"),
                "Metal profile QK operand bytes"),
            2, "Metal profile QK shader reads"));

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, kv_cache_qk_pipeline, operation,
            std::move(workload));
        [encoder setBuffer:query
                   offset:checked_bytes_for(query_offset_elements,
                                            sizeof(float),
                                            "Resident QK query offset")
                  atIndex:0];
        [encoder setBuffer:key_cache offset:0 atIndex:1];
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident QK output offset")
                  atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        const DispatchGeometry geometry = dispatch_flat(
            encoder, kv_cache_qk_pipeline,
            checked_elements(query_rows, key_length,
                             "Resident QK dispatch"));
        end_async_encoder(execution, encoder, geometry);
        return output;
    }

    void encode_kv_cache_av(AsyncExecution& execution,
                            const DeviceMatrix& scores,
                            id<MTLBuffer> value_cache,
                            const DeviceMatrix& output,
                            size_t query_rows,
                            size_t key_length,
                            size_t cache_stride,
                            size_t score_stride,
                            size_t head_dim,
                            size_t cache_head,
                            size_t output_stride,
                            size_t output_offset,
                            const char* operation) const {
        if (scores.buffer == nil || value_cache == nil ||
            output.buffer == nil ||
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

        KernelWorkload workload;
        workload.m = query_rows;
        workload.n = head_dim;
        workload.k = key_length;
        workload.value_type = "F32";
        workload.metrics = profile_gemm_metrics(
            query_rows, head_dim, key_length);
        workload.shader_read_bytes = static_cast<uint64_t>(checked_bytes_for(
            checked_bytes(
                checked_elements(
                    checked_elements(query_rows, head_dim,
                                     "Metal profile AV outputs"),
                    key_length, "Metal profile AV products"),
                "Metal profile AV operand bytes"),
            2, "Metal profile AV shader reads"));

        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, kv_cache_av_pipeline, operation,
            std::move(workload));
        [encoder setBuffer:scores.buffer
                   offset:checked_bytes_for(scores.offset_elements,
                                            sizeof(float),
                                            "Resident AV score offset")
                  atIndex:0];
        [encoder setBuffer:value_cache offset:0 atIndex:1];
        [encoder setBuffer:output.buffer
                   offset:checked_bytes_for(output.offset_elements,
                                            sizeof(float),
                                            "Resident AV output offset")
                  atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        const DispatchGeometry geometry = dispatch_flat(
            encoder, kv_cache_av_pipeline,
            checked_elements(query_rows, head_dim,
                             "Resident AV dispatch"));
        end_async_encoder(execution, encoder, geometry);
    }

    id<MTLBuffer> encode_argmax(AsyncExecution& execution,
                                const DeviceVector& input,
                                size_t length,
                                const char* operation) const {
        if (input.buffer == nil || length == 0) {
            throw std::invalid_argument(
                "Metal resident argmax input cannot be empty");
        }
        id<MTLBuffer> output = async_buffer(
            execution, nullptr, sizeof(uint32_t), operation);
        MetalArgmaxParamsHost params;
        params.length = checked_uint(length, "Resident argmax length");
        KernelWorkload workload;
        workload.m = 1;
        workload.n = length;
        workload.value_type = "F32";
        workload.metrics.flops = length > 0 ? length - 1 : 0;
        workload.metrics.read_bytes = checked_bytes(
            length, "Metal profile argmax reads");
        workload.metrics.write_bytes = sizeof(uint32_t);
        workload.shader_read_bytes = workload.metrics.read_bytes;
        id<MTLComputeCommandEncoder> encoder = async_encoder(
            execution, argmax_pipeline, operation, std::move(workload));
        [encoder setBuffer:input.buffer
                   offset:checked_bytes_for(input.offset_elements,
                                            sizeof(float),
                                            "Resident argmax input offset")
                  atIndex:0];
        [encoder setBuffer:output offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        const DispatchGeometry geometry = dispatch_threads(
            encoder, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
        end_async_encoder(execution, encoder, geometry);
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
                                      size_t position,
                                      ActivationSlot output_slot) const;

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
                                     size_t position,
                                     ActivationSlot output_slot) const;

    void upload_model(MetalRawModel& model);
    void allocate_kv_cache(size_t capacity);
    void install_activation_arena(const ActivationArenaPlan& plan,
                                  ActivationArenaPhase phase,
                                  const char* operation);
    void prepare_prefill_activation_arena(size_t prompt_tokens);
    void ensure_decode_activation_arena();
    void release_activation_arena() noexcept;
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
    size_t position,
    ActivationSlot output_slot) const {
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

    // Every helper below appends another dispatch to the same command buffer.
    // Ending an encoder establishes command order; Metal's tracked resources
    // make one kernel's writes visible to the next without a CPU-side barrier.
    const DeviceVector& attn_norm = layer.attn_norm_weight;
    const DeviceMatrix normalized = encode_rmsnorm_matrix(
        execution, hidden, attn_norm, epsilon, ActivationSlot::Norm,
        "resident prefill attention norm");

    const QuantizedMatrix& q_weight = layer.attn_q_weight;
    const DeviceVector& q_bias = layer.attn_q_bias;
    const QuantizedMatrix& k_weight = layer.attn_k_weight;
    const DeviceVector& k_bias = layer.attn_k_bias;
    const QuantizedMatrix& v_weight = layer.attn_v_weight;
    const DeviceVector& v_bias = layer.attn_v_bias;

    const DeviceMatrix query = encode_matrix_product(
        execution, normalized, q_weight, false, true, &q_bias, 1.0f,
        ActivationSlot::Query,
        "resident prefill Q projection");
    const DeviceMatrix key = encode_matrix_product(
        execution, normalized, k_weight, false, true, &k_bias, 1.0f,
        ActivationSlot::Key,
        "resident prefill K projection");
    const DeviceMatrix value = encode_matrix_product(
        execution, normalized, v_weight, false, true, &v_bias, 1.0f,
        ActivationSlot::Value,
        "resident prefill V projection");

    const DeviceMatrix rotated_query = encode_rope_heads_matrix(
        execution, query, d_head, d_rope, position, theta,
        ActivationSlot::RotatedQuery,
        "resident prefill Q RoPE");
    const DeviceMatrix rotated_key = encode_rope_heads_matrix(
        execution, key, d_head, d_rope, position, theta,
        ActivationSlot::RotatedKey,
        "resident prefill K RoPE");
    encode_kv_cache_write(
        execution, rotated_key, value, key_cache, value_cache, position,
        "resident prefill KV write");

    // AV writes each head directly into its final packed [sequence, Q]
    // layout. This removes both per-head split buffers and the later concat.
    const DeviceMatrix attention_output = arena_matrix(
        ActivationSlot::AttentionOutput, hidden.rows, q_dimension,
        "resident prefill attention output");
    const size_t group_size = q_head_count / kv_head_count;
    for (size_t head = 0; head < q_head_count; ++head) {
        execution.current_head = head;
        const size_t group = head / group_size;
        const size_t head_offset = checked_elements(
            head, d_head, "Resident prefill Q head offset");
        const DeviceMatrix scores = encode_kv_cache_qk(
            execution, rotated_query.buffer,
            rotated_query.offset_elements + head_offset, hidden.rows,
            key_length, key_cache, cache_stride, q_dimension, d_head, group,
            position, 1.0f / std::sqrt(static_cast<float>(d_head)),
            ActivationSlot::AttentionScores,
            "resident prefill QK");
        const DeviceMatrix probabilities = encode_softmax_matrix(
            execution, scores, ActivationSlot::AttentionScores,
            "resident prefill Softmax");
        encode_kv_cache_av(
            execution, probabilities, value_cache, attention_output,
            hidden.rows, key_length, cache_stride,
            key_length, d_head, group, q_dimension, head_offset,
            "resident prefill AV");
    }
    execution.current_head = kNoProfileIndex;

    const QuantizedMatrix& output_weight = layer.attn_output_weight;
    const DeviceMatrix attention_projected = encode_matrix_product(
        execution, attention_output, output_weight, false, true, nullptr, 1.0f,
        ActivationSlot::Projection,
        "resident prefill attention output projection");
    const DeviceMatrix after_attention = encode_binary_matrix(
        execution, hidden, attention_projected, residual_pipeline,
        output_slot,
        "resident prefill attention residual");

    const DeviceVector& ffn_norm = layer.ffn_norm_weight;
    const DeviceMatrix ffn_input = encode_rmsnorm_matrix(
        execution, after_attention, ffn_norm, epsilon,
        ActivationSlot::Norm,
        "resident prefill FFN norm");
    const QuantizedMatrix& gate_weight = layer.ffn_gate_weight;
    const QuantizedMatrix& up_weight = layer.ffn_up_weight;
    const QuantizedMatrix& down_weight = layer.ffn_down_weight;
    const DeviceMatrix gate = encode_matrix_product(
        execution, ffn_input, gate_weight, false, true, nullptr, 1.0f,
        ActivationSlot::Gate,
        "resident prefill FFN gate projection");
    const DeviceMatrix up = encode_matrix_product(
        execution, ffn_input, up_weight, false, true, nullptr, 1.0f,
        ActivationSlot::Up,
        "resident prefill FFN up projection");
    const DeviceMatrix activated = encode_binary_matrix(
        execution, gate, up, swiglu_pipeline, ActivationSlot::Gate,
        "resident prefill SwiGLU");
    const DeviceMatrix down = encode_matrix_product(
        execution, activated, down_weight, false, true, nullptr, 1.0f,
        ActivationSlot::Projection,
        "resident prefill FFN down projection");
    return encode_binary_matrix(
        execution, after_attention, down, residual_pipeline,
        output_slot,
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
    size_t position,
    ActivationSlot output_slot) const {
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

    // Decode records the same logical layer as prefill, but all dense linear
    // projections use GEVM because the query row count is exactly one.
    const DeviceVector& attn_norm = layer.attn_norm_weight;
    const DeviceVector normalized = encode_rmsnorm_vector(
        execution, hidden, attn_norm, epsilon, ActivationSlot::Norm,
        "resident decode attention norm");

    const QuantizedMatrix& q_weight = layer.attn_q_weight;
    const DeviceVector& q_bias = layer.attn_q_bias;
    const QuantizedMatrix& k_weight = layer.attn_k_weight;
    const DeviceVector& k_bias = layer.attn_k_bias;
    const QuantizedMatrix& v_weight = layer.attn_v_weight;
    const DeviceVector& v_bias = layer.attn_v_bias;

    const DeviceVector query = encode_vector_product(
        execution, q_weight, normalized, true, &q_bias, 1.0f,
        ActivationSlot::Query,
        "resident decode Q projection");
    const DeviceVector key = encode_vector_product(
        execution, k_weight, normalized, true, &k_bias, 1.0f,
        ActivationSlot::Key,
        "resident decode K projection");
    const DeviceVector value = encode_vector_product(
        execution, v_weight, normalized, true, &v_bias, 1.0f,
        ActivationSlot::Value,
        "resident decode V projection");
    const DeviceVector rotated_query = encode_rope_heads_vector(
        execution, query, d_head, d_rope, position, theta,
        ActivationSlot::RotatedQuery,
        "resident decode Q RoPE");
    const DeviceVector rotated_key = encode_rope_heads_vector(
        execution, key, d_head, d_rope, position, theta,
        ActivationSlot::RotatedKey,
        "resident decode K RoPE");

    const DeviceMatrix key_source = {
        rotated_key.buffer, 1, rotated_key.length, rotated_key.length,
        rotated_key.offset_elements};
    const DeviceMatrix value_source = {
        value.buffer, 1, value.length, value.length, value.offset_elements};
    encode_kv_cache_write(
        execution, key_source, value_source, key_cache, value_cache, position,
        "resident decode KV write");

    const DeviceVector attention_output = arena_vector(
        ActivationSlot::AttentionOutput, q_dimension,
        "resident decode attention output");
    const DeviceMatrix attention_output_matrix = {
        attention_output.buffer, 1, attention_output.length,
        attention_output.length, attention_output.offset_elements};
    const size_t group_size = q_head_count / kv_head_count;
    for (size_t head = 0; head < q_head_count; ++head) {
        execution.current_head = head;
        const size_t group = head / group_size;
        const size_t head_offset = checked_elements(
            head, d_head, "Resident decode Q head offset");
        const DeviceMatrix scores = encode_kv_cache_qk(
            execution, rotated_query.buffer,
            rotated_query.offset_elements + head_offset, 1, key_length,
            key_cache, cache_stride, q_dimension, d_head, group, position,
            1.0f / std::sqrt(static_cast<float>(d_head)),
            ActivationSlot::AttentionScores,
            "resident decode QK");
        const DeviceVector probabilities = encode_softmax_vector(
            execution,
            {scores.buffer, key_length, scores.offset_elements},
            ActivationSlot::AttentionScores,
            "resident decode Softmax");
        encode_kv_cache_av(
            execution,
            {probabilities.buffer, 1, key_length, key_length,
             probabilities.offset_elements},
            value_cache, attention_output_matrix, 1, key_length, cache_stride,
            key_length,
            d_head, group, q_dimension, head_offset, "resident decode AV");
    }
    execution.current_head = kNoProfileIndex;

    const QuantizedMatrix& output_weight = layer.attn_output_weight;
    const DeviceVector attention_projected = encode_vector_product(
        execution, output_weight, attention_output, true, nullptr, 1.0f,
        ActivationSlot::Projection,
        "resident decode attention output projection");
    const DeviceVector after_attention = encode_binary_vector(
        execution, hidden, attention_projected, residual_pipeline,
        output_slot,
        "resident decode attention residual");

    const DeviceVector& ffn_norm = layer.ffn_norm_weight;
    const DeviceVector ffn_input = encode_rmsnorm_vector(
        execution, after_attention, ffn_norm, epsilon,
        ActivationSlot::Norm,
        "resident decode FFN norm");
    const QuantizedMatrix& gate_weight = layer.ffn_gate_weight;
    const QuantizedMatrix& up_weight = layer.ffn_up_weight;
    const QuantizedMatrix& down_weight = layer.ffn_down_weight;
    const DeviceVector gate = encode_vector_product(
        execution, gate_weight, ffn_input, true, nullptr, 1.0f,
        ActivationSlot::Gate,
        "resident decode FFN gate projection");
    const DeviceVector up = encode_vector_product(
        execution, up_weight, ffn_input, true, nullptr, 1.0f,
        ActivationSlot::Up,
        "resident decode FFN up projection");
    const DeviceVector activated = encode_binary_vector(
        execution, gate, up, swiglu_pipeline, ActivationSlot::Gate,
        "resident decode SwiGLU");
    const DeviceVector down = encode_vector_product(
        execution, down_weight, activated, true, nullptr, 1.0f,
        ActivationSlot::Projection,
        "resident decode FFN down projection");
    return encode_binary_vector(
        execution, after_attention, down, residual_pipeline,
        output_slot,
        "resident decode FFN residual");
}

void MetalLLM::Impl::upload_model(MetalRawModel& model) {
    require_available();
    if (model.config().layer_count == 0) {
        throw std::invalid_argument(
            "Metal model preparation requires at least one layer");
    }

    @autoreleasepool {
        PreparedModel next;
        const auto upload_matrix_counted =
            [&](RawTensor tensor, const char* name) -> QuantizedMatrix {
                const QuantizedMatrix uploaded = upload_matrix(tensor, name);
                next.weight_bytes = checked_add_size(
                    next.weight_bytes,
                    static_cast<size_t>([uploaded.buffer length]),
                    "Metal weight storage");
                return uploaded;
            };
        const auto upload_vector_counted =
            [&](RawTensor tensor, const char* name) -> DeviceVector {
                const DeviceVector uploaded = upload_vector(tensor, name);
                next.weight_bytes = checked_add_size(
                    next.weight_bytes,
                    static_cast<size_t>([uploaded.buffer length]),
                    "Metal weight storage");
                return uploaded;
            };

        // Each load returns one raw tensor. Its CPU bytes are released at the
        // end of the expression after make_buffer() copies them, so startup
        // never holds a full duplicate model or any dequantized matrices.
        next.token_embedding_weight = upload_matrix_counted(
            model.load_token_embedding(), "Metal token embedding weight");
        next.output_norm_weight = upload_vector_counted(
            model.load_output_norm(), "Metal output norm weight");
        next.output_weight = upload_matrix_counted(
            model.load_output_weight(), "Metal output weight");
        next.layers.reserve(model.config().layer_count);
        for (size_t layer_index = 0;
             layer_index < model.config().layer_count;
             ++layer_index) {
            MetalRawLayer layer = model.load_layer(layer_index);
            PreparedLayer prepared_layer;
            prepared_layer.attn_norm_weight = upload_vector_counted(
                std::move(layer.attn_norm_weight),
                "Metal attention norm weight");
            prepared_layer.attn_q_weight = upload_matrix_counted(
                std::move(layer.attn_q_weight),
                "Metal Q projection weight");
            prepared_layer.attn_q_bias = upload_vector_counted(
                std::move(layer.attn_q_bias), "Metal Q projection bias");
            prepared_layer.attn_k_weight = upload_matrix_counted(
                std::move(layer.attn_k_weight),
                "Metal K projection weight");
            prepared_layer.attn_k_bias = upload_vector_counted(
                std::move(layer.attn_k_bias), "Metal K projection bias");
            prepared_layer.attn_v_weight = upload_matrix_counted(
                std::move(layer.attn_v_weight),
                "Metal V projection weight");
            prepared_layer.attn_v_bias = upload_vector_counted(
                std::move(layer.attn_v_bias), "Metal V projection bias");
            prepared_layer.attn_output_weight = upload_matrix_counted(
                std::move(layer.attn_output_weight),
                "Metal attention output weight");
            prepared_layer.ffn_norm_weight = upload_vector_counted(
                std::move(layer.ffn_norm_weight), "Metal FFN norm weight");
            prepared_layer.ffn_gate_weight = upload_matrix_counted(
                std::move(layer.ffn_gate_weight), "Metal FFN gate weight");
            prepared_layer.ffn_down_weight = upload_matrix_counted(
                std::move(layer.ffn_down_weight), "Metal FFN down weight");
            prepared_layer.ffn_up_weight = upload_matrix_counted(
                std::move(layer.ffn_up_weight), "Metal FFN up weight");
            next.layers.push_back(std::move(prepared_layer));
        }
        model.validate_all_tensors_loaded();
        if (next.weight_bytes != model.config().stored_weight_bytes) {
            throw std::runtime_error(
                "Metal uploaded weight bytes do not match GGUF tensor bytes");
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
    size_t next_kv_cache_bytes = 0;
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
            next_kv_cache_bytes = checked_add_size(
                next_kv_cache_bytes, bytes, "Metal KV-cache allocation");
            next_kv_cache_bytes = checked_add_size(
                next_kv_cache_bytes, bytes, "Metal KV-cache allocation");
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
    kv_cache_bytes = next_kv_cache_bytes;
    max_sequence = capacity;
    sequence_length = 0;
}

void MetalLLM::Impl::install_activation_arena(
    const ActivationArenaPlan& plan,
    ActivationArenaPhase phase,
    const char* operation) {
    require_available();
    if (!prepared.ready || prepared.layers.empty()) {
        throw std::runtime_error(
            "Metal weights must be uploaded before allocating activation arena");
    }
    if (plan.bytes == 0 || phase == ActivationArenaPhase::None) {
        throw std::invalid_argument(
            "Metal activation arena plan or phase is invalid");
    }
    if (activation_arena != nil ||
        activation_phase != ActivationArenaPhase::None) {
        throw std::logic_error(
            "Metal activation arena is already active");
    }
    const NSUInteger max_buffer_length = [device maxBufferLength];
    if (max_buffer_length != 0 &&
        plan.bytes > static_cast<size_t>(max_buffer_length)) {
        throw std::length_error(
            std::string(operation) +
            " exceeds the device maximum buffer length");
    }

    id<MTLBuffer> next_buffer = [device
        newBufferWithLength:plan.bytes
                    options:MTLResourceStorageModePrivate];
    if (next_buffer == nil) {
        throw std::runtime_error(
            std::string("Metal failed to allocate ") + operation);
    }
    activation_arena = next_buffer;
    activation_plan = plan;
    activation_phase = phase;
    turn_peak_activation_bytes = std::max(
        turn_peak_activation_bytes, plan.bytes);
}

void MetalLLM::Impl::prepare_prefill_activation_arena(
    size_t prompt_tokens) {
    if (prompt_tokens == 0 || prompt_tokens > max_sequence) {
        throw std::invalid_argument(
            "Metal prefill arena prompt length is invalid");
    }
    install_activation_arena(
        make_prefill_activation_plan(prompt_tokens),
        ActivationArenaPhase::Prefill,
        "private prefill activation arena");
}

void MetalLLM::Impl::ensure_decode_activation_arena() {
    if (activation_phase == ActivationArenaPhase::Decode) {
        if (activation_arena == nil || activation_plan.bytes == 0) {
            throw std::logic_error(
                "Metal decode activation arena state is inconsistent");
        }
        return;
    }
    if (activation_phase != ActivationArenaPhase::None ||
        activation_arena != nil) {
        throw std::logic_error(
            "Metal decode cannot replace an active prefill arena");
    }
    install_activation_arena(
        make_decode_activation_plan(), ActivationArenaPhase::Decode,
        "private decode activation arena");
}

void MetalLLM::Impl::release_activation_arena() noexcept {
    activation_arena = nil;
    activation_plan = {};
    activation_phase = ActivationArenaPhase::None;
}

void MetalLLM::Impl::load_model(const std::string& gguf_path,
                                size_t capacity) {
    require_available();
    MetalRawModel model(gguf_path);
    MetalModelConfig loaded_config = model.config();
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

    config = std::move(loaded_config);
    upload_model(model);
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
    // Decode keeps its small arena between generated tokens. Every public
    // operation is synchronous, so reset can release it before the next prompt.
    impl_->release_activation_arena();
    impl_->turn_peak_activation_bytes = 0;
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
        impl_->prepare_prefill_activation_arena(token_ids.size());
        struct PrefillArenaRelease {
            Impl* impl = nullptr;
            ~PrefillArenaRelease() {
                impl->release_activation_arena();
            }
        } arena_release{impl_.get()};
        Impl::AsyncExecution execution = impl_->begin_async(
            "prefill", token_ids.size());
        const Impl::QuantizedMatrix& embedding_weight =
            impl_->prepared.token_embedding_weight;
        const id<MTLBuffer> token_buffer = impl_->async_token_ids(
            execution, token_ids);
        Impl::DeviceMatrix hidden = impl_->encode_embedding(
            execution, token_buffer, embedding_weight, token_ids.size(),
            impl_->config.vocabulary_size, Impl::ActivationSlot::HiddenA,
            "resident prefill embedding");

        for (size_t layer_index = 0;
             layer_index < impl_->prepared.layers.size();
             ++layer_index) {
            execution.current_layer = layer_index;
            const Impl::PreparedLayer& layer =
                impl_->prepared.layers[layer_index];
            const Impl::KVCacheLayer& storage =
                impl_->kv_cache[layer_index];
            hidden = impl_->encode_prefill_layer(
                execution, hidden, layer, impl_->config.norm_epsilon,
                impl_->config.rotary_dimension, impl_->config.rope_theta,
                impl_->config.head_size,
                storage.key, storage.value, storage.kv_dimension, 0,
                layer_index % 2 == 0
                    ? Impl::ActivationSlot::HiddenB
                    : Impl::ActivationSlot::HiddenA);
        }
        execution.current_layer = kNoProfileIndex;

        const Impl::DeviceVector& output_norm =
            impl_->prepared.output_norm_weight;
        const Impl::DeviceMatrix final_norm = impl_->encode_rmsnorm_matrix(
            execution, hidden,
            output_norm, impl_->config.norm_epsilon,
            Impl::ActivationSlot::Norm,
            "resident prefill final RMSNorm");
        const size_t last_row_offset = checked_elements(
            final_norm.rows - 1, final_norm.stride,
            "Resident prefill final row offset");
        const Impl::DeviceVector last_hidden = {
            final_norm.buffer, final_norm.cols,
            final_norm.offset_elements + last_row_offset};
        const Impl::QuantizedMatrix& output_weight =
            impl_->prepared.output_weight;
        const Impl::DeviceVector logits = impl_->encode_vector_product(
            execution, output_weight, last_hidden, true, nullptr, 1.0f,
            Impl::ActivationSlot::Logits,
            "resident prefill LM head");
        const id<MTLBuffer> token_output = impl_->encode_argmax(
            execution, logits, logits.length,
            "resident prefill argmax");

        impl_->finish_async(execution, "Metal resident prefill");
        const uint32_t token = *static_cast<const uint32_t*>(
            [token_output contents]);
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
        // The first decode step allocates one-row activation slots and one
        // [1, max_sequence] score slot. Later decode steps reuse this buffer.
        impl_->ensure_decode_activation_arena();
        Impl::AsyncExecution execution = impl_->begin_async(
            "decode", position + 1);
        const Impl::QuantizedMatrix& embedding_weight =
            impl_->prepared.token_embedding_weight;
        const std::vector<int32_t> one_token{token_id};
        const id<MTLBuffer> token_buffer = impl_->async_token_ids(
            execution, one_token);
        const Impl::DeviceMatrix embedded = impl_->encode_embedding(
            execution, token_buffer, embedding_weight, 1,
            impl_->config.vocabulary_size, Impl::ActivationSlot::HiddenA,
            "resident decode embedding");
        Impl::DeviceVector hidden = {
            embedded.buffer, embedded.cols, embedded.offset_elements};

        for (size_t layer_index = 0;
             layer_index < impl_->prepared.layers.size();
             ++layer_index) {
            execution.current_layer = layer_index;
            const Impl::PreparedLayer& layer =
                impl_->prepared.layers[layer_index];
            const Impl::KVCacheLayer& storage =
                impl_->kv_cache[layer_index];
            hidden = impl_->encode_decode_layer(
                execution, hidden, layer, impl_->config.norm_epsilon,
                impl_->config.rotary_dimension, impl_->config.rope_theta,
                impl_->config.head_size,
                storage.key, storage.value, storage.kv_dimension, position,
                layer_index % 2 == 0
                    ? Impl::ActivationSlot::HiddenB
                    : Impl::ActivationSlot::HiddenA);
        }
        execution.current_layer = kNoProfileIndex;

        const Impl::DeviceVector& output_norm =
            impl_->prepared.output_norm_weight;
        const Impl::DeviceVector final_norm = impl_->encode_rmsnorm_vector(
            execution, hidden, output_norm, impl_->config.norm_epsilon,
            Impl::ActivationSlot::Norm,
            "resident decode final RMSNorm");
        const Impl::QuantizedMatrix& output_weight =
            impl_->prepared.output_weight;
        const Impl::DeviceVector logits = impl_->encode_vector_product(
            execution, output_weight, final_norm, true, nullptr, 1.0f,
            Impl::ActivationSlot::Logits,
            "resident decode LM head");
        const id<MTLBuffer> token_output = impl_->encode_argmax(
            execution, logits, logits.length,
            "resident decode argmax");

        impl_->finish_async(execution, "Metal resident decode");
        const uint32_t token = *static_cast<const uint32_t*>(
            [token_output contents]);
        if (token > static_cast<uint32_t>(
                        std::numeric_limits<int32_t>::max())) {
            throw std::overflow_error(
                "Metal resident decode token id does not fit in int32_t");
        }
        impl_->sequence_length = position + 1;
        return static_cast<int32_t>(token);
    }
}

void MetalLLM::enable_profiling(const std::string& csv_path,
                                bool detailed_kernel_timestamps) {
    if (impl_ == nullptr) {
        throw std::runtime_error("Metal model is not initialized");
    }
    auto profiler = std::make_unique<Profiler>(csv_path);
    auto metal_profiler = std::make_unique<MetalProfiler>(
        csv_path, detailed_kernel_timestamps);

    impl_->timestamp_counter_set = nil;
    impl_->counter_sampling_mode = Impl::CounterSamplingMode::None;
    impl_->timestamp_counter_status = detailed_kernel_timestamps
        ? "timestamp counter unavailable" : "not requested";
    if (detailed_kernel_timestamps) {
        @autoreleasepool {
            if ([impl_->device supportsCounterSampling:
                    MTLCounterSamplingPointAtDispatchBoundary]) {
                impl_->counter_sampling_mode =
                    Impl::CounterSamplingMode::DispatchBoundary;
            } else if ([impl_->device supportsCounterSampling:
                           MTLCounterSamplingPointAtStageBoundary]) {
                impl_->counter_sampling_mode =
                    Impl::CounterSamplingMode::StageBoundary;
            } else {
                impl_->timestamp_counter_status =
                    "compute counter sampling unsupported";
            }
            if (impl_->counter_sampling_mode !=
                Impl::CounterSamplingMode::None) {
                NSArray<id<MTLCounterSet>>* counter_sets =
                    [impl_->device counterSets];
                for (id<MTLCounterSet> counter_set in counter_sets) {
                    if ([[counter_set name]
                            isEqualToString:MTLCommonCounterSetTimestamp]) {
                        impl_->timestamp_counter_set = counter_set;
                        impl_->timestamp_counter_status = "available";
                        break;
                    }
                }
                if (impl_->timestamp_counter_set == nil) {
                    impl_->timestamp_counter_status =
                        "timestamp counter set missing";
                }
            }
        }
    }
    impl_->profiler = std::move(profiler);
    impl_->metal_profiler = std::move(metal_profiler);
}

void MetalLLM::disable_profiling() {
    if (impl_ != nullptr) {
        impl_->profiler.reset();
        impl_->metal_profiler.reset();
        impl_->timestamp_counter_set = nil;
        impl_->counter_sampling_mode = Impl::CounterSamplingMode::None;
        impl_->timestamp_counter_status = "not requested";
    }
}

const MetalModelConfig& MetalLLM::config() const {
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

MemoryStats MetalLLM::memory_stats() const noexcept {
    if (impl_ == nullptr) {
        return {};
    }
    return {
        static_cast<std::uint64_t>(impl_->prepared.weight_bytes),
        static_cast<std::uint64_t>(impl_->kv_cache_bytes),
        static_cast<std::uint64_t>(impl_->turn_peak_activation_bytes),
        false,
    };
}

} // namespace llm

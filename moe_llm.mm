#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "moe_llm.h"

#include "moe_model.h"
#include "profiler.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <mach-o/dyld.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace llm {
namespace {

constexpr std::uint32_t kTileSize = 16;
constexpr std::uint32_t kReductionThreads = 256;
constexpr std::uint32_t kArgmaxElementsPerGroup = 1024;
constexpr std::size_t kDefaultMoeContext = 1024;
constexpr std::uint32_t kExpertBundleVersion = 1;
constexpr std::array<char, 8> kExpertBundleMagic{
    {'M', 'Y', 'L', 'L', 'M', 'E', 'X', '1'}};

struct ExpertBundleHeader {
    char magic[8]{};
    std::uint32_t version = 0;
    std::uint32_t header_bytes = 0;
    std::uint64_t model_size = 0;
    std::int64_t model_mtime_seconds = 0;
    std::int64_t model_mtime_nanoseconds = 0;
    std::uint32_t layer_count = 0;
    std::uint32_t expert_count = 0;
    std::uint64_t entry_count = 0;
    std::uint64_t index_offset = 0;
    std::uint64_t data_offset = 0;
};

struct ExpertBundleEntry {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::uint32_t ready = 0;
    std::uint32_t reserved = 0;
};

static_assert(sizeof(ExpertBundleHeader) == 72);
static_assert(sizeof(ExpertBundleEntry) == 24);

struct MetalMatmulParamsHost {
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    std::uint32_t lhs_stride = 0;
    std::uint32_t rhs_stride = 0;
    std::uint32_t lhs_transposed = 0;
    std::uint32_t rhs_transposed = 0;
    std::uint32_t has_bias = 0;
    float scale = 1.0f;
};

struct MetalGemvParamsHost {
    std::uint32_t output_size = 0;
    std::uint32_t input_size = 0;
    std::uint32_t matrix_stride = 0;
    std::uint32_t matrix_transposed = 0;
    std::uint32_t has_bias = 0;
    float scale = 1.0f;
};

struct MetalQuantizedProductParamsHost {
    std::uint32_t m = 0;
    std::uint32_t n = 0;
    std::uint32_t k = 0;
    std::uint32_t activation_stride = 0;
    std::uint32_t weight_row_bytes = 0;
    std::uint32_t has_bias = 0;
    float scale = 1.0f;
};

struct MetalElementwiseParamsHost {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
};

struct MetalRmsNormParamsHost {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    float epsilon = 0.0f;
};

struct MetalRopeHeadsParamsHost {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rotary_dimension = 0;
    std::uint32_t position = 0;
    float theta = 0.0f;
};

struct MetalEmbeddingParamsHost {
    std::uint32_t sequence_length = 0;
    std::uint32_t embedding_size = 0;
    std::uint32_t vocabulary_size = 0;
    std::uint32_t weight_row_bytes = 0;
};

struct MetalKVCacheWriteParamsHost {
    std::uint32_t source_rows = 0;
    std::uint32_t source_cols = 0;
    std::uint32_t cache_stride = 0;
    std::uint32_t position = 0;
};

struct MetalKVCacheQKParamsHost {
    std::uint32_t query_rows = 0;
    std::uint32_t key_length = 0;
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t cache_stride = 0;
    std::uint32_t query_stride = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t query_position = 0;
    float scale = 1.0f;
};

struct MetalKVCacheAVParamsHost {
    std::uint32_t query_rows = 0;
    std::uint32_t key_length = 0;
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t cache_stride = 0;
    std::uint32_t score_stride = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t output_stride = 0;
};

struct MetalArgmaxParamsHost {
    std::uint32_t length = 0;
    std::uint32_t elements_per_group = 0;
};

struct MetalSplitQGateParamsHost {
    std::uint32_t rows = 0;
    std::uint32_t head_count = 0;
    std::uint32_t head_dim = 0;
};

struct MetalBroadcastParamsHost {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
};

struct MetalRowScaleParamsHost {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t weight_stride = 0;
    std::uint32_t weight_column = 0;
};

struct MetalHeadNormParamsHost {
    std::uint32_t tokens = 0;
    std::uint32_t head_count = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t input_token_stride = 0;
    float epsilon = 0.0f;
};

struct MetalDepthwiseConvParamsHost {
    std::uint32_t tokens = 0;
    std::uint32_t channels = 0;
    std::uint32_t kernel_size = 0;
};

struct MetalGdnParamsHost {
    std::uint32_t tokens = 0;
    std::uint32_t key_head_count = 0;
    std::uint32_t value_head_count = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t q_stride = 0;
    std::uint32_t k_stride = 0;
    std::uint32_t v_stride = 0;
    std::uint32_t output_stride = 0;
};

struct MetalTopKParamsHost {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t k = 0;
};

struct MetalExpertProductParamsHost {
    std::uint32_t tokens = 0;
    std::uint32_t routes = 0;
    std::uint32_t output_size = 0;
    std::uint32_t input_size = 0;
    std::uint32_t activation_stride = 0;
    std::uint32_t weight_row_bytes = 0;
    std::uint32_t expert_stride_bytes = 0;
    std::uint32_t activation_is_routed = 0;
};

struct MetalExpertWeightedReduceParamsHost {
    std::uint32_t tokens = 0;
    std::uint32_t routes = 0;
    std::uint32_t cols = 0;
};

static_assert(sizeof(MetalMatmulParamsHost) == 36);
static_assert(sizeof(MetalGemvParamsHost) == 24);
static_assert(sizeof(MetalQuantizedProductParamsHost) == 28);
static_assert(sizeof(MetalElementwiseParamsHost) == 8);
static_assert(sizeof(MetalRmsNormParamsHost) == 12);
static_assert(sizeof(MetalRopeHeadsParamsHost) == 24);
static_assert(sizeof(MetalEmbeddingParamsHost) == 16);
static_assert(sizeof(MetalKVCacheWriteParamsHost) == 16);
static_assert(sizeof(MetalKVCacheQKParamsHost) == 36);
static_assert(sizeof(MetalKVCacheAVParamsHost) == 32);
static_assert(sizeof(MetalArgmaxParamsHost) == 8);
static_assert(sizeof(MetalSplitQGateParamsHost) == 12);
static_assert(sizeof(MetalBroadcastParamsHost) == 8);
static_assert(sizeof(MetalRowScaleParamsHost) == 16);
static_assert(sizeof(MetalHeadNormParamsHost) == 20);
static_assert(sizeof(MetalDepthwiseConvParamsHost) == 12);
static_assert(sizeof(MetalGdnParamsHost) == 32);
static_assert(sizeof(MetalTopKParamsHost) == 12);
static_assert(sizeof(MetalExpertProductParamsHost) == 32);
static_assert(sizeof(MetalExpertWeightedReduceParamsHost) == 12);

std::size_t checked_mul(std::size_t left,
                        std::size_t right,
                        const char* name) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(std::string(name) + " size overflows");
    }
    return left * right;
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        const char* name) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error(std::string(name) + " size overflows");
    }
    return left + right;
}

std::size_t float_bytes(std::size_t elements, const char* name) {
    return checked_mul(elements, sizeof(float), name);
}

std::size_t align_up(std::size_t value,
                     std::size_t alignment,
                     const char* name) {
    const std::size_t remainder = value % alignment;
    return remainder == 0
        ? value
        : checked_add(value, alignment - remainder, name);
}

std::uint32_t to_uint(std::size_t value, const char* name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error(std::string(name) + " does not fit Metal uint");
    }
    return static_cast<std::uint32_t>(value);
}

std::uint64_t monotonic_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct FileIdentity {
    std::uint64_t size = 0;
    std::int64_t mtime_seconds = 0;
    std::int64_t mtime_nanoseconds = 0;
};

bool query_file_identity(const std::filesystem::path& path,
                         FileIdentity& identity) noexcept {
    struct stat status {};
    if (::stat(path.c_str(), &status) != 0 || status.st_size < 0) return false;
    identity.size = static_cast<std::uint64_t>(status.st_size);
#if defined(__APPLE__)
    identity.mtime_seconds = status.st_mtimespec.tv_sec;
    identity.mtime_nanoseconds = status.st_mtimespec.tv_nsec;
#else
    identity.mtime_seconds = status.st_mtim.tv_sec;
    identity.mtime_nanoseconds = status.st_mtim.tv_nsec;
#endif
    return true;
}

bool read_all_at(int descriptor,
                 void* destination,
                 std::size_t bytes,
                 std::uint64_t offset) noexcept {
    auto* output = static_cast<std::uint8_t*>(destination);
    while (bytes != 0) {
        if (offset > static_cast<std::uint64_t>(
                         std::numeric_limits<off_t>::max())) {
            return false;
        }
        const std::size_t maximum = static_cast<std::size_t>(
            std::numeric_limits<ssize_t>::max());
        const std::size_t chunk = std::min(bytes, maximum);
        const ssize_t count = ::pread(
            descriptor, output, chunk, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        const std::size_t consumed = static_cast<std::size_t>(count);
        output += consumed;
        bytes -= consumed;
        offset += consumed;
    }
    return true;
}

bool write_all_at(int descriptor,
                  const void* source,
                  std::size_t bytes,
                  std::uint64_t offset) noexcept {
    const auto* input = static_cast<const std::uint8_t*>(source);
    while (bytes != 0) {
        if (offset > static_cast<std::uint64_t>(
                         std::numeric_limits<off_t>::max())) {
            return false;
        }
        const std::size_t maximum = static_cast<std::size_t>(
            std::numeric_limits<ssize_t>::max());
        const std::size_t chunk = std::min(bytes, maximum);
        const ssize_t count = ::pwrite(
            descriptor, input, chunk, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        const std::size_t consumed = static_cast<std::size_t>(count);
        input += consumed;
        bytes -= consumed;
        offset += consumed;
    }
    return true;
}

std::uint64_t fnv1a_append(std::uint64_t hash,
                           const void* data,
                           std::size_t bytes) noexcept {
    const auto* input = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0; index < bytes; ++index) {
        hash ^= input[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string model_fingerprint(const std::filesystem::path& path,
                              const FileIdentity& identity) {
    std::error_code error;
    std::filesystem::path normalized = std::filesystem::weakly_canonical(
        path, error);
    if (error) normalized = std::filesystem::absolute(path, error);
    if (error) normalized = path.lexically_normal();
    const std::string name = normalized.string();
    std::uint64_t hash = 1469598103934665603ull;
    hash = fnv1a_append(hash, name.data(), name.size());
    hash = fnv1a_append(hash, &identity, sizeof(identity));
    std::ostringstream result;
    result << std::hex << std::setfill('0') << std::setw(16) << hash;
    return result.str();
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    return std::string(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string locate_shader(const std::string& requested_path) {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    if (!requested_path.empty()) candidates.emplace_back(requested_path);
    if (const char* environment = std::getenv("MYLLM_METAL_SHADER");
        environment != nullptr && environment[0] != '\0') {
        candidates.emplace_back(environment);
    }
    candidates.emplace_back("metal_llm.metal");
    char executable[PATH_MAX] = {};
    std::uint32_t length = sizeof(executable);
    if (_NSGetExecutablePath(executable, &length) == 0) {
        candidates.emplace_back(
            fs::path(executable).parent_path() / "metal_llm.metal");
    }
    candidates.emplace_back(fs::path(__FILE__).parent_path() / "metal_llm.metal");
    for (const fs::path& candidate : candidates) {
        std::string source = read_text_file(candidate);
        if (!source.empty()) return source;
    }
    throw std::runtime_error(
        "Cannot find metal_llm.metal; set MYLLM_METAL_SHADER to its path");
}

std::string error_description(NSError* error) {
    if (error == nil || [error localizedDescription] == nil) {
        return "unknown Metal error";
    }
    const char* text = [[error localizedDescription] UTF8String];
    return text == nullptr ? "unknown Metal error" : std::string(text);
}

[[noreturn]] void throw_metal(const char* operation, NSError* error) {
    throw std::runtime_error(
        std::string(operation) + ": " + error_description(error));
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> library,
                                          const char* function_name) {
    NSString* name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        throw std::runtime_error(
            std::string("Metal kernel is missing: ") + function_name);
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) throw_metal(function_name, error);
    return pipeline;
}

} // namespace

struct MoeLLM::Impl {
    struct Matrix {
        id<MTLBuffer> buffer = nil;
        std::size_t rows = 0;
        std::size_t cols = 0;
        std::size_t stride = 0;
        std::size_t offset = 0;
    };

    struct Weight {
        id<MTLBuffer> buffer = nil;
        MetalGgmlType type = MetalGgmlType::F32;
        std::size_t rows = 0;
        std::size_t cols = 0;
        std::size_t row_bytes = 0;
    };

    struct Vector {
        id<MTLBuffer> buffer = nil;
        std::size_t length = 0;
    };

    struct ExpertWeight {
        id<MTLBuffer> buffer = nil;
        RawTensorDescriptor source;
        MetalGgmlType type = MetalGgmlType::F32;
        std::size_t slots = 0;
        std::size_t rows = 0;
        std::size_t cols = 0;
        std::size_t row_bytes = 0;
        std::size_t expert_stride_bytes = 0;
    };

    struct FullAttention {
        Weight q_gate;
        Weight key;
        Weight value;
        Vector query_norm;
        Vector key_norm;
        Weight output;
        id<MTLBuffer> key_cache = nil;
        id<MTLBuffer> value_cache = nil;
    };

    struct DeltaNet {
        Weight qkv;
        Weight z;
        Weight alpha;
        Weight beta;
        Weight convolution;
        Vector time_bias;
        Vector a;
        Vector state_norm;
        Weight output;
        id<MTLBuffer> recurrent_state = nil;
        id<MTLBuffer> convolution_history = nil;
    };

    struct Experts {
        Weight router;
        ExpertWeight gate;
        ExpertWeight up;
        ExpertWeight down;
        Weight shared_router;
        Weight shared_gate;
        Weight shared_up;
        Weight shared_down;
        std::vector<std::int32_t> expert_to_slot;
        std::vector<std::int32_t> slot_to_expert;
        std::vector<std::uint64_t> last_used;
        std::uint64_t lru_clock = 0;
    };

    struct Layer {
        MoeLayerKind kind = MoeLayerKind::GatedDeltaNet;
        Vector attention_norm;
        Vector post_attention_norm;
        FullAttention attention;
        DeltaNet delta;
        Experts experts;
    };

    enum class Slot : std::uint8_t {
        HiddenA,
        HiddenB,
        HiddenC,
        HiddenScratch,
        Norm,
        WideA,
        WideB,
        FeatureA,
        FeatureB,
        FeatureC,
        FeatureD,
        KVA,
        KVB,
        KVC,
        SmallA,
        SmallB,
        SmallC,
        ExpertGate,
        ExpertUp,
        ExpertActivation,
        ExpertDown,
        RouteIds,
        RouteWeights,
        Scores,
        Logits,
        ArgmaxValues,
        ArgmaxIndices,
        Count,
    };

    struct Region {
        std::size_t offset_bytes = 0;
        std::size_t capacity_bytes = 0;
    };

    struct ArenaPlan {
        std::array<Region, static_cast<std::size_t>(Slot::Count)> regions{};
        std::size_t bytes = 0;
    };

    struct Execution {
        id<MTLCommandBuffer> command = nil;
        id<MTLComputeCommandEncoder> encoder = nil;
        std::vector<id<MTLBuffer>> io_buffers;
        std::vector<MetalKernelProfileRecord> kernel_records;
        std::string phase;
        std::size_t sequence_tokens = 0;
        std::size_t command_index = 0;
        std::uint64_t begin_ns = 0;
        std::uint64_t encode_begin_ns = 0;
        std::uint64_t command_create_ns = 0;
        std::uint64_t kernel_count = 0;
        std::uint64_t threadgroups = 0;
        std::uint64_t dispatched_threads = 0;
        std::uint64_t one_group_kernels = 0;
    };

    struct ExpertBundleSidecar {
        int descriptor = -1;
        std::filesystem::path path;
        ExpertBundleHeader header;
        std::vector<ExpertBundleEntry> entries;
        std::uint64_t append_offset = 0;
        std::vector<std::uint8_t> scratch;
        const std::uint8_t* mapping = nullptr;
        std::size_t mapping_size = 0;

        ~ExpertBundleSidecar() {
            if (mapping != nullptr) {
                ::munmap(const_cast<std::uint8_t*>(mapping), mapping_size);
            }
            if (descriptor >= 0) ::close(descriptor);
        }

        bool enabled() const noexcept { return descriptor >= 0; }

        void disable() noexcept {
            if (mapping != nullptr) {
                ::munmap(const_cast<std::uint8_t*>(mapping), mapping_size);
            }
            mapping = nullptr;
            mapping_size = 0;
            if (descriptor >= 0) ::close(descriptor);
            descriptor = -1;
            path.clear();
            header = {};
            entries.clear();
            append_offset = 0;
        }
    };

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLBuffer> dummy = nil;

    id<MTLComputePipelineState> gemm_f32 = nil;
    id<MTLComputePipelineState> gevm_f32 = nil;
    id<MTLComputePipelineState> gemm_q5_0 = nil;
    id<MTLComputePipelineState> gemm_q8_0 = nil;
    id<MTLComputePipelineState> gemm_q4_k = nil;
    id<MTLComputePipelineState> gemm_q5_k = nil;
    id<MTLComputePipelineState> gemm_q6_k = nil;
    id<MTLComputePipelineState> gevm_q5_0 = nil;
    id<MTLComputePipelineState> gevm_q8_0 = nil;
    id<MTLComputePipelineState> gevm_q4_k = nil;
    id<MTLComputePipelineState> gevm_q5_k = nil;
    id<MTLComputePipelineState> gevm_q6_k = nil;
    id<MTLComputePipelineState> embedding_f32 = nil;
    id<MTLComputePipelineState> embedding_q5_0 = nil;
    id<MTLComputePipelineState> embedding_q8_0 = nil;
    id<MTLComputePipelineState> embedding_q4_k = nil;
    id<MTLComputePipelineState> embedding_q6_k = nil;
    id<MTLComputePipelineState> rmsnorm = nil;
    id<MTLComputePipelineState> softmax = nil;
    id<MTLComputePipelineState> swiglu = nil;
    id<MTLComputePipelineState> residual = nil;
    id<MTLComputePipelineState> rope = nil;
    id<MTLComputePipelineState> kv_write = nil;
    id<MTLComputePipelineState> kv_qk = nil;
    id<MTLComputePipelineState> kv_av = nil;
    id<MTLComputePipelineState> argmax_partial = nil;
    id<MTLComputePipelineState> argmax_finalize = nil;
    id<MTLComputePipelineState> split_q_gate = nil;
    id<MTLComputePipelineState> sigmoid = nil;
    id<MTLComputePipelineState> silu = nil;
    id<MTLComputePipelineState> softplus = nil;
    id<MTLComputePipelineState> exponential = nil;
    id<MTLComputePipelineState> multiply = nil;
    id<MTLComputePipelineState> add_channel = nil;
    id<MTLComputePipelineState> multiply_channel = nil;
    id<MTLComputePipelineState> row_scale = nil;
    id<MTLComputePipelineState> l2norm_heads = nil;
    id<MTLComputePipelineState> depthwise_conv = nil;
    id<MTLComputePipelineState> commit_conv_history = nil;
    id<MTLComputePipelineState> gdn = nil;
    id<MTLComputePipelineState> topk = nil;
    id<MTLComputePipelineState> topk_renorm = nil;
    id<MTLComputePipelineState> expert_q4_k = nil;
    id<MTLComputePipelineState> expert_q6_k = nil;
    id<MTLComputePipelineState> expert_weighted_reduce = nil;

    std::string device_name;
    MetalModelConfig config;
    std::unique_ptr<MoeRawModel> raw_model;
    Weight embedding;
    Vector output_norm;
    Weight output;
    std::vector<Layer> layers;
    id<MTLBuffer> arena = nil;
    ArenaPlan arena_plan;
    std::size_t weight_bytes = 0;
    std::size_t kv_cache_bytes = 0;
    std::size_t recurrent_state_bytes = 0;
    std::size_t turn_peak_arena_bytes = 0;
    std::size_t capacity = 0;
    std::size_t sequence_length = 0;
    std::size_t expert_cache_count = 0;
    std::uint64_t turn_expert_cache_hits = 0;
    std::uint64_t turn_expert_cache_misses = 0;
    ExpertBundleSidecar expert_sidecar;
    std::unique_ptr<Profiler> profiler;
    std::unique_ptr<MetalProfiler> metal_profiler;

    static constexpr std::size_t kArenaAlignment = 256;

    explicit Impl(const std::string& shader_path) {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
                if ([devices count] != 0) device = [devices objectAtIndex:0];
            }
            if (device == nil) {
                device_name = "unavailable (no Metal device)";
                return;
            }
            const char* name = [[device name] UTF8String];
            device_name = name == nullptr ? "Metal device" : std::string(name);
            queue = [device newCommandQueue];
            if (queue == nil) {
                throw std::runtime_error("Metal could not create a command queue");
            }
            dummy = [device newBufferWithLength:sizeof(float)
                                         options:MTLResourceStorageModeShared];
            if (dummy == nil) {
                throw std::runtime_error("Metal could not create a dummy buffer");
            }
            *static_cast<float*>([dummy contents]) = 0.0f;

            const std::string source = locate_shader(shader_path);
            NSString* source_string = [[NSString alloc]
                initWithBytes:source.data()
                       length:source.size()
                     encoding:NSUTF8StringEncoding];
            if (source_string == nil) {
                throw std::runtime_error("metal_llm.metal is not valid UTF-8");
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:source_string
                                                          options:nil
                                                            error:&error];
            if (library == nil) throw_metal("compile metal_llm.metal", error);

#define MYLLM_PIPELINE(member, function) \
            member = make_pipeline(device, library, function)
            MYLLM_PIPELINE(gemm_f32, "metal_gemm_f32");
            MYLLM_PIPELINE(gevm_f32, "metal_gevm_f32");
            MYLLM_PIPELINE(gemm_q5_0, "metal_gemm_q5_0_f32");
            MYLLM_PIPELINE(gemm_q8_0, "metal_gemm_q8_0_f32");
            MYLLM_PIPELINE(gemm_q4_k, "metal_gemm_q4_k_f32");
            MYLLM_PIPELINE(gemm_q5_k, "metal_gemm_q5_k_f32");
            MYLLM_PIPELINE(gemm_q6_k, "metal_gemm_q6_k_f32");
            MYLLM_PIPELINE(gevm_q5_0, "metal_gevm_q5_0_f32");
            MYLLM_PIPELINE(gevm_q8_0, "metal_gevm_q8_0_f32");
            MYLLM_PIPELINE(gevm_q4_k, "metal_gevm_q4_k_f32");
            MYLLM_PIPELINE(gevm_q5_k, "metal_gevm_q5_k_f32");
            MYLLM_PIPELINE(gevm_q6_k, "metal_gevm_q6_k_f32");
            MYLLM_PIPELINE(embedding_f32, "metal_embedding_f32");
            MYLLM_PIPELINE(embedding_q5_0, "metal_embedding_q5_0_f32");
            MYLLM_PIPELINE(embedding_q8_0, "metal_embedding_q8_0_f32");
            MYLLM_PIPELINE(embedding_q4_k, "metal_embedding_q4_k_f32");
            MYLLM_PIPELINE(embedding_q6_k, "metal_embedding_q6_k_f32");
            MYLLM_PIPELINE(rmsnorm, "metal_rmsnorm_f32");
            MYLLM_PIPELINE(softmax, "metal_softmax_f32");
            MYLLM_PIPELINE(swiglu, "metal_swiglu_f32");
            MYLLM_PIPELINE(residual, "metal_residual_f32");
            MYLLM_PIPELINE(rope, "metal_rope_heads_f32");
            MYLLM_PIPELINE(kv_write, "metal_kv_cache_write_f32");
            MYLLM_PIPELINE(kv_qk, "metal_kv_cache_qk_f32");
            MYLLM_PIPELINE(kv_av, "metal_kv_cache_av_f32");
            MYLLM_PIPELINE(argmax_partial, "metal_argmax_partial_f32");
            MYLLM_PIPELINE(argmax_finalize, "metal_argmax_finalize_f32");
            MYLLM_PIPELINE(split_q_gate,
                           "metal_split_interleaved_q_gate_f32");
            MYLLM_PIPELINE(sigmoid, "metal_sigmoid_f32");
            MYLLM_PIPELINE(silu, "metal_silu_only_f32");
            MYLLM_PIPELINE(softplus, "metal_softplus_f32");
            MYLLM_PIPELINE(exponential, "metal_exp_f32");
            MYLLM_PIPELINE(multiply, "metal_mul_f32");
            MYLLM_PIPELINE(add_channel, "metal_add_channel_bias_f32");
            MYLLM_PIPELINE(multiply_channel, "metal_mul_channel_f32");
            MYLLM_PIPELINE(row_scale, "metal_row_scale_selected_f32");
            MYLLM_PIPELINE(l2norm_heads, "metal_l2norm_heads_f32");
            MYLLM_PIPELINE(depthwise_conv,
                           "metal_depthwise_conv1d_causal_f32");
            MYLLM_PIPELINE(commit_conv_history,
                           "metal_conv_history_commit_f32");
            MYLLM_PIPELINE(gdn, "metal_gdn_recurrence_f32");
            MYLLM_PIPELINE(topk, "metal_topk_f32");
            MYLLM_PIPELINE(topk_renorm, "metal_topk_renorm_f32");
            MYLLM_PIPELINE(expert_q4_k, "metal_expert_gemm_q4_k_f32");
            MYLLM_PIPELINE(expert_q6_k, "metal_expert_gemm_q6_k_f32");
            MYLLM_PIPELINE(expert_weighted_reduce,
                           "metal_expert_weighted_reduce_f32");
#undef MYLLM_PIPELINE
        }
    }

    void require_available() const {
        if (device == nil || queue == nil || gemm_f32 == nil ||
            gevm_f32 == nil || gdn == nil || expert_q4_k == nil ||
            expert_q6_k == nil || expert_weighted_reduce == nil ||
            argmax_partial == nil || argmax_finalize == nil) {
            throw std::runtime_error(
                "Metal MoE backend is unavailable: " + device_name);
        }
    }

    id<MTLBuffer> make_buffer(const void* data,
                              std::size_t bytes,
                              const char* name) const {
        if (bytes == 0) return nil;
        if (bytes > static_cast<std::size_t>([device maxBufferLength])) {
            throw std::length_error(
                std::string(name) + " exceeds Metal maxBufferLength");
        }
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw std::runtime_error(
                std::string("Metal could not allocate ") + name);
        }
        if (data != nullptr) std::memcpy([buffer contents], data, bytes);
        return buffer;
    }

    Weight upload_weight(const RawTensor& tensor, const char* name) {
        const std::size_t row_bytes = metal_ggml_row_bytes(
            tensor.type, tensor.cols);
        if (tensor.rows == 0 || tensor.cols == 0 ||
            tensor.row_bytes != row_bytes ||
            tensor.data.size() != checked_mul(tensor.rows, row_bytes, name)) {
            throw std::runtime_error(
                std::string(name) + " has invalid GGUF storage");
        }
        switch (tensor.type) {
        case MetalGgmlType::F32:
        case MetalGgmlType::Q5_0:
        case MetalGgmlType::Q8_0:
        case MetalGgmlType::Q4_K:
        case MetalGgmlType::Q5_K:
        case MetalGgmlType::Q6_K:
            break;
        default:
            throw std::runtime_error(
                std::string(name) + " uses unsupported type " +
                metal_ggml_type_name(tensor.type));
        }
        id<MTLBuffer> buffer = make_buffer(
            tensor.data.data(), tensor.data.size(), name);
        weight_bytes = checked_add(
            weight_bytes, tensor.data.size(), "MoE model weights");
        return {buffer, tensor.type, tensor.rows, tensor.cols, row_bytes};
    }

    Vector upload_vector(const RawTensor& tensor, const char* name) {
        if (tensor.type != MetalGgmlType::F32 || tensor.rows != 1 ||
            tensor.row_bytes != float_bytes(tensor.cols, name) ||
            tensor.data.size() != tensor.row_bytes) {
            throw std::runtime_error(
                std::string(name) + " must be a contiguous F32 vector");
        }
        id<MTLBuffer> buffer = make_buffer(
            tensor.data.data(), tensor.data.size(), name);
        weight_bytes = checked_add(
            weight_bytes, tensor.data.size(), "MoE model weights");
        return {buffer, tensor.cols};
    }

    ExpertWeight prepare_expert_cache(const RawTensorDescriptor& tensor,
                                      std::size_t experts,
                                      std::size_t slots,
                                      std::size_t rows,
                                      std::size_t cols,
                                      const char* name) {
        if (tensor.rows != checked_mul(experts, rows, name) ||
            tensor.cols != cols) {
            throw std::runtime_error(
                std::string(name) + " has invalid expert dimensions");
        }
        switch (tensor.type) {
        case MetalGgmlType::Q4_K:
        case MetalGgmlType::Q6_K:
            break;
        default:
            throw std::runtime_error(
                std::string(name) + " uses unsupported type " +
                metal_ggml_type_name(tensor.type));
        }
        const std::size_t row_bytes = metal_ggml_row_bytes(
            tensor.type, tensor.cols);
        if (tensor.row_bytes != row_bytes ||
            tensor.byte_size != checked_mul(tensor.rows, row_bytes, name)) {
            throw std::runtime_error(
                std::string(name) + " has invalid GGUF storage");
        }
        const std::size_t expert_stride = checked_mul(
            rows, row_bytes, "expert weight stride");
        const std::size_t cache_bytes = checked_mul(
            slots, expert_stride, "expert cache buffer");
        id<MTLBuffer> buffer = make_buffer(nullptr, cache_bytes, name);
        weight_bytes = checked_add(
            weight_bytes, cache_bytes, "MoE cached expert weights");
        return {buffer, tensor, tensor.type, slots, rows, cols,
                row_bytes, expert_stride};
    }

    void load_model(const std::string& gguf_path,
                    std::size_t requested_capacity,
                    std::size_t requested_expert_cache_count);
    void reset_state();
    void initialize_expert_sidecar(const std::string& gguf_path) noexcept;
    const std::uint8_t* read_expert_sidecar(
        std::size_t layer_index,
        std::uint32_t expert,
        std::size_t expected_bytes) noexcept;
    void write_expert_sidecar(std::size_t layer_index,
                              std::uint32_t expert,
                              std::size_t bytes) noexcept;
    void install_expert_bundle(std::size_t layer_index,
                               std::uint32_t expert,
                               std::size_t slot);
    void ensure_experts_cached(std::size_t layer_index,
                               std::uint32_t* route_ids,
                               std::size_t route_count);

    static std::size_t slot_index(Slot slot) {
        return static_cast<std::size_t>(slot);
    }

    ArenaPlan make_arena_plan(std::size_t rows,
                              std::size_t score_columns) const;
    void install_arena(const ArenaPlan& plan, const char* phase);
    void release_arena() noexcept {
        arena = nil;
        arena_plan = {};
    }

    Matrix matrix(Slot slot,
                  std::size_t rows,
                  std::size_t cols,
                  const char* name) const {
        if (arena == nil) {
            throw std::runtime_error("MoE activation arena is not allocated");
        }
        const Region& region = arena_plan.regions[slot_index(slot)];
        const std::size_t bytes = float_bytes(checked_mul(rows, cols, name), name);
        if (bytes > region.capacity_bytes) {
            throw std::length_error(
                std::string(name) + " exceeds its activation arena region");
        }
        return {arena, rows, cols, cols,
                region.offset_bytes / sizeof(float)};
    }

    Matrix view(const Matrix& source,
                std::size_t offset,
                std::size_t rows,
                std::size_t cols,
                std::size_t stride,
                const char* name) const {
        if (source.buffer == nil || rows == 0 || cols == 0 || stride < cols) {
            throw std::invalid_argument(
                std::string(name) + " has invalid view dimensions");
        }
        const std::size_t last = checked_add(
            offset, checked_add(checked_mul(rows - 1, stride, name), cols, name),
            name);
        const std::size_t source_span = checked_mul(
            source.rows, source.stride, name);
        if (last > source_span) {
            throw std::out_of_range(std::string(name) + " view exceeds source");
        }
        return {source.buffer, rows, cols, stride, source.offset + offset};
    }

    std::size_t byte_offset(const Matrix& value, const char* name) const {
        return float_bytes(value.offset, name);
    }

    Execution begin_execution(const char* phase,
                              std::size_t sequence_tokens) const;
    void finish_execution(Execution& execution, const char* operation) const;
    id<MTLBuffer> io_buffer(Execution& execution,
                            const void* data,
                            std::size_t bytes,
                            const char* name) const;
    void set_pipeline(Execution& execution,
                      id<MTLComputePipelineState> pipeline,
                      const char* operation,
                      std::size_t grid_x,
                      std::size_t grid_y,
                      std::size_t threads_x,
                      std::size_t threads_y) const;
    void dispatch_flat(Execution& execution,
                       id<MTLComputePipelineState> pipeline,
                       const char* operation,
                       std::size_t count) const;
    void dispatch_rows(Execution& execution,
                       id<MTLComputePipelineState> pipeline,
                       const char* operation,
                       std::size_t rows) const;
    void dispatch_tiles(Execution& execution,
                        id<MTLComputePipelineState> pipeline,
                        const char* operation,
                        std::size_t rows,
                        std::size_t cols) const;

    id<MTLComputePipelineState> product_pipeline(MetalGgmlType type,
                                                 bool vector) const;
    id<MTLComputePipelineState> embedding_pipeline(MetalGgmlType type) const;

    Matrix encode_embedding(Execution& execution,
                            id<MTLBuffer> token_ids,
                            std::size_t rows,
                            const Matrix& destination) const;
    Matrix encode_product(Execution& execution,
                          const Matrix& activation,
                          const Weight& weight,
                          const Matrix& destination,
                          const char* operation) const;
    Matrix encode_rmsnorm(Execution& execution,
                          const Matrix& input,
                          const Vector& gamma,
                          const Matrix& destination,
                          const char* operation) const;
    Matrix encode_softmax(Execution& execution,
                          const Matrix& input,
                          const Matrix& destination,
                          const char* operation) const;
    Matrix encode_unary(Execution& execution,
                        const Matrix& input,
                        const Matrix& destination,
                        id<MTLComputePipelineState> pipeline,
                        const char* operation) const;
    Matrix encode_binary(Execution& execution,
                         const Matrix& left,
                         const Matrix& right,
                         const Matrix& destination,
                         id<MTLComputePipelineState> pipeline,
                         const char* operation) const;
    Matrix encode_channel(Execution& execution,
                          const Matrix& input,
                          const Vector& channel,
                          const Matrix& destination,
                          id<MTLComputePipelineState> pipeline,
                          const char* operation) const;
    Matrix encode_row_scale(Execution& execution,
                            const Matrix& input,
                            const Matrix& weights,
                            std::size_t column,
                            const Matrix& destination,
                            const char* operation) const;

    Matrix encode_full_attention(Execution& execution,
                                 const Matrix& normalized,
                                 FullAttention& attention,
                                 std::size_t position) const;
    Matrix encode_delta_net(Execution& execution,
                            const Matrix& normalized,
                            DeltaNet& delta) const;
    void encode_routes(Execution& execution,
                       const Matrix& normalized,
                       const Experts& experts) const;
    Matrix encode_selected_moe(Execution& execution,
                               const Matrix& normalized,
                               const Experts& experts) const;
    Matrix encode_layer_route(Execution& execution,
                              const Matrix& hidden,
                              Layer& layer,
                              std::size_t position) const;
    Matrix encode_layer_experts(Execution& execution,
                                const Matrix& attention_residual,
                                const Layer& layer) const;
    std::int32_t forward(const std::vector<std::int32_t>& token_ids,
                         std::size_t position,
                         const char* phase,
                         bool produce_output);
};

void MoeLLM::Impl::initialize_expert_sidecar(
    const std::string& gguf_path) noexcept {
    expert_sidecar.disable();
    try {
        FileIdentity identity;
        if (!query_file_identity(gguf_path, identity)) {
            std::cerr << "[moe] expert_bundle_cache=disabled reason=model_stat\n";
            return;
        }

        std::error_code error;
        const std::filesystem::path directory =
            std::filesystem::path("output") / "expert_cache";
        std::filesystem::create_directories(directory, error);
        if (error) {
            std::cerr << "[moe] expert_bundle_cache=disabled reason="
                      << error.message() << '\n';
            return;
        }
        expert_sidecar.path = directory /
            ("qwen35moe_" + model_fingerprint(gguf_path, identity) +
             ".bundles");
        expert_sidecar.descriptor = ::open(
            expert_sidecar.path.c_str(), O_RDWR | O_CREAT, 0644);
        if (expert_sidecar.descriptor < 0) {
            std::cerr << "[moe] expert_bundle_cache=disabled reason="
                      << std::strerror(errno) << '\n';
            expert_sidecar.disable();
            return;
        }

        const std::size_t entry_count = checked_mul(
            config.layer_count, config.expert_count,
            "expert bundle index entries");
        const std::size_t index_bytes = checked_mul(
            entry_count, sizeof(ExpertBundleEntry),
            "expert bundle index bytes");
        ExpertBundleHeader expected;
        std::memcpy(expected.magic, kExpertBundleMagic.data(),
                    kExpertBundleMagic.size());
        expected.version = kExpertBundleVersion;
        expected.header_bytes = sizeof(ExpertBundleHeader);
        expected.model_size = identity.size;
        expected.model_mtime_seconds = identity.mtime_seconds;
        expected.model_mtime_nanoseconds = identity.mtime_nanoseconds;
        expected.layer_count = to_uint(config.layer_count,
                                       "expert bundle layers");
        expected.expert_count = to_uint(config.expert_count,
                                        "expert bundle experts");
        expected.entry_count = entry_count;
        expected.index_offset = sizeof(ExpertBundleHeader);
        expected.data_offset = align_up(
            checked_add(sizeof(ExpertBundleHeader), index_bytes,
                        "expert bundle data offset"),
            4096, "expert bundle data alignment");

        struct stat status {};
        bool valid = ::fstat(expert_sidecar.descriptor, &status) == 0 &&
            status.st_size >= 0 &&
            static_cast<std::uint64_t>(status.st_size) >= expected.data_offset;
        ExpertBundleHeader existing;
        if (valid) {
            valid = read_all_at(expert_sidecar.descriptor, &existing,
                                sizeof(existing), 0) &&
                std::memcmp(existing.magic, expected.magic,
                            kExpertBundleMagic.size()) == 0 &&
                existing.version == expected.version &&
                existing.header_bytes == expected.header_bytes &&
                existing.model_size == expected.model_size &&
                existing.model_mtime_seconds == expected.model_mtime_seconds &&
                existing.model_mtime_nanoseconds ==
                    expected.model_mtime_nanoseconds &&
                existing.layer_count == expected.layer_count &&
                existing.expert_count == expected.expert_count &&
                existing.entry_count == expected.entry_count &&
                existing.index_offset == expected.index_offset &&
                existing.data_offset == expected.data_offset;
        }

        expert_sidecar.entries.assign(entry_count, {});
        if (valid) {
            valid = read_all_at(
                expert_sidecar.descriptor, expert_sidecar.entries.data(),
                index_bytes, existing.index_offset);
        }

        if (!valid) {
            if (::ftruncate(expert_sidecar.descriptor, 0) != 0 ||
                !write_all_at(expert_sidecar.descriptor, &expected,
                              sizeof(expected), 0) ||
                !write_all_at(expert_sidecar.descriptor,
                              expert_sidecar.entries.data(), index_bytes,
                              expected.index_offset) ||
                expected.data_offset > static_cast<std::uint64_t>(
                                           std::numeric_limits<off_t>::max()) ||
                ::ftruncate(expert_sidecar.descriptor,
                            static_cast<off_t>(expected.data_offset)) != 0) {
                std::cerr << "[moe] expert_bundle_cache=disabled reason="
                          << std::strerror(errno) << '\n';
                expert_sidecar.disable();
                return;
            }
            expert_sidecar.header = expected;
            expert_sidecar.append_offset = expected.data_offset;
        } else {
            expert_sidecar.header = existing;
            expert_sidecar.append_offset = static_cast<std::uint64_t>(
                status.st_size);
        }
        if (expert_sidecar.append_offset != 0 &&
            expert_sidecar.append_offset <=
                std::numeric_limits<std::size_t>::max()) {
            expert_sidecar.mapping_size = static_cast<std::size_t>(
                expert_sidecar.append_offset);
            void* mapping = ::mmap(
                nullptr, expert_sidecar.mapping_size, PROT_READ, MAP_SHARED,
                expert_sidecar.descriptor, 0);
            if (mapping != MAP_FAILED) {
                expert_sidecar.mapping =
                    static_cast<const std::uint8_t*>(mapping);
            } else {
                expert_sidecar.mapping_size = 0;
            }
        }
        std::cerr << "[moe] expert_bundle_cache="
                  << expert_sidecar.path.string()
                  << " mode=" << (valid ? "reuse" : "create") << '\n';
    } catch (const std::exception& error) {
        std::cerr << "[moe] expert_bundle_cache=disabled reason="
                  << error.what() << '\n';
        expert_sidecar.disable();
    } catch (...) {
        std::cerr << "[moe] expert_bundle_cache=disabled reason=unknown\n";
        expert_sidecar.disable();
    }
}

const std::uint8_t* MoeLLM::Impl::read_expert_sidecar(
    std::size_t layer_index,
    std::uint32_t expert,
    std::size_t expected_bytes) noexcept {
    if (!expert_sidecar.enabled() || layer_index >= config.layer_count ||
        expert >= config.expert_count) {
        return nullptr;
    }
    try {
        const std::size_t index = checked_add(
            checked_mul(layer_index, config.expert_count,
                        "expert bundle entry"),
            expert, "expert bundle entry");
        if (index >= expert_sidecar.entries.size()) return nullptr;
        ExpertBundleEntry& entry = expert_sidecar.entries[index];
        struct stat status {};
        const bool valid = entry.ready == 1 && entry.size == expected_bytes &&
            entry.offset >= expert_sidecar.header.data_offset &&
            entry.offset <= std::numeric_limits<std::uint64_t>::max() -
                                entry.size &&
            ::fstat(expert_sidecar.descriptor, &status) == 0 &&
            status.st_size >= 0 &&
            entry.offset + entry.size <=
                static_cast<std::uint64_t>(status.st_size);
        if (!valid) return nullptr;
        if (expert_sidecar.mapping != nullptr &&
            entry.offset + entry.size <= expert_sidecar.mapping_size) {
            return expert_sidecar.mapping + entry.offset;
        }
        expert_sidecar.scratch.resize(expected_bytes);
        if (!read_all_at(expert_sidecar.descriptor,
                         expert_sidecar.scratch.data(), expected_bytes,
                         entry.offset)) {
            expert_sidecar.disable();
            return nullptr;
        }
        return expert_sidecar.scratch.data();
    } catch (...) {
        expert_sidecar.disable();
        return nullptr;
    }
}

void MoeLLM::Impl::write_expert_sidecar(
    std::size_t layer_index,
    std::uint32_t expert,
    std::size_t bytes) noexcept {
    if (!expert_sidecar.enabled() || bytes == 0 ||
        bytes > expert_sidecar.scratch.size() ||
        layer_index >= config.layer_count || expert >= config.expert_count) {
        return;
    }
    try {
        const std::size_t index = checked_add(
            checked_mul(layer_index, config.expert_count,
                        "expert bundle entry"),
            expert, "expert bundle entry");
        if (index >= expert_sidecar.entries.size()) return;
        const std::uint64_t payload_offset = align_up(
            expert_sidecar.append_offset, 64, "expert bundle payload");
        if (!write_all_at(expert_sidecar.descriptor,
                          expert_sidecar.scratch.data(), bytes,
                          payload_offset)) {
            expert_sidecar.disable();
            return;
        }

        ExpertBundleEntry entry;
        entry.offset = payload_offset;
        entry.size = bytes;
        const std::uint64_t entry_offset = checked_add(
            expert_sidecar.header.index_offset,
            checked_mul(index, sizeof(ExpertBundleEntry),
                        "expert bundle entry offset"),
            "expert bundle entry offset");
        if (!write_all_at(expert_sidecar.descriptor, &entry, sizeof(entry),
                          entry_offset)) {
            expert_sidecar.disable();
            return;
        }

        const std::uint32_t ready = 1;
        if (!write_all_at(
                expert_sidecar.descriptor, &ready, sizeof(ready),
                checked_add(entry_offset, offsetof(ExpertBundleEntry, ready),
                            "expert bundle ready offset"))) {
            expert_sidecar.disable();
            return;
        }
        entry.ready = ready;
        expert_sidecar.entries[index] = entry;
        expert_sidecar.append_offset = checked_add(
            payload_offset, bytes, "expert bundle append offset");
    } catch (...) {
        expert_sidecar.disable();
    }
}

void MoeLLM::Impl::install_expert_bundle(
    std::size_t layer_index,
    std::uint32_t expert,
    std::size_t slot) {
    if (raw_model == nullptr || layer_index >= layers.size() ||
        expert >= config.expert_count || slot >= expert_cache_count) {
        throw std::invalid_argument("invalid expert bundle installation");
    }
    Experts& cache = layers[layer_index].experts;
    const std::size_t gate_offset = 0;
    const std::size_t up_offset = cache.gate.expert_stride_bytes;
    const std::size_t down_offset = checked_add(
        up_offset, cache.up.expert_stride_bytes, "expert bundle down offset");
    const std::size_t bundle_bytes = checked_add(
        down_offset, cache.down.expert_stride_bytes, "expert bundle size");

    const std::uint8_t* bundle = read_expert_sidecar(
        layer_index, expert, bundle_bytes);
    if (bundle == nullptr) {
        expert_sidecar.scratch.resize(bundle_bytes);
        const auto read_weight = [&](const ExpertWeight& weight,
                                     std::size_t bundle_offset) {
            const std::size_t source_offset = checked_mul(
                static_cast<std::size_t>(expert),
                weight.expert_stride_bytes, "expert source offset");
            raw_model->read_tensor_slice(
                weight.source, source_offset,
                expert_sidecar.scratch.data() + bundle_offset,
                weight.expert_stride_bytes);
        };
        read_weight(cache.gate, gate_offset);
        read_weight(cache.up, up_offset);
        read_weight(cache.down, down_offset);
        bundle = expert_sidecar.scratch.data();
        write_expert_sidecar(layer_index, expert, bundle_bytes);
    }

    const auto copy_weight = [&](const ExpertWeight& weight,
                                 std::size_t bundle_offset) {
        if (weight.buffer == nil || weight.slots != expert_cache_count) {
            throw std::runtime_error("invalid expert cache weight");
        }
        const std::size_t destination_offset = checked_mul(
            slot, weight.expert_stride_bytes, "expert cache offset");
        auto* destination = static_cast<std::uint8_t*>(
            [weight.buffer contents]) + destination_offset;
        std::memcpy(destination,
                    bundle + bundle_offset,
                    weight.expert_stride_bytes);
    };
    copy_weight(cache.gate, gate_offset);
    copy_weight(cache.up, up_offset);
    copy_weight(cache.down, down_offset);
}

void MoeLLM::Impl::load_model(const std::string& gguf_path,
                              std::size_t requested_capacity,
                              std::size_t requested_expert_cache_count) {
    require_available();
    raw_model = std::make_unique<MoeRawModel>(gguf_path);
    config = raw_model->config();
    if (config.architecture != "qwen35moe") {
        throw std::runtime_error(
            "MoeLLM requires qwen35moe, got " + config.architecture);
    }
    capacity = requested_capacity == 0
        ? std::min(config.context_length, kDefaultMoeContext)
        : requested_capacity;
    if (capacity == 0 || capacity > config.context_length) {
        throw std::invalid_argument(
            "MoE sequence capacity is outside the model context length");
    }
    if (requested_expert_cache_count < config.expert_used_count ||
        requested_expert_cache_count > config.expert_count) {
        throw std::invalid_argument(
            "per-layer expert cache count must be between top-k (" +
            std::to_string(config.expert_used_count) + ") and expert count (" +
            std::to_string(config.expert_count) + ")");
    }
    expert_cache_count = requested_expert_cache_count;

    embedding = upload_weight(
        raw_model->load_token_embedding(), "token embedding");
    output_norm = upload_vector(
        raw_model->load_output_norm(), "output norm");
    output = upload_weight(
        raw_model->load_output_weight(), "output projection");

    const std::size_t hidden = config.embedding_size;
    const std::size_t expert_ff = config.expert_feed_forward_size;
    const std::size_t shared_ff = config.shared_expert_feed_forward_size;
    const std::size_t experts = config.expert_count;
    const std::size_t kv_width = checked_mul(
        config.kv_head_count, config.head_size, "MoE KV width");
    const std::size_t qkv_width = checked_add(
        checked_mul(
            checked_mul(config.ssm_group_count, config.ssm_state_size,
                        "DeltaNet key width"),
            2, "DeltaNet QK width"),
        config.ssm_inner_size, "DeltaNet QKV width");

    layers.clear();
    layers.reserve(config.layer_count);
    for (std::size_t index = 0; index < config.layer_count; ++index) {
        MoeRawLayer raw = raw_model->load_layer(index);
        Layer layer;
        layer.kind = raw.kind;
        layer.attention_norm = upload_vector(
            raw.attention_norm_weight, "attention norm");
        layer.post_attention_norm = upload_vector(
            raw.post_attention_norm_weight, "post-attention norm");

        if (raw.kind == MoeLayerKind::FullAttention) {
            layer.attention.q_gate = upload_weight(
                raw.attention.q_gate_weight, "attention Q+gate");
            layer.attention.key = upload_weight(
                raw.attention.k_weight, "attention K");
            layer.attention.value = upload_weight(
                raw.attention.v_weight, "attention V");
            layer.attention.query_norm = upload_vector(
                raw.attention.q_norm_weight, "attention Q norm");
            layer.attention.key_norm = upload_vector(
                raw.attention.k_norm_weight, "attention K norm");
            layer.attention.output = upload_weight(
                raw.attention.output_weight, "attention output");

            const std::size_t cache_elements = checked_mul(
                capacity, kv_width, "full-attention cache");
            const std::size_t cache_bytes = float_bytes(
                cache_elements, "full-attention cache");
            layer.attention.key_cache = make_buffer(
                nullptr, cache_bytes, "full-attention key cache");
            layer.attention.value_cache = make_buffer(
                nullptr, cache_bytes, "full-attention value cache");
            kv_cache_bytes = checked_add(
                kv_cache_bytes, checked_mul(cache_bytes, 2, "KV caches"),
                "KV caches");
        } else {
            layer.delta.qkv = upload_weight(
                raw.delta_net.qkv_weight, "DeltaNet QKV");
            layer.delta.z = upload_weight(
                raw.delta_net.z_weight, "DeltaNet output gate");
            layer.delta.alpha = upload_weight(
                raw.delta_net.alpha_weight, "DeltaNet alpha");
            layer.delta.beta = upload_weight(
                raw.delta_net.beta_weight, "DeltaNet beta");
            layer.delta.convolution = upload_weight(
                raw.delta_net.conv_weight, "DeltaNet convolution");
            layer.delta.time_bias = upload_vector(
                raw.delta_net.dt_bias, "DeltaNet time bias");
            layer.delta.a = upload_vector(raw.delta_net.a, "DeltaNet A");
            layer.delta.state_norm = upload_vector(
                raw.delta_net.state_norm_weight, "DeltaNet state norm");
            layer.delta.output = upload_weight(
                raw.delta_net.output_weight, "DeltaNet output");

            if (layer.delta.qkv.rows != qkv_width ||
                layer.delta.qkv.cols != hidden) {
                throw std::runtime_error("DeltaNet prepared QKV shape mismatch");
            }
            const std::size_t state_elements = checked_mul(
                checked_mul(config.ssm_time_step_rank,
                            config.ssm_state_size, "DeltaNet state"),
                config.ssm_state_size, "DeltaNet state");
            const std::size_t state_bytes = float_bytes(
                state_elements, "DeltaNet recurrent state");
            const std::size_t history_elements = checked_mul(
                config.ssm_conv_kernel - 1, qkv_width,
                "DeltaNet convolution history");
            const std::size_t history_bytes = float_bytes(
                history_elements, "DeltaNet convolution history");
            layer.delta.recurrent_state = make_buffer(
                nullptr, state_bytes, "DeltaNet recurrent state");
            layer.delta.convolution_history = make_buffer(
                nullptr, history_bytes, "DeltaNet convolution history");
            recurrent_state_bytes = checked_add(
                recurrent_state_bytes,
                checked_add(state_bytes, history_bytes, "DeltaNet state"),
                "DeltaNet state");
        }

        layer.experts.router = upload_weight(
            raw.experts.router_weight, "MoE router");
        layer.experts.gate = prepare_expert_cache(
            raw.experts.gate_weight, experts, expert_cache_count,
            expert_ff, hidden,
            "MoE expert gate");
        layer.experts.up = prepare_expert_cache(
            raw.experts.up_weight, experts, expert_cache_count,
            expert_ff, hidden,
            "MoE expert up");
        layer.experts.down = prepare_expert_cache(
            raw.experts.down_weight, experts, expert_cache_count,
            hidden, expert_ff,
            "MoE expert down");
        layer.experts.shared_router = upload_weight(
            raw.experts.shared_router_weight, "shared expert router");
        layer.experts.shared_gate = upload_weight(
            raw.experts.shared_gate_weight, "shared expert gate");
        layer.experts.shared_up = upload_weight(
            raw.experts.shared_up_weight, "shared expert up");
        layer.experts.shared_down = upload_weight(
            raw.experts.shared_down_weight, "shared expert down");
        if (layer.experts.shared_gate.rows != shared_ff ||
            layer.experts.shared_up.rows != shared_ff ||
            layer.experts.shared_down.cols != shared_ff) {
            throw std::runtime_error("shared expert prepared shape mismatch");
        }
        layer.experts.expert_to_slot.assign(experts, -1);
        layer.experts.slot_to_expert.assign(expert_cache_count, -1);
        layer.experts.last_used.assign(expert_cache_count, 0);
        layers.push_back(std::move(layer));
    }
    raw_model->validate_all_tensors_loaded();
    initialize_expert_sidecar(gguf_path);
    reset_state();
}

void MoeLLM::Impl::reset_state() {
    release_arena();
    turn_peak_arena_bytes = 0;
    turn_expert_cache_hits = 0;
    turn_expert_cache_misses = 0;
    sequence_length = 0;
    for (Layer& layer : layers) {
        if (layer.kind != MoeLayerKind::GatedDeltaNet) continue;
        if (layer.delta.recurrent_state != nil) {
            std::memset([layer.delta.recurrent_state contents], 0,
                        [layer.delta.recurrent_state length]);
        }
        if (layer.delta.convolution_history != nil) {
            std::memset([layer.delta.convolution_history contents], 0,
                        [layer.delta.convolution_history length]);
        }
    }
}

void MoeLLM::Impl::ensure_experts_cached(
    std::size_t layer_index,
    std::uint32_t* route_ids,
    std::size_t route_count) {
    if (raw_model == nullptr || route_ids == nullptr ||
        layer_index >= layers.size()) {
        throw std::invalid_argument("invalid expert cache update");
    }
    Experts& cache = layers[layer_index].experts;
    if (cache.expert_to_slot.size() != config.expert_count ||
        cache.slot_to_expert.size() != expert_cache_count ||
        cache.last_used.size() != expert_cache_count) {
        throw std::runtime_error("expert cache metadata is inconsistent");
    }

    std::vector<std::uint32_t> required;
    required.reserve(std::min(route_count, config.expert_count));
    std::vector<bool> seen(config.expert_count, false);
    for (std::size_t index = 0; index < route_count; ++index) {
        const std::uint32_t expert = route_ids[index];
        if (expert >= config.expert_count) {
            throw std::runtime_error("router produced an invalid expert id");
        }
        if (!seen[expert]) {
            seen[expert] = true;
            required.push_back(expert);
        }
    }
    if (required.size() > expert_cache_count) {
        throw std::runtime_error(
            "one MoE routing chunk requires more experts than its cache");
    }

    for (std::uint32_t expert : required) {
        if (cache.expert_to_slot[expert] >= 0) {
            ++turn_expert_cache_hits;
        } else {
            ++turn_expert_cache_misses;
        }
    }

    // Slots required by this command are pinned while misses are installed, so
    // an early miss cannot evict a later route that was already resident.
    std::vector<bool> pinned(expert_cache_count, false);
    for (std::uint32_t expert : required) {
        const std::int32_t slot = cache.expert_to_slot[expert];
        if (slot >= 0) {
            pinned[static_cast<std::size_t>(slot)] = true;
            cache.last_used[static_cast<std::size_t>(slot)] = ++cache.lru_clock;
        }
    }

    for (std::uint32_t expert : required) {
        if (cache.expert_to_slot[expert] >= 0) continue;

        std::size_t slot = expert_cache_count;
        for (std::size_t candidate = 0;
             candidate < expert_cache_count;
             ++candidate) {
            if (!pinned[candidate] && cache.slot_to_expert[candidate] < 0) {
                slot = candidate;
                break;
            }
        }
        if (slot == expert_cache_count) {
            std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
            for (std::size_t candidate = 0;
                 candidate < expert_cache_count;
                 ++candidate) {
                if (!pinned[candidate] && cache.last_used[candidate] < oldest) {
                    oldest = cache.last_used[candidate];
                    slot = candidate;
                }
            }
        }
        if (slot == expert_cache_count) {
            throw std::runtime_error("no evictable MoE expert cache slot");
        }

        install_expert_bundle(layer_index, expert, slot);

        const std::int32_t evicted = cache.slot_to_expert[slot];
        if (evicted >= 0) {
            cache.expert_to_slot[static_cast<std::size_t>(evicted)] = -1;
        }
        cache.slot_to_expert[slot] = static_cast<std::int32_t>(expert);
        cache.expert_to_slot[expert] = static_cast<std::int32_t>(slot);
        cache.last_used[slot] = ++cache.lru_clock;
        pinned[slot] = true;
    }

    for (std::size_t index = 0; index < route_count; ++index) {
        const std::int32_t slot = cache.expert_to_slot[route_ids[index]];
        if (slot < 0) {
            throw std::runtime_error("routed expert was not installed in cache");
        }
        route_ids[index] = static_cast<std::uint32_t>(slot);
    }
}

MoeLLM::Impl::ArenaPlan MoeLLM::Impl::make_arena_plan(
    std::size_t rows,
    std::size_t score_columns) const {
    if (rows == 0 || score_columns == 0) {
        throw std::invalid_argument("MoE arena dimensions cannot be zero");
    }
    const std::size_t hidden = config.embedding_size;
    const std::size_t query_width = checked_mul(
        config.attention_head_count, config.head_size, "attention query width");
    const std::size_t kv_width = checked_mul(
        config.kv_head_count, config.head_size, "attention KV width");
    const std::size_t key_width = checked_mul(
        config.ssm_group_count, config.ssm_state_size,
        "DeltaNet key width");
    const std::size_t qkv_width = checked_add(
        checked_mul(key_width, 2, "DeltaNet QK width"),
        config.ssm_inner_size, "DeltaNet QKV width");
    const std::size_t wide_width = std::max(
        qkv_width,
        checked_mul(query_width, 2, "attention Q+gate width"));
    const std::size_t feature_width = std::max(
        query_width, config.ssm_inner_size);
    const std::size_t small_width = std::max(
        config.expert_count, config.ssm_time_step_rank);
    const std::size_t expert_width = std::max(
        config.expert_feed_forward_size,
        config.shared_expert_feed_forward_size);
    const std::size_t routed_width = checked_mul(
        config.expert_used_count, expert_width, "routed expert width");
    const std::size_t routed_hidden_width = checked_mul(
        config.expert_used_count, hidden, "routed expert output width");
    const std::size_t score_width = checked_mul(
        config.attention_head_count, score_columns, "attention score width");
    const std::size_t argmax_partials =
        (config.vocabulary_size - 1) / kArgmaxElementsPerGroup + 1;

    ArenaPlan plan;
    const auto add = [&](Slot slot, std::size_t bytes, const char* name) {
        const std::size_t offset = align_up(
            plan.bytes, kArenaAlignment, name);
        plan.regions[slot_index(slot)] = {offset, bytes};
        plan.bytes = checked_add(offset, bytes, name);
    };
    const auto add_f32 = [&](Slot slot,
                             std::size_t width,
                             const char* name) {
        add(slot, float_bytes(checked_mul(rows, width, name), name), name);
    };

    add_f32(Slot::HiddenA, hidden, "arena hidden A");
    add_f32(Slot::HiddenB, hidden, "arena hidden B");
    add_f32(Slot::HiddenC, hidden, "arena hidden C");
    add_f32(Slot::HiddenScratch, hidden, "arena hidden scratch");
    add_f32(Slot::Norm, hidden, "arena norm");
    add_f32(Slot::WideA, wide_width, "arena wide A");
    add_f32(Slot::WideB, wide_width, "arena wide B");
    add_f32(Slot::FeatureA, feature_width, "arena feature A");
    add_f32(Slot::FeatureB, feature_width, "arena feature B");
    add_f32(Slot::FeatureC, feature_width, "arena feature C");
    add_f32(Slot::FeatureD, feature_width, "arena feature D");
    add_f32(Slot::KVA, kv_width, "arena KV A");
    add_f32(Slot::KVB, kv_width, "arena KV B");
    add_f32(Slot::KVC, kv_width, "arena KV C");
    add_f32(Slot::SmallA, small_width, "arena small A");
    add_f32(Slot::SmallB, small_width, "arena small B");
    add_f32(Slot::SmallC, small_width, "arena small C");
    add_f32(Slot::ExpertGate, routed_width, "arena expert gate");
    add_f32(Slot::ExpertUp, routed_width, "arena expert up");
    add_f32(Slot::ExpertActivation, routed_width,
            "arena expert activation");
    add_f32(Slot::ExpertDown, routed_hidden_width,
            "arena routed expert output");
    add(Slot::RouteIds,
        checked_mul(checked_mul(rows, config.expert_used_count,
                                "arena route ids"),
                    sizeof(std::uint32_t), "arena route ids"),
        "arena route ids");
    add_f32(Slot::RouteWeights, config.expert_used_count,
            "arena route weights");
    add_f32(Slot::Scores, score_width, "arena scores");
    add(Slot::Logits,
        float_bytes(config.vocabulary_size, "arena logits"),
        "arena logits");
    add(Slot::ArgmaxValues,
        float_bytes(argmax_partials, "arena argmax values"),
        "arena argmax values");
    add(Slot::ArgmaxIndices,
        checked_mul(argmax_partials, sizeof(std::uint32_t),
                    "arena argmax indices"),
        "arena argmax indices");
    plan.bytes = align_up(plan.bytes, kArenaAlignment, "MoE arena");
    return plan;
}

void MoeLLM::Impl::install_arena(const ArenaPlan& plan, const char* phase) {
    if (plan.bytes == 0) {
        throw std::invalid_argument("cannot install an empty MoE arena");
    }
    release_arena();
    arena = make_buffer(nullptr, plan.bytes, phase);
    arena_plan = plan;
    turn_peak_arena_bytes = std::max(turn_peak_arena_bytes, plan.bytes);
}

MoeLLM::Impl::Execution MoeLLM::Impl::begin_execution(
    const char* phase,
    std::size_t sequence_tokens) const {
    Execution execution;
    execution.phase = phase;
    execution.sequence_tokens = sequence_tokens;
    execution.begin_ns = monotonic_ns();
    const std::uint64_t create_begin = monotonic_ns();
    execution.command = [queue commandBuffer];
    if (execution.command == nil) {
        throw std::runtime_error("Metal could not create a command buffer");
    }
    execution.encoder = [execution.command computeCommandEncoder];
    if (execution.encoder == nil) {
        throw std::runtime_error("Metal could not create a compute encoder");
    }
    execution.command_create_ns = monotonic_ns() - create_begin;
    execution.encode_begin_ns = monotonic_ns();
    if (metal_profiler != nullptr) {
        execution.command_index = metal_profiler->acquire_command_index();
    }
    return execution;
}

id<MTLBuffer> MoeLLM::Impl::io_buffer(Execution& execution,
                                      const void* data,
                                      std::size_t bytes,
                                      const char* name) const {
    id<MTLBuffer> buffer = make_buffer(data, bytes, name);
    execution.io_buffers.push_back(buffer);
    return buffer;
}

void MoeLLM::Impl::set_pipeline(Execution& execution,
                                id<MTLComputePipelineState> pipeline,
                                const char* operation,
                                std::size_t grid_x,
                                std::size_t grid_y,
                                std::size_t threads_x,
                                std::size_t threads_y) const {
    [execution.encoder setComputePipelineState:pipeline];
    NSString* label = [NSString stringWithUTF8String:operation];
    if (label != nil) [execution.encoder pushDebugGroup:label];

    ++execution.kernel_count;
    const std::size_t groups = checked_mul(grid_x, grid_y, "dispatch groups");
    const std::size_t group_threads = checked_mul(
        threads_x, threads_y, "dispatch threads");
    execution.threadgroups += groups;
    execution.dispatched_threads += checked_mul(
        groups, group_threads, "dispatched threads");
    if (groups == 1) ++execution.one_group_kernels;

    if (metal_profiler != nullptr &&
        metal_profiler->detailed_kernel_timestamps()) {
        MetalKernelProfileRecord record;
        record.order = execution.kernel_records.size();
        record.operation = operation;
        const char* pipeline_label = [[pipeline label] UTF8String];
        record.pipeline = pipeline_label == nullptr
            ? operation : std::string(pipeline_label);
        record.value_type = "F32 activation";
        record.dispatch_type = "threadgroups";
        record.grid_x = grid_x;
        record.grid_y = grid_y;
        record.grid_z = 1;
        record.threads_x = threads_x;
        record.threads_y = threads_y;
        record.threads_z = 1;
        record.threadgroups = groups;
        record.dispatched_threads = checked_mul(
            groups, group_threads, "profile dispatched threads");
        execution.kernel_records.push_back(std::move(record));
    }
}

void MoeLLM::Impl::dispatch_flat(Execution& execution,
                                 id<MTLComputePipelineState> pipeline,
                                 const char* operation,
                                 std::size_t count) const {
    const std::size_t threads = std::min<std::size_t>(
        kReductionThreads,
        std::max<std::size_t>(1, [pipeline maxTotalThreadsPerThreadgroup]));
    const std::size_t groups = (count + threads - 1) / threads;
    set_pipeline(execution, pipeline, operation, groups, 1, threads, 1);
    [execution.encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [execution.encoder popDebugGroup];
}

void MoeLLM::Impl::dispatch_rows(Execution& execution,
                                 id<MTLComputePipelineState> pipeline,
                                 const char* operation,
                                 std::size_t rows) const {
    if ([pipeline maxTotalThreadsPerThreadgroup] < kReductionThreads) {
        throw std::runtime_error(
            std::string(operation) + " requires 256 threads per group");
    }
    set_pipeline(execution, pipeline, operation, rows, 1,
                 kReductionThreads, 1);
    [execution.encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(
                          kReductionThreads, 1, 1)];
    [execution.encoder popDebugGroup];
}

void MoeLLM::Impl::dispatch_tiles(Execution& execution,
                                  id<MTLComputePipelineState> pipeline,
                                  const char* operation,
                                  std::size_t rows,
                                  std::size_t cols) const {
    const std::size_t groups_x = (cols + kTileSize - 1) / kTileSize;
    const std::size_t groups_y = (rows + kTileSize - 1) / kTileSize;
    set_pipeline(execution, pipeline, operation, groups_x, groups_y,
                 kTileSize, kTileSize);
    [execution.encoder dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, 1)
                      threadsPerThreadgroup:MTLSizeMake(
                          kTileSize, kTileSize, 1)];
    [execution.encoder popDebugGroup];
}

void MoeLLM::Impl::finish_execution(Execution& execution,
                                    const char* operation) const {
    [execution.encoder endEncoding];
    const std::uint64_t encode_ns = monotonic_ns() - execution.encode_begin_ns;
    const std::uint64_t commit_begin = monotonic_ns();
    [execution.command commit];
    const std::uint64_t commit_ns = monotonic_ns() - commit_begin;
    const std::uint64_t wait_begin = monotonic_ns();
    [execution.command waitUntilCompleted];
    const std::uint64_t wait_ns = monotonic_ns() - wait_begin;
    if ([execution.command error] != nil ||
        [execution.command status] != MTLCommandBufferStatusCompleted) {
        throw_metal(operation, [execution.command error]);
    }

    if (metal_profiler != nullptr) {
        MetalCommandProfileRecord record;
        record.command_index = execution.command_index;
        record.phase = execution.phase;
        record.sequence_tokens = execution.sequence_tokens;
        record.kernel_count = execution.kernel_count;
        record.threadgroup_count = execution.threadgroups;
        record.dispatched_threads = execution.dispatched_threads;
        record.single_threadgroup_kernels = execution.one_group_kernels;
        record.counter_requested =
            metal_profiler->detailed_kernel_timestamps();
        record.counter_active = false;
        record.counter_status = record.counter_requested
            ? "MoeLLM records dispatches but not per-kernel GPU counters"
            : "not requested";
        record.cpu_command_create_ns = execution.command_create_ns;
        record.cpu_encode_ns = encode_ns;
        record.cpu_commit_ns = commit_ns;
        record.cpu_wait_ns = wait_ns;
        record.cpu_total_ns = monotonic_ns() - execution.begin_ns;
        const CFTimeInterval gpu_start = [execution.command GPUStartTime];
        const CFTimeInterval gpu_end = [execution.command GPUEndTime];
        if (gpu_end >= gpu_start && gpu_start > 0.0) {
            record.gpu_duration_ms = (gpu_end - gpu_start) * 1000.0;
        }
        metal_profiler->write(record, execution.kernel_records);
    }
}

id<MTLComputePipelineState> MoeLLM::Impl::product_pipeline(
    MetalGgmlType type,
    bool vector) const {
    switch (type) {
    case MetalGgmlType::F32: return vector ? gevm_f32 : gemm_f32;
    case MetalGgmlType::Q5_0: return vector ? gevm_q5_0 : gemm_q5_0;
    case MetalGgmlType::Q8_0: return vector ? gevm_q8_0 : gemm_q8_0;
    case MetalGgmlType::Q4_K: return vector ? gevm_q4_k : gemm_q4_k;
    case MetalGgmlType::Q5_K: return vector ? gevm_q5_k : gemm_q5_k;
    case MetalGgmlType::Q6_K: return vector ? gevm_q6_k : gemm_q6_k;
    default:
        throw std::runtime_error(
            std::string("unsupported Metal product type ") +
            metal_ggml_type_name(type));
    }
}

id<MTLComputePipelineState> MoeLLM::Impl::embedding_pipeline(
    MetalGgmlType type) const {
    switch (type) {
    case MetalGgmlType::F32: return embedding_f32;
    case MetalGgmlType::Q5_0: return embedding_q5_0;
    case MetalGgmlType::Q8_0: return embedding_q8_0;
    case MetalGgmlType::Q4_K: return embedding_q4_k;
    case MetalGgmlType::Q6_K: return embedding_q6_k;
    default:
        throw std::runtime_error(
            std::string("unsupported Metal embedding type ") +
            metal_ggml_type_name(type));
    }
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_embedding(
    Execution& execution,
    id<MTLBuffer> token_ids,
    std::size_t rows,
    const Matrix& destination) const {
    if (token_ids == nil || rows == 0 || embedding.rows != config.vocabulary_size ||
        destination.rows != rows || destination.cols != embedding.cols) {
        throw std::invalid_argument("invalid MoE embedding dimensions");
    }
    MetalEmbeddingParamsHost params;
    params.sequence_length = to_uint(rows, "embedding rows");
    params.embedding_size = to_uint(embedding.cols, "embedding width");
    params.vocabulary_size = to_uint(embedding.rows, "embedding vocabulary");
    params.weight_row_bytes = to_uint(
        embedding.row_bytes, "embedding row bytes");
    id<MTLComputePipelineState> pipeline = embedding_pipeline(embedding.type);
    [execution.encoder setBuffer:token_ids offset:0 atIndex:0];
    [execution.encoder setBuffer:embedding.buffer offset:0 atIndex:1];
    [execution.encoder setBuffer:destination.buffer
                           offset:byte_offset(destination, "embedding output")
                          atIndex:2];
    [execution.encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_flat(execution, pipeline, "moe.embedding",
                  checked_mul(rows, embedding.cols, "embedding dispatch"));
    return destination;
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_product(
    Execution& execution,
    const Matrix& activation,
    const Weight& weight,
    const Matrix& destination,
    const char* operation) const {
    if (activation.buffer == nil || weight.buffer == nil ||
        activation.rows == 0 || activation.cols != weight.cols ||
        destination.rows != activation.rows ||
        destination.cols != weight.rows ||
        destination.stride != destination.cols) {
        throw std::invalid_argument(
            std::string(operation) + " product dimensions do not match");
    }
    const bool vector = activation.rows == 1;
    id<MTLComputePipelineState> pipeline = product_pipeline(weight.type, vector);

    if (vector) {
        [execution.encoder setBuffer:weight.buffer offset:0 atIndex:0];
        [execution.encoder setBuffer:activation.buffer
                               offset:byte_offset(activation, operation)
                              atIndex:1];
        [execution.encoder setBuffer:dummy offset:0 atIndex:2];
        [execution.encoder setBuffer:destination.buffer
                               offset:byte_offset(destination, operation)
                              atIndex:3];
        if (weight.type == MetalGgmlType::F32) {
            MetalGemvParamsHost params;
            params.output_size = to_uint(weight.rows, "GEVM output");
            params.input_size = to_uint(weight.cols, "GEVM input");
            params.matrix_stride = to_uint(
                weight.row_bytes / sizeof(float), "GEVM weight stride");
            params.matrix_transposed = 1;
            params.has_bias = 0;
            [execution.encoder setBytes:&params length:sizeof(params) atIndex:4];
        } else {
            MetalQuantizedProductParamsHost params;
            params.m = 1;
            params.n = to_uint(weight.rows, "quantized GEVM output");
            params.k = to_uint(weight.cols, "quantized GEVM input");
            params.activation_stride = to_uint(
                activation.stride, "quantized GEVM activation stride");
            params.weight_row_bytes = to_uint(
                weight.row_bytes, "quantized GEVM row bytes");
            params.has_bias = 0;
            [execution.encoder setBytes:&params length:sizeof(params) atIndex:4];
        }
        dispatch_flat(execution, pipeline, operation, weight.rows);
        return destination;
    }

    [execution.encoder setBuffer:activation.buffer
                           offset:byte_offset(activation, operation)
                          atIndex:0];
    [execution.encoder setBuffer:weight.buffer offset:0 atIndex:1];
    [execution.encoder setBuffer:dummy offset:0 atIndex:2];
    [execution.encoder setBuffer:destination.buffer
                           offset:byte_offset(destination, operation)
                          atIndex:3];
    if (weight.type == MetalGgmlType::F32) {
        MetalMatmulParamsHost params;
        params.m = to_uint(activation.rows, "GEMM rows");
        params.n = to_uint(weight.rows, "GEMM output width");
        params.k = to_uint(weight.cols, "GEMM input width");
        params.lhs_stride = to_uint(activation.stride, "GEMM activation stride");
        params.rhs_stride = to_uint(
            weight.row_bytes / sizeof(float), "GEMM weight stride");
        params.rhs_transposed = 1;
        [execution.encoder setBytes:&params length:sizeof(params) atIndex:4];
    } else {
        MetalQuantizedProductParamsHost params;
        params.m = to_uint(activation.rows, "quantized GEMM rows");
        params.n = to_uint(weight.rows, "quantized GEMM output width");
        params.k = to_uint(weight.cols, "quantized GEMM input width");
        params.activation_stride = to_uint(
            activation.stride, "quantized GEMM activation stride");
        params.weight_row_bytes = to_uint(
            weight.row_bytes, "quantized GEMM row bytes");
        [execution.encoder setBytes:&params length:sizeof(params) atIndex:4];
    }
    dispatch_tiles(execution, pipeline, operation,
                   activation.rows, weight.rows);
    return destination;
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_rmsnorm(
    Execution& execution,
    const Matrix& input,
    const Vector& gamma,
    const Matrix& destination,
    const char* operation) const {
    if (input.rows == 0 || input.cols != gamma.length ||
        input.stride != input.cols || destination.rows != input.rows ||
        destination.cols != input.cols || destination.stride != input.cols) {
        throw std::invalid_argument(
            std::string(operation) + " RMSNorm dimensions do not match");
    }
    MetalRmsNormParamsHost params;
    params.rows = to_uint(input.rows, "RMSNorm rows");
    params.cols = to_uint(input.cols, "RMSNorm columns");
    params.epsilon = config.norm_epsilon;
    [execution.encoder setBuffer:input.buffer
                           offset:byte_offset(input, operation)
                          atIndex:0];
    [execution.encoder setBuffer:gamma.buffer offset:0 atIndex:1];
    [execution.encoder setBuffer:destination.buffer
                           offset:byte_offset(destination, operation)
                          atIndex:2];
    [execution.encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_rows(execution, rmsnorm, operation, input.rows);
    return destination;
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_softmax(
    Execution& execution,
    const Matrix& input,
    const Matrix& destination,
    const char* operation) const {
    if (input.rows == 0 || input.cols == 0 || input.stride != input.cols ||
        destination.rows != input.rows || destination.cols != input.cols ||
        destination.stride != destination.cols) {
        throw std::invalid_argument(
            std::string(operation) + " softmax dimensions do not match");
    }
    MetalElementwiseParamsHost params{
        to_uint(input.rows, "softmax rows"),
        to_uint(input.cols, "softmax columns")};
    [execution.encoder setBuffer:input.buffer
                           offset:byte_offset(input, operation)
                          atIndex:0];
    [execution.encoder setBuffer:destination.buffer
                           offset:byte_offset(destination, operation)
                          atIndex:1];
    [execution.encoder setBytes:&params length:sizeof(params) atIndex:2];
    dispatch_rows(execution, softmax, operation, input.rows);
    return destination;
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_unary(
    Execution& execution,
    const Matrix& input,
    const Matrix& destination,
    id<MTLComputePipelineState> pipeline,
    const char* operation) const {
    if (input.rows == 0 || input.cols == 0 || input.stride != input.cols ||
        destination.rows != input.rows || destination.cols != input.cols ||
        destination.stride != destination.cols) {
        throw std::invalid_argument(
            std::string(operation) + " unary dimensions do not match");
    }
    MetalElementwiseParamsHost params{
        to_uint(input.rows, "unary rows"),
        to_uint(input.cols, "unary columns")};
    [execution.encoder setBuffer:input.buffer
                           offset:byte_offset(input, operation)
                          atIndex:0];
    [execution.encoder setBuffer:destination.buffer
                           offset:byte_offset(destination, operation)
                          atIndex:1];
    [execution.encoder setBytes:&params length:sizeof(params) atIndex:2];
    dispatch_flat(execution, pipeline, operation,
                  checked_mul(input.rows, input.cols, operation));
    return destination;
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_binary(
    Execution& execution,
    const Matrix& left,
    const Matrix& right,
    const Matrix& destination,
    id<MTLComputePipelineState> pipeline,
    const char* operation) const {
    if (left.rows == 0 || left.cols == 0 || left.rows != right.rows ||
        left.cols != right.cols || left.stride != left.cols ||
        right.stride != right.cols || destination.rows != left.rows ||
        destination.cols != left.cols || destination.stride != left.cols) {
        throw std::invalid_argument(
            std::string(operation) + " binary dimensions do not match");
    }
    MetalElementwiseParamsHost params{
        to_uint(left.rows, "binary rows"),
        to_uint(left.cols, "binary columns")};
    [execution.encoder setBuffer:left.buffer
                           offset:byte_offset(left, operation)
                          atIndex:0];
    [execution.encoder setBuffer:right.buffer
                           offset:byte_offset(right, operation)
                          atIndex:1];
    [execution.encoder setBuffer:destination.buffer
                           offset:byte_offset(destination, operation)
                          atIndex:2];
    [execution.encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_flat(execution, pipeline, operation,
                  checked_mul(left.rows, left.cols, operation));
    return destination;
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_channel(
    Execution& execution,
    const Matrix& input,
    const Vector& channel,
    const Matrix& destination,
    id<MTLComputePipelineState> pipeline,
    const char* operation) const {
    if (input.rows == 0 || input.cols != channel.length ||
        input.stride != input.cols || destination.rows != input.rows ||
        destination.cols != input.cols || destination.stride != input.cols) {
        throw std::invalid_argument(
            std::string(operation) + " channel dimensions do not match");
    }
    MetalBroadcastParamsHost params{
        to_uint(input.rows, "channel rows"),
        to_uint(input.cols, "channel columns")};
    [execution.encoder setBuffer:input.buffer
                           offset:byte_offset(input, operation)
                          atIndex:0];
    [execution.encoder setBuffer:channel.buffer offset:0 atIndex:1];
    [execution.encoder setBuffer:destination.buffer
                           offset:byte_offset(destination, operation)
                          atIndex:2];
    [execution.encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_flat(execution, pipeline, operation,
                  checked_mul(input.rows, input.cols, operation));
    return destination;
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_row_scale(
    Execution& execution,
    const Matrix& input,
    const Matrix& weights,
    std::size_t column,
    const Matrix& destination,
    const char* operation) const {
    if (input.rows == 0 || input.rows != weights.rows ||
        column >= weights.cols || input.stride != input.cols ||
        destination.rows != input.rows || destination.cols != input.cols ||
        destination.stride != input.cols) {
        throw std::invalid_argument(
            std::string(operation) + " row-scale dimensions do not match");
    }
    MetalRowScaleParamsHost params;
    params.rows = to_uint(input.rows, "row-scale rows");
    params.cols = to_uint(input.cols, "row-scale columns");
    params.weight_stride = to_uint(weights.stride, "row-scale weight stride");
    params.weight_column = to_uint(column, "row-scale weight column");
    [execution.encoder setBuffer:input.buffer
                           offset:byte_offset(input, operation)
                          atIndex:0];
    [execution.encoder setBuffer:weights.buffer
                           offset:byte_offset(weights, operation)
                          atIndex:1];
    [execution.encoder setBuffer:destination.buffer
                           offset:byte_offset(destination, operation)
                          atIndex:2];
    [execution.encoder setBytes:&params length:sizeof(params) atIndex:3];
    dispatch_flat(execution, row_scale, operation,
                  checked_mul(input.rows, input.cols, operation));
    return destination;
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_full_attention(
    Execution& execution,
    const Matrix& normalized,
    FullAttention& attention,
    std::size_t position) const {
    const std::size_t tokens = normalized.rows;
    const std::size_t head_dim = config.head_size;
    const std::size_t query_heads = config.attention_head_count;
    const std::size_t kv_heads = config.kv_head_count;
    const std::size_t query_width = checked_mul(
        query_heads, head_dim, "full attention query width");
    const std::size_t kv_width = checked_mul(
        kv_heads, head_dim, "full attention KV width");
    const std::size_t key_length = checked_add(
        position, tokens, "full attention key length");
    if (query_heads % kv_heads != 0 || key_length > capacity) {
        throw std::runtime_error("invalid full-attention head or cache shape");
    }

    const Matrix mixed = encode_product(
        execution, normalized, attention.q_gate,
        matrix(Slot::WideA, tokens, query_width * 2,
               "attention Q+gate output"),
        "moe.attention.q_gate");
    const Matrix query = matrix(
        Slot::FeatureA, tokens, query_width, "attention query");
    const Matrix gate = matrix(
        Slot::FeatureB, tokens, query_width, "attention gate");
    MetalSplitQGateParamsHost split_params{
        to_uint(tokens, "Q+gate rows"),
        to_uint(query_heads, "Q+gate heads"),
        to_uint(head_dim, "Q+gate head size")};
    [execution.encoder setBuffer:mixed.buffer
                           offset:byte_offset(mixed, "Q+gate split")
                          atIndex:0];
    [execution.encoder setBuffer:query.buffer
                           offset:byte_offset(query, "Q+gate split")
                          atIndex:1];
    [execution.encoder setBuffer:gate.buffer
                           offset:byte_offset(gate, "Q+gate split")
                          atIndex:2];
    [execution.encoder setBytes:&split_params
                         length:sizeof(split_params)
                        atIndex:3];
    dispatch_flat(execution, split_q_gate, "moe.attention.split_q_gate",
                  checked_mul(tokens, query_width, "Q+gate split"));

    const Matrix query_vectors{
        query.buffer, tokens * query_heads, head_dim, head_dim, query.offset};
    const Matrix normalized_query_vectors = encode_rmsnorm(
        execution, query_vectors, attention.query_norm,
        matrix(Slot::FeatureC, tokens * query_heads, head_dim,
               "normalized attention query"),
        "moe.attention.q_norm");
    const Matrix normalized_query{
        normalized_query_vectors.buffer, tokens, query_width, query_width,
        normalized_query_vectors.offset};

    const Matrix raw_key = encode_product(
        execution, normalized, attention.key,
        matrix(Slot::KVA, tokens, kv_width, "attention key"),
        "moe.attention.k");
    const Matrix value = encode_product(
        execution, normalized, attention.value,
        matrix(Slot::KVC, tokens, kv_width, "attention value"),
        "moe.attention.v");
    const Matrix key_vectors{
        raw_key.buffer, tokens * kv_heads, head_dim, head_dim, raw_key.offset};
    const Matrix normalized_key_vectors = encode_rmsnorm(
        execution, key_vectors, attention.key_norm,
        matrix(Slot::KVB, tokens * kv_heads, head_dim,
               "normalized attention key"),
        "moe.attention.k_norm");
    const Matrix normalized_key{
        normalized_key_vectors.buffer, tokens, kv_width, kv_width,
        normalized_key_vectors.offset};

    const auto apply_rope = [&](const Matrix& input,
                                const Matrix& destination,
                                const char* operation) {
        MetalRopeHeadsParamsHost params;
        params.rows = to_uint(input.rows, "RoPE rows");
        params.cols = to_uint(input.cols, "RoPE columns");
        params.head_dim = to_uint(head_dim, "RoPE head dimension");
        params.rotary_dimension = to_uint(
            config.rotary_dimension, "RoPE rotary dimension");
        params.position = to_uint(position, "RoPE position");
        params.theta = config.rope_theta;
        [execution.encoder setBuffer:input.buffer
                               offset:byte_offset(input, operation)
                              atIndex:0];
        [execution.encoder setBuffer:destination.buffer
                               offset:byte_offset(destination, operation)
                              atIndex:1];
        [execution.encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_flat(execution, rope, operation,
                      checked_mul(input.rows, input.cols, operation));
        return destination;
    };
    const Matrix rotated_query = apply_rope(
        normalized_query,
        matrix(Slot::FeatureA, tokens, query_width,
               "rotated attention query"),
        "moe.attention.rope_q");
    const Matrix rotated_key = apply_rope(
        normalized_key,
        matrix(Slot::KVA, tokens, kv_width, "rotated attention key"),
        "moe.attention.rope_k");

    MetalKVCacheWriteParamsHost write_params;
    write_params.source_rows = to_uint(tokens, "KV write rows");
    write_params.source_cols = to_uint(kv_width, "KV write columns");
    write_params.cache_stride = to_uint(kv_width, "KV cache stride");
    write_params.position = to_uint(position, "KV write position");
    [execution.encoder setBuffer:rotated_key.buffer
                           offset:byte_offset(rotated_key, "KV write key")
                          atIndex:0];
    [execution.encoder setBuffer:value.buffer
                           offset:byte_offset(value, "KV write value")
                          atIndex:1];
    [execution.encoder setBuffer:attention.key_cache offset:0 atIndex:2];
    [execution.encoder setBuffer:attention.value_cache offset:0 atIndex:3];
    [execution.encoder setBytes:&write_params
                         length:sizeof(write_params)
                        atIndex:4];
    dispatch_flat(execution, kv_write, "moe.attention.kv_write",
                  checked_mul(tokens, kv_width, "KV write dispatch"));

    const Matrix attention_output = matrix(
        Slot::FeatureD, tokens, query_width, "attention output");
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const Matrix scores = matrix(
        Slot::Scores, checked_mul(tokens, query_heads, "attention score rows"),
        key_length, "attention scores");
    MetalKVCacheQKParamsHost qk_params;
    qk_params.query_rows = to_uint(tokens, "QK query rows");
    qk_params.key_length = to_uint(key_length, "QK key length");
    qk_params.query_heads = to_uint(query_heads, "QK query heads");
    qk_params.kv_heads = to_uint(kv_heads, "QK KV heads");
    qk_params.cache_stride = to_uint(kv_width, "QK cache stride");
    qk_params.query_stride = to_uint(query_width, "QK query stride");
    qk_params.head_dim = to_uint(head_dim, "QK head dimension");
    qk_params.query_position = to_uint(position, "QK query position");
    qk_params.scale = attention_scale;
    [execution.encoder setBuffer:rotated_query.buffer
                           offset:byte_offset(rotated_query, "QK query")
                          atIndex:0];
    [execution.encoder setBuffer:attention.key_cache offset:0 atIndex:1];
    [execution.encoder setBuffer:scores.buffer
                           offset:byte_offset(scores, "QK scores")
                          atIndex:2];
    [execution.encoder setBytes:&qk_params length:sizeof(qk_params) atIndex:3];
    dispatch_flat(
        execution, kv_qk, "moe.attention.qk",
        checked_mul(checked_mul(tokens, query_heads, "QK head rows"),
                    key_length, "QK dispatch"));

    encode_softmax(execution, scores, scores, "moe.attention.softmax");

    MetalKVCacheAVParamsHost av_params;
    av_params.query_rows = to_uint(tokens, "AV query rows");
    av_params.key_length = to_uint(key_length, "AV key length");
    av_params.query_heads = to_uint(query_heads, "AV query heads");
    av_params.kv_heads = to_uint(kv_heads, "AV KV heads");
    av_params.cache_stride = to_uint(kv_width, "AV cache stride");
    av_params.score_stride = to_uint(key_length, "AV score stride");
    av_params.head_dim = to_uint(head_dim, "AV head dimension");
    av_params.output_stride = to_uint(query_width, "AV output stride");
    [execution.encoder setBuffer:scores.buffer
                           offset:byte_offset(scores, "AV scores")
                          atIndex:0];
    [execution.encoder setBuffer:attention.value_cache offset:0 atIndex:1];
    [execution.encoder setBuffer:attention_output.buffer
                           offset:byte_offset(attention_output, "AV output")
                          atIndex:2];
    [execution.encoder setBytes:&av_params length:sizeof(av_params) atIndex:3];
    dispatch_flat(
        execution, kv_av, "moe.attention.av",
        checked_mul(checked_mul(tokens, query_heads, "AV head rows"),
                    head_dim, "AV dispatch"));

    const Matrix gate_sigmoid = encode_unary(
        execution, gate,
        matrix(Slot::FeatureC, tokens, query_width,
               "attention gate sigmoid"),
        sigmoid, "moe.attention.gate_sigmoid");
    const Matrix gated_attention = encode_binary(
        execution, attention_output, gate_sigmoid,
        matrix(Slot::FeatureA, tokens, query_width,
               "gated attention output"),
        multiply, "moe.attention.gate_multiply");
    return encode_product(
        execution, gated_attention, attention.output,
        matrix(Slot::HiddenB, tokens, config.embedding_size,
               "attention projection"),
        "moe.attention.output");
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_delta_net(
    Execution& execution,
    const Matrix& normalized,
    DeltaNet& delta) const {
    const std::size_t tokens = normalized.rows;
    const std::size_t head_dim = config.ssm_state_size;
    const std::size_t key_heads = config.ssm_group_count;
    const std::size_t value_heads = config.ssm_time_step_rank;
    const std::size_t key_width = checked_mul(
        key_heads, head_dim, "DeltaNet key width");
    const std::size_t value_width = config.ssm_inner_size;
    const std::size_t qkv_width = checked_add(
        checked_mul(key_width, 2, "DeltaNet QK width"), value_width,
        "DeltaNet QKV width");
    if (value_width != checked_mul(value_heads, head_dim,
                                   "DeltaNet value width") ||
        value_heads % key_heads != 0) {
        throw std::runtime_error("invalid DeltaNet head configuration");
    }

    const Matrix qkv = encode_product(
        execution, normalized, delta.qkv,
        matrix(Slot::WideA, tokens, qkv_width, "DeltaNet QKV"),
        "moe.deltanet.qkv");
    const Matrix z = encode_product(
        execution, normalized, delta.z,
        matrix(Slot::FeatureB, tokens, value_width, "DeltaNet z"),
        "moe.deltanet.z");
    const Matrix beta_raw = encode_product(
        execution, normalized, delta.beta,
        matrix(Slot::SmallA, tokens, value_heads, "DeltaNet beta"),
        "moe.deltanet.beta");
    const Matrix beta = encode_unary(
        execution, beta_raw,
        matrix(Slot::SmallB, tokens, value_heads,
               "DeltaNet beta sigmoid"),
        sigmoid, "moe.deltanet.beta_sigmoid");
    const Matrix alpha = encode_product(
        execution, normalized, delta.alpha,
        matrix(Slot::SmallA, tokens, value_heads, "DeltaNet alpha"),
        "moe.deltanet.alpha");
    const Matrix biased_alpha = encode_channel(
        execution, alpha, delta.time_bias,
        matrix(Slot::SmallC, tokens, value_heads,
               "DeltaNet biased alpha"),
        add_channel, "moe.deltanet.alpha_bias");
    const Matrix positive_alpha = encode_unary(
        execution, biased_alpha,
        matrix(Slot::SmallA, tokens, value_heads,
               "DeltaNet positive alpha"),
        softplus, "moe.deltanet.softplus");
    const Matrix log_decay = encode_channel(
        execution, positive_alpha, delta.a,
        matrix(Slot::SmallC, tokens, value_heads,
               "DeltaNet log decay"),
        multiply_channel, "moe.deltanet.multiply_a");
    const Matrix decay = encode_unary(
        execution, log_decay,
        matrix(Slot::SmallA, tokens, value_heads, "DeltaNet decay"),
        exponential, "moe.deltanet.exp_decay");

    if (delta.convolution.type != MetalGgmlType::F32 ||
        delta.convolution.rows != qkv_width ||
        delta.convolution.cols != config.ssm_conv_kernel) {
        throw std::runtime_error("DeltaNet convolution weight is invalid");
    }
    const Matrix convolved = matrix(
        Slot::WideB, tokens, qkv_width, "DeltaNet convolution output");
    MetalDepthwiseConvParamsHost conv_params{
        to_uint(tokens, "convolution tokens"),
        to_uint(qkv_width, "convolution channels"),
        to_uint(config.ssm_conv_kernel, "convolution kernel size")};
    [execution.encoder setBuffer:qkv.buffer
                           offset:byte_offset(qkv, "DeltaNet convolution")
                          atIndex:0];
    [execution.encoder setBuffer:delta.convolution_history
                           offset:0 atIndex:1];
    [execution.encoder setBuffer:delta.convolution.buffer
                           offset:0 atIndex:2];
    [execution.encoder setBuffer:convolved.buffer
                           offset:byte_offset(convolved, "DeltaNet convolution")
                          atIndex:3];
    [execution.encoder setBytes:&conv_params length:sizeof(conv_params)
                        atIndex:4];
    dispatch_flat(execution, depthwise_conv, "moe.deltanet.conv1d",
                  checked_mul(tokens, qkv_width, "convolution dispatch"));

    [execution.encoder setBuffer:qkv.buffer
                           offset:byte_offset(qkv, "conv history commit")
                          atIndex:0];
    [execution.encoder setBuffer:delta.convolution_history
                           offset:0 atIndex:1];
    [execution.encoder setBytes:&conv_params length:sizeof(conv_params)
                        atIndex:2];
    dispatch_flat(execution, commit_conv_history,
                  "moe.deltanet.conv_history", qkv_width);

    const Matrix activated_qkv = encode_unary(
        execution, convolved,
        matrix(Slot::WideA, tokens, qkv_width,
               "activated DeltaNet QKV"),
        silu, "moe.deltanet.conv_silu");
    const Matrix query_view = view(
        activated_qkv, 0, tokens, key_width, qkv_width,
        "DeltaNet query view");
    const Matrix key_view = view(
        activated_qkv, key_width, tokens, key_width, qkv_width,
        "DeltaNet key view");
    const Matrix value_view = view(
        activated_qkv, key_width * 2, tokens, value_width, qkv_width,
        "DeltaNet value view");

    const auto normalize_heads = [&](const Matrix& input,
                                     const Matrix& destination,
                                     const char* operation) {
        MetalHeadNormParamsHost params;
        params.tokens = to_uint(tokens, "L2 norm tokens");
        params.head_count = to_uint(key_heads, "L2 norm heads");
        params.head_dim = to_uint(head_dim, "L2 norm head size");
        params.input_token_stride = to_uint(
            input.stride, "L2 norm input stride");
        params.epsilon = config.norm_epsilon;
        [execution.encoder setBuffer:input.buffer
                               offset:byte_offset(input, operation)
                              atIndex:0];
        [execution.encoder setBuffer:destination.buffer
                               offset:byte_offset(destination, operation)
                              atIndex:1];
        [execution.encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_rows(execution, l2norm_heads, operation,
                      checked_mul(tokens, key_heads, operation));
        return destination;
    };
    const Matrix query = normalize_heads(
        query_view,
        matrix(Slot::FeatureA, tokens, key_width,
               "normalized DeltaNet query"),
        "moe.deltanet.q_l2norm");
    const Matrix key = normalize_heads(
        key_view,
        matrix(Slot::FeatureC, tokens, key_width,
               "normalized DeltaNet key"),
        "moe.deltanet.k_l2norm");

    const Matrix recurrent_output = matrix(
        Slot::FeatureD, tokens, value_width, "DeltaNet recurrent output");
    MetalGdnParamsHost gdn_params;
    gdn_params.tokens = to_uint(tokens, "GDN tokens");
    gdn_params.key_head_count = to_uint(key_heads, "GDN key heads");
    gdn_params.value_head_count = to_uint(value_heads, "GDN value heads");
    gdn_params.head_dim = to_uint(head_dim, "GDN head size");
    gdn_params.q_stride = to_uint(query.stride, "GDN query stride");
    gdn_params.k_stride = to_uint(key.stride, "GDN key stride");
    gdn_params.v_stride = to_uint(value_view.stride, "GDN value stride");
    gdn_params.output_stride = to_uint(
        recurrent_output.stride, "GDN output stride");
    [execution.encoder setBuffer:query.buffer
                           offset:byte_offset(query, "GDN query") atIndex:0];
    [execution.encoder setBuffer:key.buffer
                           offset:byte_offset(key, "GDN key") atIndex:1];
    [execution.encoder setBuffer:value_view.buffer
                           offset:byte_offset(value_view, "GDN value") atIndex:2];
    [execution.encoder setBuffer:decay.buffer
                           offset:byte_offset(decay, "GDN decay") atIndex:3];
    [execution.encoder setBuffer:beta.buffer
                           offset:byte_offset(beta, "GDN beta") atIndex:4];
    [execution.encoder setBuffer:delta.recurrent_state offset:0 atIndex:5];
    [execution.encoder setBuffer:recurrent_output.buffer
                           offset:byte_offset(recurrent_output, "GDN output")
                          atIndex:6];
    [execution.encoder setBytes:&gdn_params length:sizeof(gdn_params)
                        atIndex:7];
    set_pipeline(execution, gdn, "moe.deltanet.recurrence",
                 value_heads, 1, head_dim, 1);
    [execution.encoder dispatchThreadgroups:MTLSizeMake(value_heads, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    [execution.encoder popDebugGroup];

    const Matrix recurrent_vectors{
        recurrent_output.buffer, tokens * value_heads, head_dim, head_dim,
        recurrent_output.offset};
    const Matrix normalized_vectors = encode_rmsnorm(
        execution, recurrent_vectors, delta.state_norm,
        matrix(Slot::FeatureA, tokens * value_heads, head_dim,
               "normalized DeltaNet state output"),
        "moe.deltanet.state_norm");
    const Matrix normalized_output{
        normalized_vectors.buffer, tokens, value_width, value_width,
        normalized_vectors.offset};
    const Matrix activated_z = encode_unary(
        execution, z,
        matrix(Slot::FeatureC, tokens, value_width,
               "activated DeltaNet output gate"),
        silu, "moe.deltanet.z_silu");
    const Matrix gated_output = encode_binary(
        execution, normalized_output, activated_z,
        matrix(Slot::WideA, tokens, value_width,
               "gated DeltaNet output"),
        multiply, "moe.deltanet.gate_multiply");
    return encode_product(
        execution, gated_output, delta.output,
        matrix(Slot::HiddenB, tokens, config.embedding_size,
               "DeltaNet output projection"),
        "moe.deltanet.output");
}

void MoeLLM::Impl::encode_routes(
    Execution& execution,
    const Matrix& normalized,
    const Experts& experts) const {
    const std::size_t tokens = normalized.rows;
    const std::size_t selected = config.expert_used_count;
    if (selected == 0 || selected > 8 || experts.router.rows != config.expert_count) {
        throw std::runtime_error("unsupported MoE routing configuration");
    }

    const Matrix router_logits = encode_product(
        execution, normalized, experts.router,
        matrix(Slot::SmallA, tokens, config.expert_count,
               "MoE router logits"),
        "moe.ffn.router");
    const Matrix probabilities = encode_softmax(
        execution, router_logits,
        matrix(Slot::SmallB, tokens, config.expert_count,
               "MoE router probabilities"),
        "moe.ffn.router_softmax");
    const Matrix route_weights = matrix(
        Slot::RouteWeights, tokens, selected, "MoE selected weights");
    const Region& route_id_region =
        arena_plan.regions[slot_index(Slot::RouteIds)];
    MetalTopKParamsHost topk_params{
        to_uint(tokens, "top-k rows"),
        to_uint(config.expert_count, "top-k experts"),
        to_uint(selected, "top-k selected count")};
    [execution.encoder setBuffer:probabilities.buffer
                           offset:byte_offset(probabilities, "MoE top-k")
                          atIndex:0];
    [execution.encoder setBuffer:arena
                           offset:route_id_region.offset_bytes
                          atIndex:1];
    [execution.encoder setBuffer:route_weights.buffer
                           offset:byte_offset(route_weights, "MoE top-k")
                          atIndex:2];
    [execution.encoder setBytes:&topk_params length:sizeof(topk_params)
                        atIndex:3];
    dispatch_flat(execution, topk, "moe.ffn.topk", tokens);

    [execution.encoder setBuffer:route_weights.buffer
                           offset:byte_offset(route_weights,
                                              "MoE top-k renorm")
                          atIndex:0];
    [execution.encoder setBytes:&topk_params length:sizeof(topk_params)
                        atIndex:1];
    dispatch_flat(execution, topk_renorm, "moe.ffn.topk_renorm", tokens);
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_selected_moe(
    Execution& execution,
    const Matrix& normalized,
    const Experts& experts) const {
    const std::size_t tokens = normalized.rows;
    const std::size_t selected = config.expert_used_count;
    const std::size_t expert_ff = config.expert_feed_forward_size;
    const std::size_t shared_ff = config.shared_expert_feed_forward_size;
    const std::size_t hidden = config.embedding_size;
    const std::size_t routed_rows = checked_mul(
        tokens, selected, "routed expert rows");
    const Region& route_id_region =
        arena_plan.regions[slot_index(Slot::RouteIds)];
    const Matrix route_weights = matrix(
        Slot::RouteWeights, tokens, selected, "MoE selected weights");

    const auto expert_product = [&](const Matrix& input,
                                    const ExpertWeight& weight,
                                    bool activation_is_routed,
                                    const Matrix& destination,
                                    const char* operation) {
        const std::size_t expected_input_rows = activation_is_routed
            ? routed_rows : tokens;
        if (input.rows != expected_input_rows || input.cols != weight.cols ||
            destination.rows != routed_rows ||
            destination.cols != weight.rows ||
            input.stride != input.cols || destination.stride != destination.cols ||
            weight.slots != expert_cache_count) {
            throw std::invalid_argument(
                std::string(operation) + " expert dimensions do not match");
        }
        id<MTLComputePipelineState> pipeline = nil;
        if (weight.type == MetalGgmlType::Q4_K) {
            pipeline = expert_q4_k;
        } else if (weight.type == MetalGgmlType::Q6_K) {
            pipeline = expert_q6_k;
        } else {
            throw std::runtime_error(
                std::string(operation) + " unsupported expert type " +
                metal_ggml_type_name(weight.type));
        }
        MetalExpertProductParamsHost params;
        params.tokens = to_uint(tokens, "expert tokens");
        params.routes = to_uint(selected, "expert routes");
        params.output_size = to_uint(weight.rows, "expert output size");
        params.input_size = to_uint(weight.cols, "expert input size");
        params.activation_stride = to_uint(input.stride,
                                           "expert activation stride");
        params.weight_row_bytes = to_uint(weight.row_bytes,
                                          "expert weight row bytes");
        params.expert_stride_bytes = to_uint(weight.expert_stride_bytes,
                                             "expert weight stride");
        params.activation_is_routed = activation_is_routed ? 1u : 0u;
        [execution.encoder setBuffer:input.buffer
                               offset:byte_offset(input, operation)
                              atIndex:0];
        [execution.encoder setBuffer:weight.buffer offset:0 atIndex:1];
        [execution.encoder setBuffer:arena
                               offset:route_id_region.offset_bytes
                              atIndex:2];
        [execution.encoder setBuffer:destination.buffer
                               offset:byte_offset(destination, operation)
                              atIndex:3];
        [execution.encoder setBytes:&params length:sizeof(params) atIndex:4];
        dispatch_flat(
            execution, pipeline, operation,
            checked_mul(routed_rows, weight.rows, "expert dispatch"));
        return destination;
    };

    const Matrix gate = expert_product(
        normalized, experts.gate, false,
        matrix(Slot::ExpertGate, routed_rows, expert_ff,
               "routed expert gate"),
        "moe.ffn.expert_gate");
    const Matrix up = expert_product(
        normalized, experts.up, false,
        matrix(Slot::ExpertUp, routed_rows, expert_ff,
               "routed expert up"),
        "moe.ffn.expert_up");
    const Matrix activation = encode_binary(
        execution, gate, up,
        matrix(Slot::ExpertActivation, routed_rows, expert_ff,
               "routed expert activation"),
        swiglu, "moe.ffn.expert_swiglu");
    const Matrix down = expert_product(
        activation, experts.down, true,
        matrix(Slot::ExpertDown, routed_rows, hidden,
               "routed expert down"),
        "moe.ffn.expert_down");

    const Matrix accumulator = matrix(
        Slot::HiddenB, tokens, hidden, "routed expert accumulator");
    MetalExpertWeightedReduceParamsHost reduce_params{
        to_uint(tokens, "expert reduction tokens"),
        to_uint(selected, "expert reduction routes"),
        to_uint(hidden, "expert reduction columns")};
    [execution.encoder setBuffer:down.buffer
                           offset:byte_offset(down, "expert reduction input")
                          atIndex:0];
    [execution.encoder setBuffer:route_weights.buffer
                           offset:byte_offset(route_weights,
                                              "expert reduction weights")
                          atIndex:1];
    [execution.encoder setBuffer:accumulator.buffer
                           offset:byte_offset(accumulator,
                                              "expert reduction output")
                          atIndex:2];
    [execution.encoder setBytes:&reduce_params
                         length:sizeof(reduce_params)
                        atIndex:3];
    dispatch_flat(
        execution, expert_weighted_reduce, "moe.ffn.expert_weighted_reduce",
        checked_mul(tokens, hidden, "expert reduction dispatch"));

    const Matrix shared_gate_raw = encode_product(
        execution, normalized, experts.shared_router,
        matrix(Slot::SmallA, tokens, 1, "shared expert scalar gate"),
        "moe.ffn.shared_router");
    const Matrix shared_gate = encode_unary(
        execution, shared_gate_raw,
        matrix(Slot::SmallC, tokens, 1,
               "shared expert sigmoid gate"),
        sigmoid, "moe.ffn.shared_router_sigmoid");
    const Matrix shared_up = encode_product(
        execution, normalized, experts.shared_up,
        matrix(Slot::ExpertUp, tokens, shared_ff, "shared expert up"),
        "moe.ffn.shared_up");
    const Matrix shared_ff_gate = encode_product(
        execution, normalized, experts.shared_gate,
        matrix(Slot::ExpertGate, tokens, shared_ff,
               "shared expert FFN gate"),
        "moe.ffn.shared_gate");
    const Matrix shared_activation = encode_binary(
        execution, shared_ff_gate, shared_up,
        matrix(Slot::ExpertActivation, tokens, shared_ff,
               "shared expert activation"),
        swiglu, "moe.ffn.shared_swiglu");
    const Matrix shared_down = encode_product(
        execution, shared_activation, experts.shared_down,
        matrix(Slot::HiddenA, tokens, hidden, "shared expert down"),
        "moe.ffn.shared_down");
    const Matrix scaled_shared = encode_row_scale(
        execution, shared_down, shared_gate, 0,
        matrix(Slot::HiddenScratch, tokens, hidden,
               "gated shared expert"),
        "moe.ffn.shared_weight");
    return encode_binary(
        execution, accumulator, scaled_shared,
        matrix(Slot::HiddenA, tokens, hidden, "combined MoE output"),
        residual, "moe.ffn.combine_shared");
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_layer_route(
    Execution& execution,
    const Matrix& hidden,
    Layer& layer,
    std::size_t position) const {
    const std::size_t tokens = hidden.rows;
    const std::size_t width = config.embedding_size;
    const Matrix normalized = encode_rmsnorm(
        execution, hidden, layer.attention_norm,
        matrix(Slot::Norm, tokens, width, "layer attention norm"),
        "moe.layer.attention_norm");
    const Matrix attention_output = layer.kind == MoeLayerKind::FullAttention
        ? encode_full_attention(execution, normalized, layer.attention, position)
        : encode_delta_net(execution, normalized, layer.delta);
    const Matrix attention_residual = encode_binary(
        execution, hidden, attention_output,
        matrix(Slot::HiddenC, tokens, width, "attention residual"),
        residual, "moe.layer.attention_residual");
    const Matrix post_attention = encode_rmsnorm(
        execution, attention_residual, layer.post_attention_norm,
        matrix(Slot::Norm, tokens, width, "post-attention norm"),
        "moe.layer.post_attention_norm");
    encode_routes(execution, post_attention, layer.experts);
    return attention_residual;
}

MoeLLM::Impl::Matrix MoeLLM::Impl::encode_layer_experts(
    Execution& execution,
    const Matrix& attention_residual,
    const Layer& layer) const {
    const std::size_t tokens = attention_residual.rows;
    const std::size_t width = config.embedding_size;
    const Matrix post_attention = matrix(
        Slot::Norm, tokens, width, "post-attention norm");
    const Matrix moe_output = encode_selected_moe(
        execution, post_attention, layer.experts);
    return encode_binary(
        execution, attention_residual, moe_output,
        matrix(Slot::HiddenA, tokens, width, "layer output"),
        residual, "moe.layer.ffn_residual");
}

std::int32_t MoeLLM::Impl::forward(
    const std::vector<std::int32_t>& token_ids,
    std::size_t position,
    const char* phase,
    bool produce_output) {
    const std::size_t tokens = token_ids.size();
    const std::size_t sequence_tokens = checked_add(
        position, tokens, "forward sequence length");
    if (layers.empty()) {
        throw std::runtime_error("MoE model has no decoder layers");
    }

    // The first command embeds the input and continues through layer zero's
    // router. Every later command consumes the route selected by the previous
    // command, evaluates those experts, and continues through the next layer's
    // router. This leaves only the unavoidable route readback boundary.
    const std::string first_phase = std::string(phase) + ".embedding_route.0";
    Execution first_execution = begin_execution(
        first_phase.c_str(), sequence_tokens);
    const id<MTLBuffer> token_buffer = io_buffer(
        first_execution, token_ids.data(),
        checked_mul(tokens, sizeof(std::int32_t), "token id buffer"),
        "token id buffer");
    Matrix hidden = encode_embedding(
        first_execution, token_buffer, tokens,
        matrix(Slot::HiddenA, tokens, config.embedding_size,
               "embedded tokens"));
    Matrix attention_residual = encode_layer_route(
        first_execution, hidden, layers.front(), position);
    finish_execution(first_execution, first_phase.c_str());

    for (std::size_t layer_index = 0;
         layer_index < layers.size();
         ++layer_index) {
        const Region& route_region =
            arena_plan.regions[slot_index(Slot::RouteIds)];
        auto* route_ids = reinterpret_cast<std::uint32_t*>(
            static_cast<std::uint8_t*>([arena contents]) +
            route_region.offset_bytes);
        ensure_experts_cached(
            layer_index, route_ids,
            checked_mul(tokens, config.expert_used_count,
                        "routed expert count"));

        const bool last_layer = layer_index + 1 == layers.size();
        const std::string command_phase = std::string(phase) + ".layer." +
            std::to_string(layer_index) +
            (last_layer
                 ? (produce_output ? ".experts_output" : ".experts")
                 : ".experts_route." + std::to_string(layer_index + 1));
        Execution execution = begin_execution(
            command_phase.c_str(), sequence_tokens);
        hidden = encode_layer_experts(
            execution, attention_residual, layers[layer_index]);

        if (!last_layer) {
            attention_residual = encode_layer_route(
                execution, hidden, layers[layer_index + 1], position);
            finish_execution(execution, command_phase.c_str());
            continue;
        }

        if (!produce_output) {
            finish_execution(execution, command_phase.c_str());
            return -1;
        }

        const Matrix final_norm = encode_rmsnorm(
            execution, hidden, output_norm,
            matrix(Slot::Norm, tokens, config.embedding_size, "final norm"),
            "moe.output_norm");
        const Matrix last_hidden = view(
            final_norm,
            checked_mul(tokens - 1, final_norm.stride, "last hidden offset"),
            1, final_norm.cols, final_norm.stride, "last hidden row");
        const Matrix logits = encode_product(
            execution, last_hidden, output,
            matrix(Slot::Logits, 1, config.vocabulary_size, "output logits"),
            "moe.output_projection");

        const std::size_t partial_count =
            (config.vocabulary_size - 1) / kArgmaxElementsPerGroup + 1;
        const Region& partial_value_region =
            arena_plan.regions[slot_index(Slot::ArgmaxValues)];
        const Region& partial_index_region =
            arena_plan.regions[slot_index(Slot::ArgmaxIndices)];
        MetalArgmaxParamsHost partial_params{
            to_uint(config.vocabulary_size, "argmax vocabulary"),
            kArgmaxElementsPerGroup};
        [execution.encoder setBuffer:logits.buffer
                               offset:byte_offset(logits, "argmax logits")
                              atIndex:0];
        [execution.encoder setBuffer:arena
                               offset:partial_value_region.offset_bytes
                              atIndex:1];
        [execution.encoder setBuffer:arena
                               offset:partial_index_region.offset_bytes
                              atIndex:2];
        [execution.encoder setBytes:&partial_params
                             length:sizeof(partial_params)
                            atIndex:3];
        if ([argmax_partial maxTotalThreadsPerThreadgroup] <
            kReductionThreads) {
            throw std::runtime_error(
                "parallel argmax requires 256 threads per group");
        }
        set_pipeline(execution, argmax_partial, "moe.argmax.partial",
                     partial_count, 1, kReductionThreads, 1);
        [execution.encoder
            dispatchThreadgroups:MTLSizeMake(partial_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kReductionThreads, 1, 1)];
        [execution.encoder popDebugGroup];

        MetalArgmaxParamsHost finalize_params{
            to_uint(partial_count, "argmax partial count"), 0};
        [execution.encoder setBuffer:arena
                               offset:partial_value_region.offset_bytes
                              atIndex:0];
        [execution.encoder setBuffer:arena
                               offset:partial_index_region.offset_bytes
                              atIndex:1];
        [execution.encoder setBuffer:arena
                               offset:partial_index_region.offset_bytes
                              atIndex:2];
        [execution.encoder setBytes:&finalize_params
                             length:sizeof(finalize_params)
                            atIndex:3];
        if ([argmax_finalize maxTotalThreadsPerThreadgroup] <
            kReductionThreads) {
            throw std::runtime_error(
                "parallel argmax finalize requires 256 threads per group");
        }
        set_pipeline(execution, argmax_finalize, "moe.argmax.finalize",
                     1, 1, kReductionThreads, 1);
        [execution.encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(
                              kReductionThreads, 1, 1)];
        [execution.encoder popDebugGroup];

        finish_execution(execution, command_phase.c_str());
        const auto* result_address = reinterpret_cast<const std::uint32_t*>(
            static_cast<const std::uint8_t*>([arena contents]) +
            partial_index_region.offset_bytes);
        const std::uint32_t result = *result_address;
        if (result > static_cast<std::uint32_t>(
                         std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("generated token does not fit int32_t");
        }
        return static_cast<std::int32_t>(result);
    }

    throw std::logic_error("MoE forward did not execute a final layer");
}

MoeLLM::MoeLLM(const std::string& gguf_path,
               std::size_t max_sequence,
               const std::string& shader_path,
               std::size_t expert_cache_count)
    : impl_(std::make_unique<Impl>(shader_path)) {
    impl_->load_model(gguf_path, max_sequence, expert_cache_count);
}

MoeLLM::~MoeLLM() = default;
MoeLLM::MoeLLM(MoeLLM&&) noexcept = default;
MoeLLM& MoeLLM::operator=(MoeLLM&&) noexcept = default;

void MoeLLM::reset() {
    if (impl_ == nullptr) {
        throw std::runtime_error("MoE model is not initialized");
    }
    impl_->require_available();
    impl_->reset_state();
}

std::int32_t MoeLLM::prefill(const std::vector<std::int32_t>& token_ids) {
    if (impl_ == nullptr) {
        throw std::runtime_error("MoE model is not initialized");
    }
    impl_->require_available();
    if (token_ids.empty()) {
        throw std::invalid_argument("MoE prefill requires at least one token");
    }
    if (impl_->sequence_length != 0) {
        throw std::invalid_argument(
            "MoE prefill requires an empty state; call reset first");
    }
    if (token_ids.size() > impl_->capacity) {
        throw std::length_error("MoE prefill exceeds sequence capacity");
    }
    for (std::int32_t token : token_ids) {
        if (token < 0 ||
            static_cast<std::size_t>(token) >= impl_->config.vocabulary_size) {
            throw std::out_of_range("MoE prefill token is outside vocabulary");
        }
    }

    Profiler::ForwardScope forward_scope(
        impl_->profiler.get(), token_ids.size());
    Profiler::Scope operation_scope(
        impl_->profiler.get(), "prefill.gpu_moe");
    const std::size_t chunk_tokens = std::max<std::size_t>(
        1, impl_->expert_cache_count / impl_->config.expert_used_count);
    impl_->install_arena(
        impl_->make_arena_plan(
            std::min(chunk_tokens, token_ids.size()), token_ids.size()),
        "MoE prefill arena");
    struct ArenaRelease {
        Impl* impl;
        ~ArenaRelease() { impl->release_arena(); }
    } release{impl_.get()};
    std::int32_t result = -1;
    for (std::size_t offset = 0; offset < token_ids.size();) {
        const std::size_t count = std::min(
            chunk_tokens, token_ids.size() - offset);
        const bool final_chunk = offset + count == token_ids.size();
        const std::vector<std::int32_t> chunk(
            token_ids.begin() + static_cast<std::ptrdiff_t>(offset),
            token_ids.begin() + static_cast<std::ptrdiff_t>(offset + count));
        result = impl_->forward(
            chunk, offset, "prefill", final_chunk);
        offset += count;
    }
    impl_->sequence_length = token_ids.size();
    return result;
}

std::int32_t MoeLLM::decode(std::int32_t token_id) {
    if (impl_ == nullptr) {
        throw std::runtime_error("MoE model is not initialized");
    }
    impl_->require_available();
    if (impl_->sequence_length == 0) {
        throw std::invalid_argument("MoE decode requires a completed prefill");
    }
    if (impl_->sequence_length >= impl_->capacity) {
        throw std::length_error("MoE decode exceeds sequence capacity");
    }
    if (token_id < 0 ||
        static_cast<std::size_t>(token_id) >= impl_->config.vocabulary_size) {
        throw std::out_of_range("MoE decode token is outside vocabulary");
    }
    if (impl_->arena == nil) {
        impl_->install_arena(
            impl_->make_arena_plan(1, impl_->capacity), "MoE decode arena");
    }
    const std::size_t position = impl_->sequence_length;
    Profiler::ForwardScope forward_scope(
        impl_->profiler.get(), position + 1);
    Profiler::Scope operation_scope(
        impl_->profiler.get(), "decode.gpu_moe");
    const std::int32_t result = impl_->forward(
        std::vector<std::int32_t>{token_id}, position, "decode", true);
    impl_->sequence_length = position + 1;
    return result;
}

void MoeLLM::enable_profiling(const std::string& csv_path,
                              bool detailed_kernel_timestamps) {
    if (impl_ == nullptr) {
        throw std::runtime_error("MoE model is not initialized");
    }
    impl_->profiler = std::make_unique<Profiler>(csv_path);
    impl_->metal_profiler = std::make_unique<MetalProfiler>(
        csv_path, detailed_kernel_timestamps);
}

void MoeLLM::disable_profiling() {
    if (impl_ != nullptr) {
        impl_->profiler.reset();
        impl_->metal_profiler.reset();
    }
}

const MetalModelConfig& MoeLLM::config() const {
    if (impl_ == nullptr) {
        throw std::runtime_error("MoE model is not initialized");
    }
    return impl_->config;
}

std::size_t MoeLLM::position() const noexcept {
    return impl_ == nullptr ? 0 : impl_->sequence_length;
}

std::size_t MoeLLM::max_sequence() const noexcept {
    return impl_ == nullptr ? 0 : impl_->capacity;
}

std::size_t MoeLLM::expert_cache_count() const noexcept {
    return impl_ == nullptr ? 0 : impl_->expert_cache_count;
}

ExpertCacheStats MoeLLM::expert_cache_stats() const noexcept {
    if (impl_ == nullptr) return {};
    return {
        impl_->turn_expert_cache_hits,
        impl_->turn_expert_cache_misses,
    };
}

bool MoeLLM::available() const noexcept {
    return impl_ != nullptr && impl_->device != nil && impl_->queue != nil &&
           impl_->gdn != nil && impl_->expert_q4_k != nil &&
           impl_->expert_q6_k != nil &&
           impl_->expert_weighted_reduce != nil &&
           impl_->argmax_partial != nil && impl_->argmax_finalize != nil;
}

bool MoeLLM::uses_gpu() const noexcept {
    return available();
}

const std::string& MoeLLM::device_name() const noexcept {
    static const std::string empty;
    return impl_ == nullptr ? empty : impl_->device_name;
}

MemoryStats MoeLLM::memory_stats() const noexcept {
    if (impl_ == nullptr) return {};
    return {
        static_cast<std::uint64_t>(impl_->weight_bytes),
        static_cast<std::uint64_t>(impl_->kv_cache_bytes),
        static_cast<std::uint64_t>(impl_->recurrent_state_bytes),
        static_cast<std::uint64_t>(impl_->turn_peak_arena_bytes),
        false,
    };
}

} // namespace llm

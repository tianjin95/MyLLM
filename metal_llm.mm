#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_llm.h"

#include "profiler.h"
#include "runtime.h"

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

static_assert(sizeof(MetalMatmulParamsHost) == 36,
              "Metal matmul parameter layout changed");
static_assert(sizeof(MetalGemvParamsHost) == 24,
              "Metal gemv parameter layout changed");

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

size_t checked_bytes(size_t elements, const char* name) {
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::length_error(std::string(name) + " byte count overflows");
    }
    return elements * sizeof(float);
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

ProfileMetrics matrix_copy_metrics(size_t rows, size_t cols) {
    const uint64_t bytes = static_cast<uint64_t>(rows) *
                           static_cast<uint64_t>(cols) * sizeof(float);
    return profile_copy_metrics(bytes, static_cast<uint64_t>(rows) + 1);
}

ProfileMetrics residual_profile_metrics(size_t rows, size_t cols) {
    const uint64_t bytes = static_cast<uint64_t>(rows) *
                           static_cast<uint64_t>(cols) * sizeof(float);
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

void validate_attention_dimensions(const Layer& layer,
                                  float theta,
                                  size_t d_rope,
                                  size_t d_head) {
    if (d_head == 0 || d_rope == 0 || d_rope > d_head || d_rope % 2 != 0) {
        throw std::invalid_argument("invalid Qwen2.5 attention head dimensions");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::invalid_argument("RoPE theta must be finite and positive");
    }
    const size_t q_dimension = layer.attn_q_weight.rows;
    const size_t k_dimension = layer.attn_k_weight.rows;
    const size_t v_dimension = layer.attn_v_weight.rows;
    if (q_dimension == 0 || k_dimension == 0 || k_dimension != v_dimension ||
        q_dimension % d_head != 0 || k_dimension % d_head != 0 ||
        layer.attn_q_weight.cols != layer.attn_k_weight.cols ||
        layer.attn_q_weight.cols != layer.attn_v_weight.cols) {
        throw std::invalid_argument("invalid Qwen2.5 Q/K/V projection shapes");
    }
    const size_t q_heads = q_dimension / d_head;
    const size_t kv_heads = k_dimension / d_head;
    if (q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("invalid Qwen2.5 GQA head configuration");
    }
}

void validate_kv_cache(const Matrix& key_cache,
                       const Matrix& value_cache,
                       size_t kv_dimension) {
    validate_matrix(key_cache, "Key cache");
    validate_matrix(value_cache, "Value cache");
    if (key_cache.rows != value_cache.rows ||
        key_cache.cols != value_cache.cols) {
        throw std::invalid_argument(
            "Key and value caches must have identical shapes");
    }
    if ((key_cache.rows != 0 || key_cache.cols != 0) &&
        key_cache.cols != kv_dimension) {
        throw std::invalid_argument(
            "Key cache column count does not match KV projection");
    }
}

} // namespace

struct MetalLLM::Impl {
    struct CachedMatrix {
        id<MTLBuffer> buffer = nil;
        size_t rows = 0;
        size_t cols = 0;
    };

    struct CachedVector {
        id<MTLBuffer> buffer = nil;
        size_t length = 0;
    };

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> gemm_pipeline = nil;
    id<MTLComputePipelineState> gevm_pipeline = nil;
    id<MTLBuffer> dummy_buffer = nil;
    std::string device_name;

    // Only objects explicitly uploaded by prepare() are kept here.  This
    // prevents a temporary activation whose address happens to be reused from
    // observing a stale cached buffer.
    mutable std::unordered_map<const Matrix*, CachedMatrix> matrix_cache;
    mutable std::unordered_map<const Vector*, CachedVector> vector_cache;

    void require_available() const {
        if (device == nil || gemm_pipeline == nil || gevm_pipeline == nil) {
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

            NSError* error = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            id<MTLLibrary> library =
                [device newLibraryWithSource:source_string
                                     options:options
                                       error:&error];
            if (library == nil) {
                throw_metal_error("MTLDevice newLibraryWithSource", error);
            }

            id<MTLFunction> gemm_function =
                [library newFunctionWithName:@"metal_gemm_f32"];
            id<MTLFunction> gevm_function =
                [library newFunctionWithName:@"metal_gevm_f32"];
            if (gemm_function == nil || gevm_function == nil) {
                throw std::runtime_error(
                    "metal_llm.metal does not contain the required kernels");
            }

            gemm_pipeline =
                [device newComputePipelineStateWithFunction:gemm_function
                                                       error:&error];
            if (gemm_pipeline == nil) {
                throw_metal_error("create GEMM pipeline", error);
            }
            gevm_pipeline =
                [device newComputePipelineStateWithFunction:gevm_function
                                                       error:&error];
            if (gevm_pipeline == nil) {
                throw_metal_error("create GEVM pipeline", error);
            }
            if ([gemm_pipeline maxTotalThreadsPerThreadgroup] <
                    static_cast<NSUInteger>(kTileSize * kTileSize)) {
                throw std::runtime_error(
                    "Metal device supports fewer than 256 GEMM threads per group");
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

    void cache_matrix(const Matrix& matrix) const {
        const std::vector<float> flat = flatten_matrix(matrix, "Metal cached matrix");
        const size_t elements = checked_elements(
            matrix.rows, matrix.cols, "Metal cached matrix");
        CachedMatrix cached;
        cached.buffer = make_buffer(
            flat.data(), checked_bytes(elements, "Metal cached matrix"));
        cached.rows = matrix.rows;
        cached.cols = matrix.cols;
        matrix_cache[&matrix] = cached;
    }

    void cache_vector(const Vector& vector) const {
        const std::vector<float> flat = flatten_vector(vector, "Metal cached vector");
        CachedVector cached;
        cached.buffer = make_buffer(
            flat.data(), checked_bytes(flat.size(), "Metal cached vector"));
        cached.length = vector.lens;
        vector_cache[&vector] = cached;
    }

    id<MTLBuffer> matrix_buffer(const Matrix& matrix) const {
        const auto cached = matrix_cache.find(&matrix);
        if (cached != matrix_cache.end()) {
            if (cached->second.rows != matrix.rows ||
                cached->second.cols != matrix.cols) {
                throw std::invalid_argument(
                    "cached Metal matrix shape changed after prepare");
            }
            return cached->second.buffer;
        }
        const std::vector<float> flat = flatten_matrix(matrix, "Metal matrix");
        return make_buffer(flat.data(), checked_bytes(flat.size(), "Metal matrix"));
    }

    id<MTLBuffer> vector_buffer(const Vector& vector) const {
        const auto cached = vector_cache.find(&vector);
        if (cached != vector_cache.end()) {
            if (cached->second.length != vector.lens) {
                throw std::invalid_argument(
                    "cached Metal vector length changed after prepare");
            }
            return cached->second.buffer;
        }
        const std::vector<float> flat = flatten_vector(vector, "Metal vector");
        return make_buffer(flat.data(), checked_bytes(flat.size(), "Metal vector"));
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

    Matrix matrix_product(const Matrix& left,
                          const Matrix& right,
                          bool left_transposed,
                          bool right_transposed,
                          const Vector* bias,
                          float scale) const {
        require_available();
        validate_matrix(left, "Metal GEMM left");
        validate_matrix(right, "Metal GEMM right");
        if (!std::isfinite(scale)) {
            throw std::invalid_argument("Metal GEMM scale must be finite");
        }

        const size_t m = left_transposed ? left.cols : left.rows;
        const size_t left_k = left_transposed ? left.rows : left.cols;
        const size_t right_k = right_transposed ? right.cols : right.rows;
        const size_t n = right_transposed ? right.rows : right.cols;
        if (left_k != right_k) {
            throw std::invalid_argument("Metal GEMM inner dimensions do not match");
        }
        if (bias != nullptr) {
            validate_vector(*bias, "Metal GEMM bias");
            if (bias->lens != n) {
                throw std::invalid_argument(
                    "Metal GEMM bias dimension does not match output");
            }
        }

        Matrix result(m, n);
        if (m == 0 || n == 0) {
            return result;
        }
        if (left_k == 0) {
            if (bias != nullptr) {
                for (size_t row = 0; row < m; ++row) {
                    std::copy(bias->values.begin(), bias->values.end(),
                              result.values[row].values.begin());
                }
            }
            return result;
        }

        const uint32_t m_u = checked_uint(m, "GEMM M");
        const uint32_t n_u = checked_uint(n, "GEMM N");
        const uint32_t k_u = checked_uint(left_k, "GEMM K");
        const uint32_t lhs_stride = checked_uint(left.cols, "GEMM lhs stride");
        const uint32_t rhs_stride = checked_uint(right.cols, "GEMM rhs stride");

        @autoreleasepool {
            id<MTLBuffer> lhs_buffer = matrix_buffer(left);
            id<MTLBuffer> rhs_buffer = matrix_buffer(right);
            id<MTLBuffer> bias_buffer =
                bias == nullptr ? dummy_buffer : vector_buffer(*bias);
            id<MTLBuffer> output_buffer = make_buffer(
                nullptr, checked_bytes(
                    checked_elements(m, n, "Metal GEMM output"),
                    "Metal GEMM output"));

            MetalMatmulParamsHost params;
            params.m = m_u;
            params.n = n_u;
            params.k = k_u;
            params.lhs_stride = lhs_stride;
            params.rhs_stride = rhs_stride;
            params.lhs_transposed = left_transposed ? 1U : 0U;
            params.rhs_transposed = right_transposed ? 1U : 0U;
            params.has_bias = bias == nullptr ? 0U : 1U;
            params.scale = scale;

            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (command == nil) {
                throw std::runtime_error("Metal failed to create GEMM command buffer");
            }
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (encoder == nil) {
                throw std::runtime_error("Metal failed to create GEMM encoder");
            }
            [encoder setComputePipelineState:gemm_pipeline];
            [encoder setBuffer:lhs_buffer offset:0 atIndex:0];
            [encoder setBuffer:rhs_buffer offset:0 atIndex:1];
            [encoder setBuffer:bias_buffer offset:0 atIndex:2];
            [encoder setBuffer:output_buffer offset:0 atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];

            const MTLSize threadgroups = MTLSizeMake(
                (static_cast<NSUInteger>(n) + kTileSize - 1) / kTileSize,
                (static_cast<NSUInteger>(m) + kTileSize - 1) / kTileSize,
                1);
            const MTLSize threads_per_group =
                MTLSizeMake(kTileSize, kTileSize, 1);
            [encoder dispatchThreadgroups:threadgroups
                     threadsPerThreadgroup:threads_per_group];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            check_command(command, "Metal GEMM");

            const float* source =
                static_cast<const float*>([output_buffer contents]);
            for (size_t row = 0; row < m; ++row) {
                std::copy_n(source + row * n, n,
                            result.values[row].values.begin());
            }
        }
        return result;
    }

    Vector vector_product(const Matrix& matrix,
                          const Vector& vector,
                          bool matrix_transposed,
                          const Vector* bias,
                          float scale) const {
        require_available();
        validate_matrix(matrix, "Metal GEVM matrix");
        validate_vector(vector, "Metal GEVM vector");
        if (!std::isfinite(scale)) {
            throw std::invalid_argument("Metal GEVM scale must be finite");
        }

        const size_t input_size = matrix_transposed ? matrix.cols : matrix.rows;
        const size_t output_size = matrix_transposed ? matrix.rows : matrix.cols;
        if (vector.lens != input_size) {
            throw std::invalid_argument("Metal GEVM inner dimensions do not match");
        }
        if (bias != nullptr) {
            validate_vector(*bias, "Metal GEVM bias");
            if (bias->lens != output_size) {
                throw std::invalid_argument(
                    "Metal GEVM bias dimension does not match output");
            }
        }

        Vector result(output_size);
        if (output_size == 0) {
            return result;
        }
        if (input_size == 0) {
            if (bias != nullptr) {
                result.values = bias->values;
            }
            return result;
        }

        const uint32_t output_u = checked_uint(output_size, "GEVM output");
        const uint32_t input_u = checked_uint(input_size, "GEVM input");
        const uint32_t stride_u = checked_uint(matrix.cols, "GEVM matrix stride");

        @autoreleasepool {
            id<MTLBuffer> matrix_buffer_value = matrix_buffer(matrix);
            id<MTLBuffer> vector_buffer_value = vector_buffer(vector);
            id<MTLBuffer> bias_buffer =
                bias == nullptr ? dummy_buffer : vector_buffer(*bias);
            id<MTLBuffer> output_buffer = make_buffer(
                nullptr, checked_bytes(output_size, "Metal GEVM output"));

            MetalGemvParamsHost params;
            params.output_size = output_u;
            params.input_size = input_u;
            params.matrix_stride = stride_u;
            params.matrix_transposed = matrix_transposed ? 1U : 0U;
            params.has_bias = bias == nullptr ? 0U : 1U;
            params.scale = scale;

            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (command == nil) {
                throw std::runtime_error("Metal failed to create GEVM command buffer");
            }
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (encoder == nil) {
                throw std::runtime_error("Metal failed to create GEVM encoder");
            }
            [encoder setComputePipelineState:gevm_pipeline];
            [encoder setBuffer:matrix_buffer_value offset:0 atIndex:0];
            [encoder setBuffer:vector_buffer_value offset:0 atIndex:1];
            [encoder setBuffer:bias_buffer offset:0 atIndex:2];
            [encoder setBuffer:output_buffer offset:0 atIndex:3];
            [encoder setBytes:&params length:sizeof(params) atIndex:4];

            const NSUInteger max_threads =
                [gevm_pipeline maxTotalThreadsPerThreadgroup];
            const NSUInteger threads = std::min<NSUInteger>(
                kDefaultGemvThreads, std::max<NSUInteger>(1, max_threads));
            const MTLSize threadgroups = MTLSizeMake(
                (static_cast<NSUInteger>(output_size) + threads - 1) / threads,
                1, 1);
            const MTLSize threads_per_group = MTLSizeMake(threads, 1, 1);
            [encoder dispatchThreadgroups:threadgroups
                     threadsPerThreadgroup:threads_per_group];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            check_command(command, "Metal GEVM");

            const float* source =
                static_cast<const float*>([output_buffer contents]);
            std::copy_n(source, output_size, result.values.begin());
        }
        return result;
    }
};

MetalLLM::MetalLLM(const std::string& shader_path)
    : impl_(std::make_unique<Impl>(shader_path)) {}

MetalLLM::~MetalLLM() = default;

MetalLLM::MetalLLM(MetalLLM&&) noexcept = default;

MetalLLM& MetalLLM::operator=(MetalLLM&&) noexcept = default;

bool MetalLLM::available() const noexcept {
    return impl_ != nullptr && impl_->device != nil &&
           impl_->gemm_pipeline != nil && impl_->gevm_pipeline != nil;
}

const std::string& MetalLLM::device_name() const noexcept {
    static const std::string empty;
    return impl_ == nullptr ? empty : impl_->device_name;
}

void MetalLLM::prepare(const std::vector<Layer>& layers,
                       const Matrix& output_weight) {
    if (!available()) {
        const std::string reason = device_name().empty()
            ? "Metal backend is not initialized"
            : device_name();
        throw std::runtime_error("Metal backend is unavailable: " + reason);
    }
    @autoreleasepool {
        impl_->matrix_cache.clear();
        impl_->vector_cache.clear();

        impl_->cache_matrix(output_weight);
        for (const Layer& layer : layers) {
            impl_->cache_matrix(layer.attn_q_weight);
            impl_->cache_vector(layer.attn_q_bias);
            impl_->cache_matrix(layer.attn_k_weight);
            impl_->cache_vector(layer.attn_k_bias);
            impl_->cache_matrix(layer.attn_v_weight);
            impl_->cache_vector(layer.attn_v_bias);
            impl_->cache_matrix(layer.attn_output_weight);
            impl_->cache_matrix(layer.ffn_gate_weight);
            impl_->cache_matrix(layer.ffn_down_weight);
            impl_->cache_matrix(layer.ffn_up_weight);
        }
    }
}

Matrix MetalLLM::gemm(const Matrix& left, const Matrix& right) const {
    return impl_->matrix_product(left, right, false, false, nullptr, 1.0f);
}

Matrix MetalLLM::gemmb(const Matrix& left, const Matrix& right,
                       const Vector& bias) const {
    return impl_->matrix_product(left, right, false, false, &bias, 1.0f);
}

Matrix MetalLLM::gemmt(const Matrix& left, const Matrix& right) const {
    return impl_->matrix_product(left, right, false, true, nullptr, 1.0f);
}

Matrix MetalLLM::gemmts(const Matrix& left, const Matrix& right,
                        size_t d_head) const {
    if (d_head == 0 || left.cols != d_head || right.cols != d_head) {
        throw std::invalid_argument("Metal GEMM head dimension does not match inputs");
    }
    return impl_->matrix_product(
        left, right, false, true, nullptr,
        1.0f / std::sqrt(static_cast<float>(d_head)));
}

Matrix MetalLLM::gemmtb(const Matrix& left, const Matrix& right,
                        const Vector& bias) const {
    return impl_->matrix_product(left, right, false, true, &bias, 1.0f);
}

Matrix MetalLLM::gemtm(const Matrix& left, const Matrix& right) const {
    return impl_->matrix_product(left, right, true, false, nullptr, 1.0f);
}

Matrix MetalLLM::gemtmb(const Matrix& left, const Matrix& right,
                        const Vector& bias) const {
    return impl_->matrix_product(left, right, true, false, &bias, 1.0f);
}

Matrix MetalLLM::gemtmt(const Matrix& left, const Matrix& right) const {
    return impl_->matrix_product(left, right, true, true, nullptr, 1.0f);
}

Matrix MetalLLM::gemtmtb(const Matrix& left, const Matrix& right,
                         const Vector& bias) const {
    return impl_->matrix_product(left, right, true, true, &bias, 1.0f);
}

Vector MetalLLM::gevm(const Matrix& matrix, const Vector& vector) const {
    return impl_->vector_product(matrix, vector, false, nullptr, 1.0f);
}

Vector MetalLLM::gevmb(const Matrix& matrix, const Vector& vector,
                       const Vector& bias) const {
    return impl_->vector_product(matrix, vector, false, &bias, 1.0f);
}

Vector MetalLLM::gevmts(const Matrix& matrix, const Vector& vector,
                        size_t d_head) const {
    if (d_head == 0 || matrix.cols != d_head || vector.lens != d_head) {
        throw std::invalid_argument("Metal GEVM head dimension does not match inputs");
    }
    return impl_->vector_product(
        matrix, vector, true, nullptr,
        1.0f / std::sqrt(static_cast<float>(d_head)));
}

Vector MetalLLM::gevmt(const Matrix& matrix, const Vector& vector) const {
    return impl_->vector_product(matrix, vector, true, nullptr, 1.0f);
}

Vector MetalLLM::gevmtb(const Matrix& matrix, const Vector& vector,
                        const Vector& bias) const {
    return impl_->vector_product(matrix, vector, true, &bias, 1.0f);
}

Matrix MetalLLM::attention(const Matrix& hidden,
                           const Layer& layer,
                           float epsilon,
                           size_t d_rope,
                           float theta,
                           size_t d_head,
                           Profiler* profiler) const {
    validate_matrix(hidden, "Metal attention hidden input");
    if (hidden.rows == 0 || hidden.cols == 0) {
        throw std::invalid_argument("Metal attention hidden input cannot be empty");
    }
    validate_attention_dimensions(layer, theta, d_rope, d_head);
    if (hidden.cols != layer.attn_q_weight.cols) {
        throw std::invalid_argument(
            "Metal attention hidden dimension does not match projection input");
    }

    Profiler::Scope attention_total(profiler, "attention.total");

    Matrix normalized;
    {
        Profiler::Scope scope(
            profiler, "attention.rmsnorm",
            profile_elementwise_metrics(hidden.rows * hidden.cols, 5, 3, 1));
        normalized = hidden;
        normalized.rmsnorm(layer.attn_norm_weight, epsilon);
    }

    Matrix Q;
    {
        Profiler::Scope scope(
            profiler, "attention.q_proj",
            profile_gemmb_metrics(normalized.rows,
                                  layer.attn_q_weight.rows,
                                  normalized.cols));
        Q = gemmtb(normalized, layer.attn_q_weight, layer.attn_q_bias);
    }
    Matrix K;
    {
        Profiler::Scope scope(
            profiler, "attention.k_proj",
            profile_gemmb_metrics(normalized.rows,
                                  layer.attn_k_weight.rows,
                                  normalized.cols));
        K = gemmtb(normalized, layer.attn_k_weight, layer.attn_k_bias);
    }
    Matrix V;
    {
        Profiler::Scope scope(
            profiler, "attention.v_proj",
            profile_gemmb_metrics(normalized.rows,
                                  layer.attn_v_weight.rows,
                                  normalized.cols));
        V = gemmtb(normalized, layer.attn_v_weight, layer.attn_v_bias);
    }

    if (Q.cols % d_head != 0 || K.cols % d_head != 0 || V.cols != K.cols) {
        throw std::invalid_argument(
            "Metal attention Q/K/V dimensions are not head-aligned");
    }
    const size_t q_head_num = Q.cols / d_head;
    const size_t kv_head_num = K.cols / d_head;
    if (q_head_num == 0 || kv_head_num == 0 ||
        q_head_num % kv_head_num != 0) {
        throw std::invalid_argument("Metal attention has invalid GQA heads");
    }

    std::vector<Matrix> query_heads;
    std::vector<Matrix> key_heads;
    std::vector<Matrix> value_heads;
    {
        Profiler::Scope scope(
            profiler, "attention.split_heads",
            matrix_copy_metrics(Q.rows, Q.cols) +
                matrix_copy_metrics(K.rows, K.cols) +
                matrix_copy_metrics(V.rows, V.cols));
        query_heads = Q.split_heads(q_head_num);
        key_heads = K.split_heads(kv_head_num);
        value_heads = V.split_heads(kv_head_num);
    }

    {
        Profiler::Scope scope(
            profiler, "attention.rope",
            profile_elementwise_metrics(
                Q.rows * (Q.cols + K.cols), 6, 1, 1));
        for (Matrix& head : query_heads) {
            head.rope(0, theta, d_rope);
        }
        for (Matrix& head : key_heads) {
            head.rope(0, theta, d_rope);
        }
    }

    const size_t group_size = q_head_num / kv_head_num;
    std::vector<Matrix> head_outputs;
    head_outputs.reserve(q_head_num);
    for (size_t head_index = 0; head_index < q_head_num; ++head_index) {
        const size_t group = head_index / group_size;
        Matrix scores;
        {
            Profiler::Scope scope(
                profiler, "attention.qk",
                profile_gemm_metrics(query_heads[head_index].rows,
                                     key_heads[group].rows,
                                     d_head));
            scores = gemmts(query_heads[head_index], key_heads[group], d_head);
        }
        {
            Profiler::Scope scope(
                profiler, "attention.mask",
                mask_profile_metrics(scores.rows, scores.cols));
            scores.causal_mask();
        }
        {
            Profiler::Scope scope(
                profiler, "attention.softmax",
                profile_elementwise_metrics(
                    scores.rows * scores.cols, 4, 3, 2));
            scores.softmax();
        }
        Matrix output;
        {
            Profiler::Scope scope(
                profiler, "attention.av",
                profile_gemm_metrics(scores.rows,
                                     value_heads[group].cols,
                                     scores.cols));
            output = gemm(scores, value_heads[group]);
        }
        head_outputs.push_back(std::move(output));
    }

    Matrix concatenated;
    {
        Profiler::Scope scope(
            profiler, "attention.concat",
            matrix_copy_metrics(hidden.rows, Q.cols));
        concatenated = concat(head_outputs);
    }

    Matrix projected;
    {
        Profiler::Scope scope(
            profiler, "attention.output_proj",
            profile_gemm_metrics(concatenated.rows,
                                 layer.attn_output_weight.rows,
                                 concatenated.cols));
        projected = gemmt(concatenated, layer.attn_output_weight);
    }

    Matrix result;
    {
        Profiler::Scope scope(
            profiler, "attention.residual",
            residual_profile_metrics(hidden.rows, hidden.cols));
        result = residual(hidden, projected);
    }
    return result;
}

Matrix MetalLLM::ffn(const Matrix& hidden,
                     const Layer& layer,
                     float epsilon,
                     Profiler* profiler) const {
    validate_matrix(hidden, "Metal FFN hidden input");
    if (hidden.rows == 0 || hidden.cols == 0) {
        throw std::invalid_argument("Metal FFN hidden input cannot be empty");
    }

    Profiler::Scope ffn_total(profiler, "ffn.total");

    Matrix normalized;
    {
        Profiler::Scope scope(
            profiler, "ffn.rmsnorm",
            profile_elementwise_metrics(hidden.rows * hidden.cols, 5, 3, 1));
        normalized = hidden;
        normalized.rmsnorm(layer.ffn_norm_weight, epsilon);
    }

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

Matrix MetalLLM::prefill(const Matrix& hidden,
                         const Layer& layer,
                         float epsilon,
                         size_t d_rope,
                         float theta,
                         size_t d_head,
                         Matrix& key_cache,
                         Matrix& value_cache,
                         Profiler* profiler) const {
    validate_matrix(hidden, "Metal prefill hidden input");
    if (hidden.rows == 0 || hidden.cols == 0) {
        throw std::invalid_argument("Metal prefill hidden input cannot be empty");
    }
    validate_attention_dimensions(layer, theta, d_rope, d_head);
    if (hidden.cols != layer.attn_q_weight.cols) {
        throw std::invalid_argument(
            "Metal prefill hidden dimension does not match attention input");
    }

    const size_t q_head_num = layer.attn_q_weight.rows / d_head;
    const size_t kv_head_num = layer.attn_k_weight.rows / d_head;
    const size_t kv_dimension = layer.attn_k_weight.rows;
    validate_kv_cache(key_cache, value_cache, kv_dimension);

    Profiler::Scope total(profiler, "prefill.total");

    Matrix normalized = hidden;
    normalized.rmsnorm(layer.attn_norm_weight, epsilon);
    Matrix Q = gemmtb(normalized, layer.attn_q_weight, layer.attn_q_bias);
    key_cache = gemmtb(normalized, layer.attn_k_weight, layer.attn_k_bias);
    value_cache = gemmtb(normalized, layer.attn_v_weight, layer.attn_v_bias);

    std::vector<Matrix> query_heads = Q.split_heads(q_head_num);
    std::vector<Matrix> key_heads = key_cache.split_heads(kv_head_num);
    std::vector<Matrix> value_heads = value_cache.split_heads(kv_head_num);
    for (Matrix& head : query_heads) {
        head.rope(0, theta, d_rope);
    }
    for (Matrix& head : key_heads) {
        head.rope(0, theta, d_rope);
    }
    key_cache = concat(key_heads);

    const size_t group_size = q_head_num / kv_head_num;
    std::vector<Matrix> head_outputs;
    head_outputs.reserve(q_head_num);
    for (size_t head_index = 0; head_index < q_head_num; ++head_index) {
        const size_t group = head_index / group_size;
        Matrix scores = gemmts(query_heads[head_index], key_heads[group], d_head);
        scores.causal_mask();
        scores.softmax();
        head_outputs.push_back(gemm(scores, value_heads[group]));
    }

    Matrix concatenated = concat(head_outputs);
    Matrix projected = gemmt(concatenated, layer.attn_output_weight);
    Matrix after_attention = residual(hidden, projected);

    Matrix ffn_input = after_attention;
    ffn_input.rmsnorm(layer.ffn_norm_weight, epsilon);
    Matrix gate = gemmt(ffn_input, layer.ffn_gate_weight);
    Matrix up = gemmt(ffn_input, layer.ffn_up_weight);
    gate.swiglu(up);
    Matrix down = gemmt(gate, layer.ffn_down_weight);
    return residual(after_attention, down);
}

Vector MetalLLM::decode(const Vector& hidden,
                        const Layer& layer,
                        float epsilon,
                        size_t d_rope,
                        float theta,
                        size_t d_head,
                        Matrix& key_cache,
                        Matrix& value_cache,
                        Profiler* profiler) const {
    validate_vector(hidden, "Metal decode hidden input");
    if (hidden.lens == 0) {
        throw std::invalid_argument("Metal decode hidden input cannot be empty");
    }
    validate_attention_dimensions(layer, theta, d_rope, d_head);
    if (hidden.lens != layer.attn_q_weight.cols) {
        throw std::invalid_argument(
            "Metal decode hidden dimension does not match attention input");
    }

    const size_t q_head_num = layer.attn_q_weight.rows / d_head;
    const size_t kv_head_num = layer.attn_k_weight.rows / d_head;
    const size_t kv_dimension = layer.attn_k_weight.rows;
    validate_kv_cache(key_cache, value_cache, kv_dimension);
    const size_t position = key_cache.rows;

    Profiler::Scope total(profiler, "decode.total");

    Vector normalized = hidden;
    normalized.rmsnorm(layer.attn_norm_weight, epsilon);
    Vector Q = gevmtb(layer.attn_q_weight, normalized, layer.attn_q_bias);
    Vector K = gevmtb(layer.attn_k_weight, normalized, layer.attn_k_bias);
    Vector V = gevmtb(layer.attn_v_weight, normalized, layer.attn_v_bias);

    std::vector<Vector> query_heads = Q.split_heads(q_head_num);
    for (Vector& head : query_heads) {
        head.rope(position, theta, d_rope);
    }
    std::vector<Vector> new_key_heads = K.split_heads(kv_head_num);
    for (Vector& head : new_key_heads) {
        head.rope(position, theta, d_rope);
    }
    key_cache.append(concat(new_key_heads));
    value_cache.append(V);

    const std::vector<Matrix> key_heads = key_cache.split_heads(kv_head_num);
    const std::vector<Matrix> value_heads = value_cache.split_heads(kv_head_num);
    const size_t group_size = q_head_num / kv_head_num;
    std::vector<Vector> head_outputs;
    head_outputs.reserve(q_head_num);
    for (size_t head_index = 0; head_index < q_head_num; ++head_index) {
        const size_t group = head_index / group_size;
        Vector scores = gevmts(key_heads[group], query_heads[head_index], d_head);
        scores.softmax();
        head_outputs.push_back(gevm(value_heads[group], scores));
    }

    Vector concatenated = concat(head_outputs);
    Vector projected = gevmt(layer.attn_output_weight, concatenated);
    Vector after_attention = residual(hidden, projected);

    Vector ffn_input = after_attention;
    ffn_input.rmsnorm(layer.ffn_norm_weight, epsilon);
    Vector gate = gevmt(layer.ffn_gate_weight, ffn_input);
    Vector up = gevmt(layer.ffn_up_weight, ffn_input);
    gate.swiglu(up);
    Vector down = gevmt(layer.ffn_down_weight, gate);
    return residual(after_attention, down);
}

int32_t MetalLLM::forward(const std::vector<int32_t>& token_ids,
                          const llm_runtime& runtime,
                          Profiler* profiler) const {
    if (token_ids.empty()) {
        throw std::invalid_argument("Metal forward requires at least one token id");
    }
    if (runtime.context_length == 0 ||
        token_ids.size() > runtime.context_length) {
        throw std::invalid_argument(
            "Metal token sequence exceeds model context length");
    }
    if (runtime.layers.size() != runtime.layer_count) {
        throw std::runtime_error("Runtime layer storage is incomplete");
    }

    Profiler::ForwardScope forward_profile(profiler, token_ids.size());

    Matrix hidden;
    {
        Profiler::Scope scope(
            profiler, "forward.embedding",
            matrix_copy_metrics(token_ids.size(), runtime.embedding_size));
        hidden = embedding(token_ids, runtime.token_embedding_weight);
    }

    for (size_t layer_index = 0; layer_index < runtime.layers.size(); ++layer_index) {
        const Layer& layer = runtime.layers[layer_index];
        Profiler::LayerScope layer_profile(profiler, layer_index);
        hidden = attention(hidden, layer, runtime.norm_epsilon,
                           runtime.rotary_dimension, runtime.rope_theta,
                           runtime.head_size, profiler);
        hidden = ffn(hidden, layer, runtime.norm_epsilon, profiler);
    }

    {
        Profiler::Scope scope(
            profiler, "forward.final_rmsnorm",
            profile_elementwise_metrics(
                hidden.rows * hidden.cols, 5, 2, 1));
        hidden.rmsnorm(runtime.output_norm_weight, runtime.norm_epsilon);
    }
    if (hidden.rows == 0 || hidden.values.empty()) {
        throw std::runtime_error("Metal forward produced an empty hidden matrix");
    }

    Vector logits;
    {
        Profiler::Scope scope(
            profiler, "forward.lm_head.gevmt",
            profile_gemv_metrics(runtime.output_weight.cols,
                                 runtime.output_weight.rows));
        logits = gevmt(runtime.output_weight, hidden.values.back());
    }
    if (logits.values.empty()) {
        throw std::runtime_error("Metal LM head produced no logits");
    }
    const auto best = std::max_element(logits.values.begin(), logits.values.end());
    const size_t token = static_cast<size_t>(
        std::distance(logits.values.begin(), best));
    if (token > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error("Generated token id does not fit in int32_t");
    }
    return static_cast<int32_t>(token);
}

} // namespace llm

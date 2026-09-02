#include "metal_model.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llm {
namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747U;

enum GgufValueType : std::uint32_t {
    kGgufUint8 = 0,
    kGgufInt8 = 1,
    kGgufUint16 = 2,
    kGgufInt16 = 3,
    kGgufUint32 = 4,
    kGgufInt32 = 5,
    kGgufFloat32 = 6,
    kGgufBool = 7,
    kGgufString = 8,
    kGgufArray = 9,
    kGgufUint64 = 10,
    kGgufInt64 = 11,
    kGgufFloat64 = 12,
};

std::size_t checked_u64_size(std::uint64_t value,
                             const std::string& description) {
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(description + " does not fit in size_t");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_product(std::size_t left,
                            std::size_t right,
                            const std::string& description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::runtime_error(description + " size overflows size_t");
    }
    return left * right;
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        const std::string& description) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::runtime_error(description + " size overflows size_t");
    }
    return left + right;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::runtime_error("invalid GGUF alignment");
    }
    const std::size_t remainder = value & (alignment - 1);
    return remainder == 0
        ? value
        : checked_add(value, alignment - remainder, "GGUF alignment");
}

class StreamReader {
public:
    StreamReader(std::ifstream& input, std::size_t file_size)
        : input_(input), file_size_(file_size) {}

    std::size_t position() const noexcept {
        return position_;
    }

    std::uint8_t u8() {
        std::uint8_t value = 0;
        read_exact(&value, sizeof(value));
        return value;
    }

    std::uint16_t u16() {
        const std::uint16_t b0 = u8();
        const std::uint16_t b1 = u8();
        return static_cast<std::uint16_t>(b0 | (b1 << 8));
    }

    std::uint32_t u32() {
        std::uint32_t value = 0;
        for (int byte = 0; byte < 4; ++byte) {
            value |= static_cast<std::uint32_t>(u8()) << (8 * byte);
        }
        return value;
    }

    std::uint64_t u64() {
        std::uint64_t value = 0;
        for (int byte = 0; byte < 8; ++byte) {
            value |= static_cast<std::uint64_t>(u8()) << (8 * byte);
        }
        return value;
    }

    std::int8_t i8() { return static_cast<std::int8_t>(u8()); }
    std::int16_t i16() { return static_cast<std::int16_t>(u16()); }
    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    std::int64_t i64() { return static_cast<std::int64_t>(u64()); }

    float f32() {
        const std::uint32_t bits = u32();
        float result = 0.0f;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    double f64() {
        const std::uint64_t bits = u64();
        double result = 0.0;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    bool boolean() { return u8() != 0; }

    std::string string() {
        const std::size_t length = checked_u64_size(
            u64(), "GGUF string length");
        ensure(length);
        std::string result(length, '\0');
        if (length != 0) {
            read_exact(result.data(), length);
        }
        return result;
    }

    void skip(std::size_t bytes) {
        ensure(bytes);
        if (bytes == 0) {
            return;
        }
        if (bytes > static_cast<std::size_t>(
                        std::numeric_limits<std::streamoff>::max())) {
            throw std::runtime_error("GGUF skip does not fit stream offset");
        }
        input_.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
        if (!input_) {
            throw std::runtime_error("cannot seek through GGUF metadata");
        }
        position_ += bytes;
    }

private:
    void ensure(std::size_t bytes) const {
        if (position_ > file_size_ || bytes > file_size_ - position_) {
            throw std::runtime_error("truncated GGUF file");
        }
    }

    void read_exact(void* destination, std::size_t bytes) {
        ensure(bytes);
        auto* output = static_cast<char*>(destination);
        std::size_t remaining = bytes;
        const std::size_t maximum_chunk = static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max());
        while (remaining != 0) {
            const std::size_t chunk = remaining < maximum_chunk
                ? remaining : maximum_chunk;
            if (!input_.read(output, static_cast<std::streamsize>(chunk))) {
                throw std::runtime_error("cannot read GGUF file");
            }
            output += chunk;
            remaining -= chunk;
            position_ += chunk;
        }
    }

    std::ifstream& input_;
    std::size_t file_size_ = 0;
    std::size_t position_ = 0;
};

struct GgufMetadata {
    std::unordered_map<std::string, std::uint64_t> integers;
    std::unordered_map<std::string, double> floats;
    std::unordered_map<std::string, std::string> strings;
    std::vector<std::string> vocabulary;
    std::vector<std::string> merges;

    std::uint64_t integer(const std::string& key,
                          std::uint64_t fallback = 0) const {
        const auto found = integers.find(key);
        return found == integers.end() ? fallback : found->second;
    }

    double real(const std::string& key, double fallback = 0.0) const {
        const auto found = floats.find(key);
        return found == floats.end() ? fallback : found->second;
    }

    std::string text(const std::string& key,
                     const std::string& fallback = {}) const {
        const auto found = strings.find(key);
        return found == strings.end() ? fallback : found->second;
    }
};

void skip_metadata_value(StreamReader& reader, std::uint32_t type);

void read_metadata_value(StreamReader& reader,
                         std::uint32_t type,
                         const std::string& key,
                         GgufMetadata& metadata) {
    switch (type) {
    case kGgufUint8: metadata.integers[key] = reader.u8(); break;
    case kGgufInt8:
        metadata.integers[key] = static_cast<std::uint64_t>(reader.i8());
        break;
    case kGgufUint16: metadata.integers[key] = reader.u16(); break;
    case kGgufInt16:
        metadata.integers[key] = static_cast<std::uint64_t>(reader.i16());
        break;
    case kGgufUint32: metadata.integers[key] = reader.u32(); break;
    case kGgufInt32:
        metadata.integers[key] = static_cast<std::uint64_t>(reader.i32());
        break;
    case kGgufFloat32: metadata.floats[key] = reader.f32(); break;
    case kGgufBool: metadata.integers[key] = reader.boolean() ? 1 : 0; break;
    case kGgufString: metadata.strings[key] = reader.string(); break;
    case kGgufArray: {
        const std::uint32_t element_type = reader.u32();
        const std::size_t count = checked_u64_size(
            reader.u64(), "GGUF metadata array count");
        if (key == "tokenizer.ggml.tokens" &&
            element_type == kGgufString) {
            metadata.vocabulary.clear();
            metadata.vocabulary.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                metadata.vocabulary.push_back(reader.string());
            }
        } else if (key == "tokenizer.ggml.merges" &&
                   element_type == kGgufString) {
            metadata.merges.clear();
            metadata.merges.reserve(count);
            for (std::size_t index = 0; index < count; ++index) {
                metadata.merges.push_back(reader.string());
            }
        } else {
            for (std::size_t index = 0; index < count; ++index) {
                skip_metadata_value(reader, element_type);
            }
        }
        break;
    }
    case kGgufUint64: metadata.integers[key] = reader.u64(); break;
    case kGgufInt64:
        metadata.integers[key] = static_cast<std::uint64_t>(reader.i64());
        break;
    case kGgufFloat64: metadata.floats[key] = reader.f64(); break;
    default:
        throw std::runtime_error("unsupported GGUF metadata type " +
                                 std::to_string(type));
    }
}

void skip_metadata_value(StreamReader& reader, std::uint32_t type) {
    switch (type) {
    case kGgufUint8:
    case kGgufInt8:
    case kGgufBool:
        reader.skip(1);
        break;
    case kGgufUint16:
    case kGgufInt16:
        reader.skip(2);
        break;
    case kGgufUint32:
    case kGgufInt32:
    case kGgufFloat32:
        reader.skip(4);
        break;
    case kGgufUint64:
    case kGgufInt64:
    case kGgufFloat64:
        reader.skip(8);
        break;
    case kGgufString:
        reader.skip(checked_u64_size(reader.u64(), "GGUF string length"));
        break;
    case kGgufArray: {
        const std::uint32_t element_type = reader.u32();
        const std::size_t count = checked_u64_size(
            reader.u64(), "GGUF nested array count");
        for (std::size_t index = 0; index < count; ++index) {
            skip_metadata_value(reader, element_type);
        }
        break;
    }
    default:
        throw std::runtime_error("unsupported GGUF array element type " +
                                 std::to_string(type));
    }
}

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    MetalGgmlType type = MetalGgmlType::F32;
    std::uint64_t relative_offset = 0;
    std::size_t data_offset = 0;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t row_bytes = 0;
    std::size_t byte_size = 0;
};

std::size_t metadata_size(const GgufMetadata& metadata, const char* key) {
    const auto found = metadata.integers.find(key);
    if (found == metadata.integers.end() || found->second == 0 ||
        found->second > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string("missing or invalid GGUF metadata: ") +
                                 key);
    }
    return static_cast<std::size_t>(found->second);
}

void validate_shape(const TensorInfo& tensor,
                    std::size_t expected_rows,
                    std::size_t expected_cols) {
    if (tensor.rows != expected_rows || tensor.cols != expected_cols) {
        throw std::runtime_error(
            "unexpected shape for tensor " + tensor.name + ": expected " +
            std::to_string(expected_rows) + "x" +
            std::to_string(expected_cols) + ", got " +
            std::to_string(tensor.rows) + "x" +
            std::to_string(tensor.cols));
    }
}

bool is_known_type(std::uint32_t value) {
    switch (static_cast<MetalGgmlType>(value)) {
    case MetalGgmlType::F32:
    case MetalGgmlType::F16:
    case MetalGgmlType::Q4_0:
    case MetalGgmlType::Q4_1:
    case MetalGgmlType::Q5_0:
    case MetalGgmlType::Q5_1:
    case MetalGgmlType::Q8_0:
    case MetalGgmlType::Q8_1:
    case MetalGgmlType::Q2_K:
    case MetalGgmlType::Q3_K:
    case MetalGgmlType::Q4_K:
    case MetalGgmlType::Q5_K:
    case MetalGgmlType::Q6_K:
        return true;
    }
    return false;
}

} // namespace

const char* metal_ggml_type_name(MetalGgmlType type) noexcept {
    switch (type) {
    case MetalGgmlType::F32: return "F32";
    case MetalGgmlType::F16: return "F16";
    case MetalGgmlType::Q4_0: return "Q4_0";
    case MetalGgmlType::Q4_1: return "Q4_1";
    case MetalGgmlType::Q5_0: return "Q5_0";
    case MetalGgmlType::Q5_1: return "Q5_1";
    case MetalGgmlType::Q8_0: return "Q8_0";
    case MetalGgmlType::Q8_1: return "Q8_1";
    case MetalGgmlType::Q2_K: return "Q2_K";
    case MetalGgmlType::Q3_K: return "Q3_K";
    case MetalGgmlType::Q4_K: return "Q4_K";
    case MetalGgmlType::Q5_K: return "Q5_K";
    case MetalGgmlType::Q6_K: return "Q6_K";
    }
    return "UNKNOWN";
}

std::size_t metal_ggml_block_size(MetalGgmlType type) {
    switch (type) {
    case MetalGgmlType::F32:
    case MetalGgmlType::F16:
        return 1;
    case MetalGgmlType::Q4_0:
    case MetalGgmlType::Q4_1:
    case MetalGgmlType::Q5_0:
    case MetalGgmlType::Q5_1:
    case MetalGgmlType::Q8_0:
    case MetalGgmlType::Q8_1:
        return 32;
    case MetalGgmlType::Q2_K:
    case MetalGgmlType::Q3_K:
    case MetalGgmlType::Q4_K:
    case MetalGgmlType::Q5_K:
    case MetalGgmlType::Q6_K:
        return 256;
    }
    throw std::runtime_error("unsupported GGML tensor type");
}

std::size_t metal_ggml_type_size(MetalGgmlType type) {
    switch (type) {
    case MetalGgmlType::F32: return 4;
    case MetalGgmlType::F16: return 2;
    case MetalGgmlType::Q4_0: return 18;
    case MetalGgmlType::Q4_1: return 20;
    case MetalGgmlType::Q5_0: return 22;
    case MetalGgmlType::Q5_1: return 24;
    case MetalGgmlType::Q8_0: return 34;
    case MetalGgmlType::Q8_1: return 40;
    case MetalGgmlType::Q2_K: return 84;
    case MetalGgmlType::Q3_K: return 110;
    case MetalGgmlType::Q4_K: return 144;
    case MetalGgmlType::Q5_K: return 176;
    case MetalGgmlType::Q6_K: return 210;
    }
    throw std::runtime_error("unsupported GGML tensor type");
}

std::size_t metal_ggml_row_bytes(MetalGgmlType type,
                                 std::size_t columns) {
    const std::size_t block_size = metal_ggml_block_size(type);
    if (columns % block_size != 0) {
        throw std::runtime_error(
            std::string(metal_ggml_type_name(type)) +
            " row is not block aligned");
    }
    return checked_product(columns / block_size,
                           metal_ggml_type_size(type),
                           std::string(metal_ggml_type_name(type)) + " row");
}

struct MetalRawModel::Impl {
    explicit Impl(const std::string& gguf_path)
        : path(gguf_path), input(gguf_path, std::ios::binary | std::ios::ate) {
        if (!input) {
            throw std::runtime_error("cannot open model: " + gguf_path);
        }
        const std::streamoff end = input.tellg();
        if (end <= 0 || static_cast<std::uintmax_t>(end) >
                            std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("invalid or oversized GGUF model file");
        }
        file_size = static_cast<std::size_t>(end);
        input.seekg(0, std::ios::beg);
        if (!input) {
            throw std::runtime_error("cannot seek model: " + gguf_path);
        }

        StreamReader reader(input, file_size);
        if (reader.u32() != kGgufMagic) {
            throw std::runtime_error("not a GGUF file");
        }
        const std::uint32_t version = reader.u32();
        if (version != 2 && version != 3) {
            throw std::runtime_error("unsupported GGUF version " +
                                     std::to_string(version));
        }
        const std::size_t tensor_count = checked_u64_size(
            reader.u64(), "GGUF tensor count");
        const std::size_t metadata_count = checked_u64_size(
            reader.u64(), "GGUF metadata count");
        for (std::size_t index = 0; index < metadata_count; ++index) {
            const std::string key = reader.string();
            read_metadata_value(reader, reader.u32(), key, metadata);
        }

        tensors.reserve(tensor_count);
        for (std::size_t index = 0; index < tensor_count; ++index) {
            TensorInfo tensor;
            tensor.name = reader.string();
            const std::uint32_t dimension_count = reader.u32();
            if (dimension_count == 0) {
                throw std::runtime_error("scalar GGUF tensors are unsupported: " +
                                         tensor.name);
            }
            tensor.dimensions.reserve(dimension_count);
            for (std::uint32_t dimension = 0;
                 dimension < dimension_count;
                 ++dimension) {
                tensor.dimensions.push_back(reader.u64());
            }
            const std::uint32_t type_id = reader.u32();
            if (!is_known_type(type_id)) {
                throw std::runtime_error("unsupported GGUF tensor type " +
                                         std::to_string(type_id) +
                                         " for " + tensor.name);
            }
            tensor.type = static_cast<MetalGgmlType>(type_id);
            tensor.relative_offset = reader.u64();
            tensor.cols = checked_u64_size(
                tensor.dimensions.front(), "GGUF tensor dimension");
            tensor.rows = 1;
            for (std::size_t dimension = 1;
                 dimension < tensor.dimensions.size();
                 ++dimension) {
                tensor.rows = checked_product(
                    tensor.rows,
                    checked_u64_size(tensor.dimensions[dimension],
                                     "GGUF tensor dimension"),
                    "GGUF tensor shape");
            }
            tensor.row_bytes = metal_ggml_row_bytes(tensor.type, tensor.cols);
            tensor.byte_size = checked_product(
                tensor.rows, tensor.row_bytes, "tensor " + tensor.name);
            tensors.push_back(std::move(tensor));
        }

        const std::size_t alignment = checked_u64_size(
            metadata.integer("general.alignment", 32), "GGUF alignment");
        const std::size_t data_start = align_up(reader.position(), alignment);
        if (data_start > file_size) {
            throw std::runtime_error("GGUF tensor data starts past end of file");
        }
        for (std::size_t index = 0; index < tensors.size(); ++index) {
            TensorInfo& tensor = tensors[index];
            tensor.data_offset = checked_add(
                data_start,
                checked_u64_size(tensor.relative_offset, "GGUF tensor offset"),
                "GGUF tensor offset");
            if (tensor.data_offset > file_size ||
                tensor.byte_size > file_size - tensor.data_offset) {
                throw std::runtime_error("tensor data is outside GGUF file: " +
                                         tensor.name);
            }
            if (!tensor_index.emplace(tensor.name, index).second) {
                throw std::runtime_error("duplicate GGUF tensor: " + tensor.name);
            }
        }

        initialize_config();
        loaded_tensors.reserve(tensors.size());
    }

    void initialize_config() {
        config.architecture = metadata.text("general.architecture");
        if (config.architecture != "qwen2") {
            throw std::runtime_error(
                "MetalRawModel currently supports qwen2 GGUF models, got " +
                config.architecture);
        }
        config.tensor_count = tensors.size();
        config.layer_count = metadata_size(metadata, "qwen2.block_count");
        config.embedding_size = metadata_size(
            metadata, "qwen2.embedding_length");
        config.feed_forward_size = metadata_size(
            metadata, "qwen2.feed_forward_length");
        config.attention_head_count = metadata_size(
            metadata, "qwen2.attention.head_count");
        config.kv_head_count = metadata_size(
            metadata, "qwen2.attention.head_count_kv");
        config.context_length = metadata_size(metadata, "qwen2.context_length");
        config.vocabulary_size = metadata.vocabulary.size();
        if (config.vocabulary_size == 0) {
            config.vocabulary_size = metadata_size(
                metadata, "tokenizer.ggml.tokens_count");
        }
        config.vocabulary = metadata.vocabulary;
        config.merges = metadata.merges;
        config.tokenizer_pre = metadata.text("tokenizer.ggml.pre");

        if (config.embedding_size % config.attention_head_count != 0) {
            throw std::runtime_error(
                "embedding size must be divisible by attention head count");
        }
        if (config.attention_head_count % config.kv_head_count != 0) {
            throw std::runtime_error(
                "attention head count must be divisible by KV head count");
        }
        config.head_size =
            config.embedding_size / config.attention_head_count;
        config.rotary_dimension = config.head_size;
        config.rope_theta = static_cast<float>(
            metadata.real("qwen2.rope.freq_base", 1000000.0));
        config.norm_epsilon = static_cast<float>(metadata.real(
            "qwen2.attention.layer_norm_rms_epsilon", 1e-6));
        if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f ||
            !std::isfinite(config.norm_epsilon) ||
            config.norm_epsilon <= 0.0f) {
            throw std::runtime_error("invalid RoPE or RMSNorm metadata");
        }
        const std::uint64_t eos = metadata.integer(
            "tokenizer.ggml.eos_token_id", 151645);
        if (eos > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("EOS token id does not fit in int32_t");
        }
        config.eos_token_id = static_cast<std::int32_t>(eos);

        for (const TensorInfo& tensor : tensors) {
            config.parameter_count = checked_add(
                config.parameter_count,
                checked_product(tensor.rows, tensor.cols,
                                "tensor " + tensor.name),
                "model parameter count");
            config.stored_weight_bytes = checked_add(
                config.stored_weight_bytes, tensor.byte_size,
                "stored model weights");
        }
    }

    const TensorInfo& require(const std::string& name) const {
        const auto found = tensor_index.find(name);
        if (found == tensor_index.end()) {
            throw std::runtime_error("missing tensor: " + name);
        }
        return tensors[found->second];
    }

    RawTensor load(const std::string& name,
                   std::size_t expected_rows,
                   std::size_t expected_cols) {
        const TensorInfo& tensor = require(name);
        validate_shape(tensor, expected_rows, expected_cols);
        if (loaded_tensors.find(name) != loaded_tensors.end()) {
            throw std::runtime_error("tensor loaded more than once: " + name);
        }

        RawTensor result;
        result.name = tensor.name;
        result.type = tensor.type;
        result.dimensions = tensor.dimensions;
        result.rows = tensor.rows;
        result.cols = tensor.cols;
        result.row_bytes = tensor.row_bytes;
        result.data.resize(tensor.byte_size);

        input.clear();
        if (tensor.data_offset > static_cast<std::size_t>(
                                     std::numeric_limits<std::streamoff>::max())) {
            throw std::runtime_error("tensor offset does not fit stream: " + name);
        }
        input.seekg(static_cast<std::streamoff>(tensor.data_offset),
                    std::ios::beg);
        if (!input) {
            throw std::runtime_error("cannot seek tensor: " + name);
        }
        std::size_t remaining = result.data.size();
        char* destination = reinterpret_cast<char*>(result.data.data());
        const std::size_t maximum_chunk = static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max());
        while (remaining != 0) {
            const std::size_t chunk = remaining < maximum_chunk
                ? remaining : maximum_chunk;
            if (!input.read(destination, static_cast<std::streamsize>(chunk))) {
                throw std::runtime_error("cannot read tensor: " + name);
            }
            destination += chunk;
            remaining -= chunk;
        }
        loaded_tensors.insert(name);
        return result;
    }

    std::string path;
    std::ifstream input;
    std::size_t file_size = 0;
    GgufMetadata metadata;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, std::size_t> tensor_index;
    std::unordered_set<std::string> loaded_tensors;
    MetalModelConfig config;
};

MetalRawModel::MetalRawModel(const std::string& gguf_path)
    : impl_(std::make_unique<Impl>(gguf_path)) {}

MetalRawModel::~MetalRawModel() = default;
MetalRawModel::MetalRawModel(MetalRawModel&&) noexcept = default;
MetalRawModel& MetalRawModel::operator=(MetalRawModel&&) noexcept = default;

const MetalModelConfig& MetalRawModel::config() const {
    return impl_->config;
}

RawTensor MetalRawModel::load_token_embedding() {
    return impl_->load("token_embd.weight", impl_->config.vocabulary_size,
                       impl_->config.embedding_size);
}

RawTensor MetalRawModel::load_output_norm() {
    return impl_->load("output_norm.weight", 1,
                       impl_->config.embedding_size);
}

RawTensor MetalRawModel::load_output_weight() {
    return impl_->load("output.weight", impl_->config.vocabulary_size,
                       impl_->config.embedding_size);
}

MetalRawLayer MetalRawModel::load_layer(std::size_t layer_index) {
    if (layer_index >= impl_->config.layer_count) {
        throw std::out_of_range("layer index is outside the Metal model");
    }
    const MetalModelConfig& config = impl_->config;
    const std::size_t kv_dimension = checked_product(
        config.kv_head_count, config.head_size, "KV projection");
    const std::string prefix = "blk." + std::to_string(layer_index) + ".";

    MetalRawLayer result;
    result.attn_norm_weight = impl_->load(
        prefix + "attn_norm.weight", 1, config.embedding_size);
    result.attn_q_weight = impl_->load(
        prefix + "attn_q.weight", config.embedding_size,
        config.embedding_size);
    result.attn_q_bias = impl_->load(
        prefix + "attn_q.bias", 1, config.embedding_size);
    result.attn_k_weight = impl_->load(
        prefix + "attn_k.weight", kv_dimension, config.embedding_size);
    result.attn_k_bias = impl_->load(
        prefix + "attn_k.bias", 1, kv_dimension);
    result.attn_v_weight = impl_->load(
        prefix + "attn_v.weight", kv_dimension, config.embedding_size);
    result.attn_v_bias = impl_->load(
        prefix + "attn_v.bias", 1, kv_dimension);
    result.attn_output_weight = impl_->load(
        prefix + "attn_output.weight", config.embedding_size,
        config.embedding_size);
    result.ffn_norm_weight = impl_->load(
        prefix + "ffn_norm.weight", 1, config.embedding_size);
    result.ffn_gate_weight = impl_->load(
        prefix + "ffn_gate.weight", config.feed_forward_size,
        config.embedding_size);
    result.ffn_down_weight = impl_->load(
        prefix + "ffn_down.weight", config.embedding_size,
        config.feed_forward_size);
    result.ffn_up_weight = impl_->load(
        prefix + "ffn_up.weight", config.feed_forward_size,
        config.embedding_size);
    return result;
}

void MetalRawModel::validate_all_tensors_loaded() const {
    if (impl_->loaded_tensors.size() == impl_->tensors.size()) {
        return;
    }
    std::string message =
        "GGUF contains tensors not mapped by MetalRawModel:";
    for (const TensorInfo& tensor : impl_->tensors) {
        if (impl_->loaded_tensors.find(tensor.name) ==
            impl_->loaded_tensors.end()) {
            message += " " + tensor.name;
        }
    }
    throw std::runtime_error(message);
}

} // namespace llm

#include "model.h"

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

// The standalone backends only need this small GGUF reader. It is kept private
// to model.cpp so the public model and backend headers stay compact.
constexpr uint32_t kGgufMagic = 0x46554747U;

enum GgmlType : uint32_t {
    kGgmlF32 = 0,
    kGgmlF16 = 1,
    kGgmlQ5_0 = 6,
    kGgmlQ8_0 = 8,
    kGgmlQ4_K = 12,
    kGgmlQ6_K = 14,
};

enum GgufValueType : uint32_t {
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

class GgufReader {
public:
    explicit GgufReader(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}

    size_t position() const {
        return position_;
    }

    uint8_t u8() {
        ensure(1);
        return bytes_[position_++];
    }

    uint16_t u16() {
        ensure(2);
        const uint16_t value = static_cast<uint16_t>(bytes_[position_]) |
                               (static_cast<uint16_t>(bytes_[position_ + 1]) << 8);
        position_ += 2;
        return value;
    }

    uint32_t u32() {
        ensure(4);
        const uint32_t value = static_cast<uint32_t>(bytes_[position_]) |
                               (static_cast<uint32_t>(bytes_[position_ + 1]) << 8) |
                               (static_cast<uint32_t>(bytes_[position_ + 2]) << 16) |
                               (static_cast<uint32_t>(bytes_[position_ + 3]) << 24);
        position_ += 4;
        return value;
    }

    uint64_t u64() {
        ensure(8);
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i) {
            value |= static_cast<uint64_t>(bytes_[position_ + i]) << (8 * i);
        }
        position_ += 8;
        return value;
    }

    int8_t i8() {
        return static_cast<int8_t>(u8());
    }

    int16_t i16() {
        return static_cast<int16_t>(u16());
    }

    int32_t i32() {
        return static_cast<int32_t>(u32());
    }

    int64_t i64() {
        return static_cast<int64_t>(u64());
    }

    float f32() {
        const uint32_t bits = u32();
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    double f64() {
        const uint64_t bits = u64();
        double value = 0.0;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    bool boolean() {
        return u8() != 0;
    }

    std::string string() {
        const uint64_t length = u64();
        if (length > static_cast<uint64_t>(bytes_.size() - position_)) {
            throw std::runtime_error("invalid GGUF string length");
        }
        std::string value(reinterpret_cast<const char*>(bytes_.data() + position_),
                          static_cast<size_t>(length));
        position_ += static_cast<size_t>(length);
        return value;
    }

    void skip(size_t count) {
        ensure(count);
        position_ += count;
    }

private:
    void ensure(size_t count) const {
        if (position_ > bytes_.size() || count > bytes_.size() - position_) {
            throw std::runtime_error("truncated GGUF file");
        }
    }

    const std::vector<uint8_t>& bytes_;
    size_t position_ = 0;
};

struct GgufMetadata {
    std::unordered_map<std::string, uint64_t> integers;
    std::unordered_map<std::string, double> floats;
    std::unordered_map<std::string, std::string> strings;
    std::vector<std::string> vocabulary;
    std::vector<std::string> merges;

    uint64_t integer(const std::string& key, uint64_t fallback = 0) const {
        const auto it = integers.find(key);
        return it == integers.end() ? fallback : it->second;
    }

    double real(const std::string& key, double fallback = 0.0) const {
        const auto it = floats.find(key);
        return it == floats.end() ? fallback : it->second;
    }

    std::string text(const std::string& key,
                     const std::string& fallback = {}) const {
        const auto it = strings.find(key);
        return it == strings.end() ? fallback : it->second;
    }
};

size_t checked_u64_size(uint64_t value, const std::string& description) {
    if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(description + " does not fit in size_t");
    }
    return static_cast<size_t>(value);
}

void skip_gguf_value(GgufReader& reader, uint32_t type);

void read_gguf_metadata_value(GgufReader& reader,
                              uint32_t type,
                              const std::string& key,
                              GgufMetadata& metadata) {
    switch (type) {
    case kGgufUint8:
        metadata.integers[key] = reader.u8();
        break;
    case kGgufInt8:
        metadata.integers[key] = static_cast<uint64_t>(reader.i8());
        break;
    case kGgufUint16:
        metadata.integers[key] = reader.u16();
        break;
    case kGgufInt16:
        metadata.integers[key] = static_cast<uint64_t>(reader.i16());
        break;
    case kGgufUint32:
        metadata.integers[key] = reader.u32();
        break;
    case kGgufInt32:
        metadata.integers[key] = static_cast<uint64_t>(reader.i32());
        break;
    case kGgufFloat32:
        metadata.floats[key] = reader.f32();
        break;
    case kGgufBool:
        metadata.integers[key] = reader.boolean() ? 1 : 0;
        break;
    case kGgufString:
        metadata.strings[key] = reader.string();
        break;
    case kGgufArray: {
        const uint32_t element_type = reader.u32();
        const uint64_t count = reader.u64();
        const size_t item_count = checked_u64_size(
            count, "GGUF metadata array count");
        if (key == "tokenizer.ggml.tokens" && element_type == kGgufString) {
            metadata.vocabulary.clear();
            metadata.vocabulary.reserve(item_count);
            for (size_t i = 0; i < item_count; ++i) {
                metadata.vocabulary.push_back(reader.string());
            }
        } else if (key == "tokenizer.ggml.merges" && element_type == kGgufString) {
            metadata.merges.clear();
            metadata.merges.reserve(item_count);
            for (size_t i = 0; i < item_count; ++i) {
                metadata.merges.push_back(reader.string());
            }
        } else {
            for (size_t i = 0; i < item_count; ++i) {
                skip_gguf_value(reader, element_type);
            }
        }
        break;
    }
    case kGgufUint64:
        metadata.integers[key] = reader.u64();
        break;
    case kGgufInt64:
        metadata.integers[key] = static_cast<uint64_t>(reader.i64());
        break;
    case kGgufFloat64:
        metadata.floats[key] = reader.f64();
        break;
    default:
        throw std::runtime_error("unsupported GGUF metadata type " +
                                 std::to_string(type));
    }
}

void skip_gguf_value(GgufReader& reader, uint32_t type) {
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
        const uint32_t element_type = reader.u32();
        const size_t count = checked_u64_size(
            reader.u64(), "GGUF nested array count");
        for (size_t i = 0; i < count; ++i) {
            skip_gguf_value(reader, element_type);
        }
        break;
    }
    default:
        throw std::runtime_error("unsupported GGUF array element type " +
                                 std::to_string(type));
    }
}

struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t type = 0;
    uint64_t relative_offset = 0;
    size_t data_offset = 0;

    size_t columns() const {
        return dimensions.empty()
            ? 1
            : checked_u64_size(dimensions.front(), "GGUF tensor dimension");
    }

    size_t rows() const {
        if (dimensions.size() <= 1) {
            return 1;
        }
        size_t count = 1;
        for (size_t i = 1; i < dimensions.size(); ++i) {
            const size_t dimension = checked_u64_size(
                dimensions[i], "GGUF tensor dimension");
            if (dimension != 0 && count >
                std::numeric_limits<size_t>::max() / dimension) {
                throw std::runtime_error("GGUF tensor shape overflows size_t");
            }
            count *= dimension;
        }
        return count;
    }
};

size_t checked_block_bytes(size_t columns,
                          size_t block,
                          size_t bytes,
                          const char* type_name) {
    if (columns % block != 0) {
        throw std::runtime_error(std::string(type_name) +
                                 " row is not block aligned");
    }
    if (columns / block > std::numeric_limits<size_t>::max() / bytes) {
        throw std::runtime_error(std::string(type_name) +
                                 " row byte size overflows size_t");
    }
    return (columns / block) * bytes;
}

size_t gguf_row_bytes(uint32_t type, size_t columns) {
    switch (type) {
    case kGgmlF32:
        if (columns > std::numeric_limits<size_t>::max() / 4) {
            throw std::runtime_error("F32 row byte size overflows size_t");
        }
        return columns * 4;
    case kGgmlF16:
        if (columns > std::numeric_limits<size_t>::max() / 2) {
            throw std::runtime_error("F16 row byte size overflows size_t");
        }
        return columns * 2;
    case kGgmlQ5_0:
        return checked_block_bytes(columns, 32, 22, "Q5_0");
    case kGgmlQ8_0:
        return checked_block_bytes(columns, 32, 34, "Q8_0");
    case kGgmlQ4_K:
        return checked_block_bytes(columns, 256, 144, "Q4_K");
    case kGgmlQ6_K:
        return checked_block_bytes(columns, 256, 210, "Q6_K");
    default:
        throw std::runtime_error("unsupported GGUF tensor type " +
                                 std::to_string(type));
    }
}

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::runtime_error("invalid GGUF alignment");
    }
    const size_t remainder = value & (alignment - 1);
    const size_t padding = remainder == 0 ? 0 : alignment - remainder;
    if (value > std::numeric_limits<size_t>::max() - padding) {
        throw std::runtime_error("GGUF alignment overflows size_t");
    }
    return value + padding;
}

class GgufFile {
public:
    explicit GgufFile(const std::string& path) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) {
            throw std::runtime_error("cannot open model: " + path);
        }
        const std::streamsize stream_size = input.tellg();
        if (stream_size <= 0) {
            throw std::runtime_error("empty model file");
        }
        if (static_cast<uintmax_t>(stream_size) >
            static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error("model file is too large for this process");
        }
        input.seekg(0, std::ios::beg);
        bytes_.resize(static_cast<size_t>(stream_size));
        if (!input.read(reinterpret_cast<char*>(bytes_.data()), stream_size)) {
            throw std::runtime_error("cannot read model: " + path);
        }

        GgufReader reader(bytes_);
        if (reader.u32() != kGgufMagic) {
            throw std::runtime_error("not a GGUF file");
        }
        const uint32_t version = reader.u32();
        if (version != 2 && version != 3) {
            throw std::runtime_error("unsupported GGUF version");
        }
        const size_t tensor_count = checked_u64_size(
            reader.u64(), "GGUF tensor count");
        const size_t metadata_count = checked_u64_size(
            reader.u64(), "GGUF metadata count");
        for (size_t i = 0; i < metadata_count; ++i) {
            const std::string key = reader.string();
            read_gguf_metadata_value(reader, reader.u32(), key, metadata_);
        }

        tensors_.reserve(tensor_count);
        for (size_t i = 0; i < tensor_count; ++i) {
            GgufTensorInfo tensor;
            tensor.name = reader.string();
            const uint32_t dimension_count = reader.u32();
            tensor.dimensions.reserve(dimension_count);
            for (uint32_t d = 0; d < dimension_count; ++d) {
                tensor.dimensions.push_back(reader.u64());
            }
            tensor.type = reader.u32();
            tensor.relative_offset = reader.u64();
            tensors_.push_back(std::move(tensor));
        }

        const size_t alignment = checked_u64_size(
            metadata_.integer("general.alignment", 32),
            "GGUF alignment");
        data_start_ = align_up(reader.position(), alignment);
        if (data_start_ > bytes_.size()) {
            throw std::runtime_error("GGUF tensor data starts past end of file");
        }
        for (size_t tensor_index = 0; tensor_index < tensors_.size();
             ++tensor_index) {
            GgufTensorInfo& tensor = tensors_[tensor_index];
            const size_t relative_offset = checked_u64_size(
                tensor.relative_offset, "GGUF tensor offset");
            if (relative_offset > bytes_.size() - data_start_) {
                throw std::runtime_error("GGUF tensor offset is outside file");
            }
            tensor.data_offset = data_start_ + relative_offset;
            if (!tensor_index_.emplace(tensor.name, tensor_index).second) {
                throw std::runtime_error("duplicate GGUF tensor: " + tensor.name);
            }
        }
    }

    const GgufMetadata& metadata() const {
        return metadata_;
    }

    const std::vector<GgufTensorInfo>& tensors() const {
        return tensors_;
    }

    const GgufTensorInfo& require(const std::string& name) const {
        const auto it = tensor_index_.find(name);
        if (it == tensor_index_.end()) {
            throw std::runtime_error("missing tensor: " + name);
        }
        return tensors_[it->second];
    }

    const uint8_t* pointer(const GgufTensorInfo& tensor, size_t row) const {
        const size_t bytes_per_row = gguf_row_bytes(tensor.type, tensor.columns());
        const size_t row_count = tensor.rows();
        if (row >= row_count || tensor.data_offset > bytes_.size()) {
            throw std::runtime_error("tensor row is outside file: " + tensor.name);
        }
        const size_t remaining = bytes_.size() - tensor.data_offset;
        if (bytes_per_row != 0 && row > remaining / bytes_per_row) {
            throw std::runtime_error("tensor row is outside file: " + tensor.name);
        }
        const size_t row_offset = row * bytes_per_row;
        if (row_offset > remaining || bytes_per_row > remaining - row_offset) {
            throw std::runtime_error("tensor row is outside file: " + tensor.name);
        }
        return bytes_.data() + tensor.data_offset + row_offset;
    }

private:
    std::vector<uint8_t> bytes_;
    GgufMetadata metadata_;
    std::vector<GgufTensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
    size_t data_start_ = 0;
};

uint16_t load_u16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) |
           (static_cast<uint16_t>(data[1]) << 8);
}

int8_t load_i8(const uint8_t* data) {
    return static_cast<int8_t>(*data);
}

float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value >> 15);
    const uint32_t exponent = (value >> 10) & 0x1f;
    const uint32_t mantissa = value & 0x3ff;
    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        const float result = std::ldexp(static_cast<float>(mantissa) / 1024.0f,
                                        -14);
        return sign ? -result : result;
    }
    if (exponent == 31) {
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float result = std::ldexp(
        1.0f + static_cast<float>(mantissa) / 1024.0f,
        static_cast<int>(exponent) - 15);
    return sign ? -result : result;
}

void get_scale_min_k4(int index,
                      const uint8_t* scales,
                      uint8_t& scale,
                      uint8_t& minimum) {
    if (index < 4) {
        scale = scales[index] & 63;
        minimum = scales[index + 4] & 63;
    } else {
        scale = (scales[index + 4] & 0x0f) |
                ((scales[index - 4] >> 6) << 4);
        minimum = (scales[index + 4] >> 4) |
                  ((scales[index] >> 6) << 4);
    }
}

class GgufDequantizer {
public:
    explicit GgufDequantizer(const GgufFile& file) : file_(file) {}

    void row(const GgufTensorInfo& tensor,
             size_t row_index,
             float* output) const {
        const size_t columns = tensor.columns();
        const uint8_t* data = file_.pointer(tensor, row_index);
        switch (tensor.type) {
        case kGgmlF32:
            for (size_t i = 0; i < columns; ++i) {
                const uint32_t bits = static_cast<uint32_t>(data[4 * i]) |
                                      (static_cast<uint32_t>(data[4 * i + 1]) << 8) |
                                      (static_cast<uint32_t>(data[4 * i + 2]) << 16) |
                                      (static_cast<uint32_t>(data[4 * i + 3]) << 24);
                std::memcpy(output + i, &bits, sizeof(float));
            }
            return;
        case kGgmlF16:
            for (size_t i = 0; i < columns; ++i) {
                output[i] = half_to_float(load_u16(data + 2 * i));
            }
            return;
        case kGgmlQ5_0:
            dequant_q5_0(data, columns, output);
            return;
        case kGgmlQ8_0:
            dequant_q8_0(data, columns, output);
            return;
        case kGgmlQ4_K:
            dequant_q4_k(data, columns, output);
            return;
        case kGgmlQ6_K:
            dequant_q6_k(data, columns, output);
            return;
        default:
            throw std::runtime_error("unsupported tensor type in dequantizer");
        }
    }

private:
    static void dequant_q5_0(const uint8_t* data,
                             size_t columns,
                             float* output) {
        for (size_t block = 0; block < columns / 32; ++block) {
            const uint8_t* current = data + block * 22;
            const float d = half_to_float(load_u16(current));
            const uint32_t high_bits = static_cast<uint32_t>(current[2]) |
                                       (static_cast<uint32_t>(current[3]) << 8) |
                                       (static_cast<uint32_t>(current[4]) << 16) |
                                       (static_cast<uint32_t>(current[5]) << 24);
            const uint8_t* quants = current + 6;
            for (int j = 0; j < 16; ++j) {
                const int high0 = static_cast<int>(((high_bits >> j) << 4) & 0x10);
                const int high1 = static_cast<int>((high_bits >> (j + 12)) & 0x10);
                output[block * 32 + j] =
                    (static_cast<int>(quants[j] & 0x0f) | high0) * d - 16.0f * d;
                output[block * 32 + j + 16] =
                    (static_cast<int>(quants[j] >> 4) | high1) * d - 16.0f * d;
            }
        }
    }

    static void dequant_q8_0(const uint8_t* data,
                             size_t columns,
                             float* output) {
        for (size_t block = 0; block < columns / 32; ++block) {
            const uint8_t* current = data + block * 34;
            const float d = half_to_float(load_u16(current));
            for (int j = 0; j < 32; ++j) {
                output[block * 32 + j] =
                    static_cast<float>(load_i8(current + 2 + j)) * d;
            }
        }
    }

    static void dequant_q4_k(const uint8_t* data,
                             size_t columns,
                             float* output) {
        for (size_t block = 0; block < columns / 256; ++block) {
            const uint8_t* current = data + block * 144;
            const float d = half_to_float(load_u16(current));
            const float minimum_scale = half_to_float(load_u16(current + 2));
            const uint8_t* scales = current + 4;
            const uint8_t* quants = current + 16;
            size_t output_index = block * 256;
            int scale_index = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t scale = 0;
                uint8_t minimum = 0;
                get_scale_min_k4(scale_index++, scales, scale, minimum);
                const float d1 = d * scale;
                const float m1 = minimum_scale * minimum;
                get_scale_min_k4(scale_index++, scales, scale, minimum);
                const float d2 = d * scale;
                const float m2 = minimum_scale * minimum;
                const uint8_t* q = quants + (j / 2);
                for (int l = 0; l < 32; ++l) {
                    output[output_index + l] = d1 * (q[l] & 0x0f) - m1;
                    output[output_index + 32 + l] = d2 * (q[l] >> 4) - m2;
                }
                output_index += 64;
            }
        }
    }

    static void dequant_q6_k(const uint8_t* data,
                             size_t columns,
                             float* output) {
        for (size_t block = 0; block < columns / 256; ++block) {
            const uint8_t* current = data + block * 210;
            const uint8_t* lower = current;
            const uint8_t* upper = current + 128;
            const int8_t* scales =
                reinterpret_cast<const int8_t*>(current + 192);
            const float d = half_to_float(load_u16(current + 208));
            size_t output_index = block * 256;
            int scale_offset = 0;
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int q1 = static_cast<int>((lower[l] & 0x0f) |
                                                     ((upper[l] & 0x03) << 4)) - 32;
                    const int q2 = static_cast<int>((lower[l + 32] & 0x0f) |
                                                     (((upper[l] >> 2) & 0x03) << 4)) - 32;
                    const int q3 = static_cast<int>((lower[l] >> 4) |
                                                     (((upper[l] >> 4) & 0x03) << 4)) - 32;
                    const int q4 = static_cast<int>((lower[l + 32] >> 4) |
                                                     (((upper[l] >> 6) & 0x03) << 4)) - 32;
                    const int scale_index = l / 16;
                    output[output_index + n + l] =
                        d * scales[scale_offset + scale_index + 0] * q1;
                    output[output_index + n + l + 32] =
                        d * scales[scale_offset + scale_index + 2] * q2;
                    output[output_index + n + l + 64] =
                        d * scales[scale_offset + scale_index + 4] * q3;
                    output[output_index + n + l + 96] =
                        d * scales[scale_offset + scale_index + 6] * q4;
                }
                lower += 64;
                upper += 32;
                scale_offset += 8;
            }
        }
    }

    const GgufFile& file_;
};

size_t metadata_size(const GgufMetadata& metadata, const char* key) {
    const auto entry = metadata.integers.find(key);
    if (entry == metadata.integers.end()) {
        throw std::runtime_error(std::string("missing GGUF metadata: ") + key);
    }
    if (entry->second == 0 || entry->second > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(std::string("invalid GGUF metadata: ") + key);
    }
    return static_cast<size_t>(entry->second);
}

size_t checked_product(size_t left, size_t right, const std::string& description) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        throw std::runtime_error(description + " size overflows size_t");
    }
    return left * right;
}

void validate_tensor_shape(const GgufTensorInfo& tensor,
                           size_t expected_rows,
                           size_t expected_columns) {
    if (tensor.rows() != expected_rows || tensor.columns() != expected_columns) {
        throw std::runtime_error(
            "unexpected shape for tensor " + tensor.name + ": expected " +
            std::to_string(expected_rows) + "x" + std::to_string(expected_columns) +
            ", got " + std::to_string(tensor.rows()) + "x" +
            std::to_string(tensor.columns()));
    }
}

} // namespace

struct ModelFile::Impl {
    explicit Impl(const std::string& gguf_path)
        : file(gguf_path), dequantizer(file) {
        const GgufMetadata& metadata = file.metadata();

        model_config.architecture = metadata.text("general.architecture");
        if (model_config.architecture != "qwen2") {
            throw std::runtime_error(
                "ModelFile currently supports qwen2 GGUF models, got " +
                model_config.architecture);
        }

        model_config.tensor_count = file.tensors().size();
        model_config.layer_count = metadata_size(metadata, "qwen2.block_count");
        model_config.embedding_size = metadata_size(metadata, "qwen2.embedding_length");
        model_config.feed_forward_size = metadata_size(metadata, "qwen2.feed_forward_length");
        model_config.attention_head_count = metadata_size(metadata, "qwen2.attention.head_count");
        model_config.kv_head_count = metadata_size(metadata, "qwen2.attention.head_count_kv");
        model_config.context_length = metadata_size(metadata, "qwen2.context_length");

        model_config.vocabulary_size = metadata.vocabulary.size();
        if (model_config.vocabulary_size == 0) {
            model_config.vocabulary_size = metadata_size(
                metadata, "tokenizer.ggml.tokens_count");
        }
        model_config.vocabulary = metadata.vocabulary;
        model_config.merges = metadata.merges;
        model_config.tokenizer_pre = metadata.text("tokenizer.ggml.pre");

        if (model_config.embedding_size % model_config.attention_head_count != 0) {
            throw std::runtime_error(
                "embedding size must be divisible by attention head count");
        }
        if (model_config.attention_head_count % model_config.kv_head_count != 0) {
            throw std::runtime_error(
                "attention head count must be divisible by KV head count");
        }
        model_config.head_size =
            model_config.embedding_size / model_config.attention_head_count;
        model_config.rotary_dimension = model_config.head_size;

        model_config.rope_theta = static_cast<float>(
            metadata.real("qwen2.rope.freq_base", 1000000.0));
        model_config.norm_epsilon = static_cast<float>(
            metadata.real("qwen2.attention.layer_norm_rms_epsilon", 1e-6));
        if (!std::isfinite(model_config.rope_theta) || model_config.rope_theta <= 0.0f) {
            throw std::runtime_error("RoPE theta must be finite and positive");
        }
        if (!std::isfinite(model_config.norm_epsilon) || model_config.norm_epsilon <= 0.0f) {
            throw std::runtime_error("RMSNorm epsilon must be finite and positive");
        }

        const uint64_t eos = metadata.integer("tokenizer.ggml.eos_token_id", 151645);
        if (eos > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("EOS token id does not fit in int32_t");
        }
        model_config.eos_token_id = static_cast<int32_t>(eos);

        for (const GgufTensorInfo& tensor : file.tensors()) {
            const size_t tensor_parameters = checked_product(
                tensor.rows(), tensor.columns(), "tensor " + tensor.name);
            if (tensor_parameters >
                std::numeric_limits<size_t>::max() - model_config.parameter_count) {
                throw std::runtime_error("model parameter count overflows size_t");
            }
            model_config.parameter_count += tensor_parameters;
        }
        model_config.dequantized_bytes = checked_product(
            model_config.parameter_count, sizeof(float), "dequantized model");
        loaded_tensors.reserve(model_config.tensor_count);
    }

    Vector require_vector(const std::string& name, size_t expected_length) {
        return dequantize_vector(file.require(name), expected_length);
    }

    Matrix require_matrix(const std::string& name,
                          size_t expected_rows,
                          size_t expected_columns) {
        const GgufTensorInfo& tensor = file.require(name);
        validate_tensor_shape(tensor, expected_rows, expected_columns);

        // Preserve the tensor's native GGUF [output, input] layout. The
        // The backend selects gemmt/gemmtb when it needs input * weight^T.
        Matrix result(expected_rows, expected_columns);
        for (size_t source_row_index = 0;
             source_row_index < expected_rows;
             ++source_row_index) {
            dequantizer.row(
                tensor, source_row_index,
                result.values[source_row_index].values.data());
        }
        mark_loaded(tensor);
        return result;
    }

    void validate_all_tensors_loaded() const {
        if (loaded_tensors.size() == model_config.tensor_count) {
            return;
        }

        std::string message = "GGUF contains tensors not mapped by ModelFile:";
        for (const GgufTensorInfo& tensor : file.tensors()) {
            if (loaded_tensors.find(tensor.name) == loaded_tensors.end()) {
                message += " " + tensor.name;
            }
        }
        throw std::runtime_error(message);
    }

    GgufFile file;
    GgufDequantizer dequantizer;
    ModelConfig model_config;
    std::unordered_set<std::string> loaded_tensors;

private:
    Vector dequantize_vector(const GgufTensorInfo& tensor,
                             size_t expected_length) {
        validate_tensor_shape(tensor, 1, expected_length);
        Vector result(expected_length);
        dequantizer.row(tensor, 0, result.values.data());
        mark_loaded(tensor);
        return result;
    }

    void mark_loaded(const GgufTensorInfo& tensor) {
        if (!loaded_tensors.insert(tensor.name).second) {
            throw std::runtime_error("tensor loaded more than once: " + tensor.name);
        }
    }
};

ModelFile::ModelFile(const std::string& gguf_path)
    : impl_(std::make_unique<Impl>(gguf_path)) {}

ModelFile::~ModelFile() = default;

const ModelConfig& ModelFile::config() const {
    return impl_->model_config;
}

Matrix ModelFile::load_token_embedding() {
    return impl_->require_matrix(
        "token_embd.weight",
        impl_->model_config.vocabulary_size,
        impl_->model_config.embedding_size);
}

Vector ModelFile::load_output_norm() {
    return impl_->require_vector(
        "output_norm.weight", impl_->model_config.embedding_size);
}

Matrix ModelFile::load_output_weight() {
    return impl_->require_matrix(
        "output.weight",
        impl_->model_config.vocabulary_size,
        impl_->model_config.embedding_size);
}

void ModelFile::validate_all_tensors_loaded() const {
    impl_->validate_all_tensors_loaded();
}

Layer::Layer(ModelFile& model_file, size_t layer_index) {
    ModelFile::Impl& loader = *model_file.impl_;
    const ModelConfig& config = loader.model_config;
    if (layer_index >= config.layer_count) {
        throw std::out_of_range("layer index is outside the model");
    }

    const std::string prefix = "blk." + std::to_string(layer_index) + ".";
    const size_t kv_projection_size = checked_product(
        config.kv_head_count, config.head_size, "KV projection");

    attn_norm_weight = loader.require_vector(
        prefix + "attn_norm.weight", config.embedding_size);
    attn_q_weight = loader.require_matrix(
        prefix + "attn_q.weight", config.embedding_size, config.embedding_size);
    attn_q_bias = loader.require_vector(
        prefix + "attn_q.bias", config.embedding_size);
    attn_k_weight = loader.require_matrix(
        prefix + "attn_k.weight", kv_projection_size, config.embedding_size);
    attn_k_bias = loader.require_vector(
        prefix + "attn_k.bias", kv_projection_size);
    attn_v_weight = loader.require_matrix(
        prefix + "attn_v.weight", kv_projection_size, config.embedding_size);
    attn_v_bias = loader.require_vector(
        prefix + "attn_v.bias", kv_projection_size);
    attn_output_weight = loader.require_matrix(
        prefix + "attn_output.weight", config.embedding_size, config.embedding_size);
    ffn_norm_weight = loader.require_vector(
        prefix + "ffn_norm.weight", config.embedding_size);
    ffn_gate_weight = loader.require_matrix(
        prefix + "ffn_gate.weight", config.feed_forward_size, config.embedding_size);
    ffn_down_weight = loader.require_matrix(
        prefix + "ffn_down.weight", config.embedding_size, config.feed_forward_size);
    ffn_up_weight = loader.require_matrix(
        prefix + "ffn_up.weight", config.feed_forward_size, config.embedding_size);
}

} // namespace llm

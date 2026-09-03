#include "chat.h"

#include "cpu_llm.h"
#include "metal_llm.h"
#include "moe_llm.h"
#include "model.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace chat {
namespace {

struct Codepoint {
    uint32_t value = 0;
    size_t begin = 0;
    size_t end = 0;
};

std::vector<Codepoint> decode_utf8(const std::string & text) {
    std::vector<Codepoint> result;
    result.reserve(text.size());
    for (size_t position = 0; position < text.size();) {
        const size_t begin = position;
        const uint8_t first = static_cast<uint8_t>(text[position]);
        uint32_t value = first;
        size_t length = 1;
        if ((first & 0xe0U) == 0xc0U && position + 1 < text.size()) {
            const uint8_t second = static_cast<uint8_t>(text[position + 1]);
            if ((second & 0xc0U) == 0x80U) {
                value = (static_cast<uint32_t>(first & 0x1fU) << 6) |
                        static_cast<uint32_t>(second & 0x3fU);
                length = 2;
            }
        } else if ((first & 0xf0U) == 0xe0U && position + 2 < text.size()) {
            const uint8_t second = static_cast<uint8_t>(text[position + 1]);
            const uint8_t third = static_cast<uint8_t>(text[position + 2]);
            if ((second & 0xc0U) == 0x80U && (third & 0xc0U) == 0x80U) {
                value = (static_cast<uint32_t>(first & 0x0fU) << 12) |
                        (static_cast<uint32_t>(second & 0x3fU) << 6) |
                        static_cast<uint32_t>(third & 0x3fU);
                length = 3;
            }
        } else if ((first & 0xf8U) == 0xf0U && position + 3 < text.size()) {
            const uint8_t second = static_cast<uint8_t>(text[position + 1]);
            const uint8_t third = static_cast<uint8_t>(text[position + 2]);
            const uint8_t fourth = static_cast<uint8_t>(text[position + 3]);
            if ((second & 0xc0U) == 0x80U && (third & 0xc0U) == 0x80U &&
                (fourth & 0xc0U) == 0x80U) {
                value = (static_cast<uint32_t>(first & 0x07U) << 18) |
                        (static_cast<uint32_t>(second & 0x3fU) << 12) |
                        (static_cast<uint32_t>(third & 0x3fU) << 6) |
                        static_cast<uint32_t>(fourth & 0x3fU);
                length = 4;
            }
        }
        // Invalid UTF-8 is retained byte-for-byte. Console input is normally
        // UTF-8, but this fallback keeps the tokenizer deterministic.
        result.push_back({value, begin, begin + length});
        position += length;
    }
    return result;
}

void append_utf8(uint32_t value, std::string & output) {
    if (value <= 0x7fU) {
        output.push_back(static_cast<char>(value));
    } else if (value <= 0x7ffU) {
        output.push_back(static_cast<char>(0xc0U | (value >> 6)));
        output.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
    } else if (value <= 0xffffU) {
        output.push_back(static_cast<char>(0xe0U | (value >> 12)));
        output.push_back(static_cast<char>(0x80U | ((value >> 6) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
    } else {
        output.push_back(static_cast<char>(0xf0U | (value >> 18)));
        output.push_back(static_cast<char>(0x80U | ((value >> 12) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | ((value >> 6) & 0x3fU)));
        output.push_back(static_cast<char>(0x80U | (value & 0x3fU)));
    }
}

bool is_ascii_letter(uint32_t value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}

bool is_number(uint32_t value);

bool is_mark(uint32_t value) {
    // The qwen35 pre-tokenizer differs from qwen2 by attaching Unicode mark
    // characters to letter runs. These ranges cover the combining-mark blocks
    // and the script-specific marks most likely to be separated by the coarse
    // script ranges in is_letter().
    return (value >= 0x0300U && value <= 0x036fU) ||
           (value >= 0x0483U && value <= 0x0489U) ||
           (value >= 0x0591U && value <= 0x05bdU) || value == 0x05bfU ||
           (value >= 0x05c1U && value <= 0x05c2U) ||
           (value >= 0x05c4U && value <= 0x05c5U) || value == 0x05c7U ||
           (value >= 0x0610U && value <= 0x061aU) ||
           (value >= 0x064bU && value <= 0x065fU) || value == 0x0670U ||
           (value >= 0x06d6U && value <= 0x06dcU) ||
           (value >= 0x06dfU && value <= 0x06e4U) ||
           (value >= 0x06e7U && value <= 0x06e8U) ||
           (value >= 0x06eaU && value <= 0x06edU) || value == 0x0711U ||
           (value >= 0x0730U && value <= 0x074aU) ||
           (value >= 0x1ab0U && value <= 0x1affU) ||
           (value >= 0x1dc0U && value <= 0x1dffU) ||
           (value >= 0x20d0U && value <= 0x20ffU) ||
           (value >= 0xfe00U && value <= 0xfe0fU) ||
           (value >= 0xfe20U && value <= 0xfe2fU) ||
           (value >= 0xe0100U && value <= 0xe01efU);
}

bool is_letter(uint32_t value) {
    if (is_ascii_letter(value)) {
        return true;
    }
    if (is_number(value) || is_mark(value)) {
        return false;
    }
    // These ranges cover the scripts normally encountered by the Qwen chat
    // demo. The byte-level tokenizer only needs the coarse L/N distinction.
    return (value >= 0x00c0U && value <= 0x02ffU) ||
           (value >= 0x0370U && value <= 0x052fU) ||
           (value >= 0x0530U && value <= 0x058fU) ||
           (value >= 0x0600U && value <= 0x06ffU) ||
           (value >= 0x0700U && value <= 0x074fU) ||
           (value >= 0x0900U && value <= 0x0dffU) ||
           (value >= 0x0e00U && value <= 0x0e7fU) ||
           (value >= 0x10a0U && value <= 0x10ffU) ||
           (value >= 0x1100U && value <= 0x11ffU) ||
           (value >= 0x3040U && value <= 0x30ffU) ||
           (value >= 0x3400U && value <= 0x9fffU) ||
           (value >= 0xac00U && value <= 0xd7afU) ||
           (value >= 0xf900U && value <= 0xfaffU) ||
           (value >= 0x20000U && value <= 0x323afU);
}

bool is_number(uint32_t value) {
    return (value >= '0' && value <= '9') ||
           (value >= 0x0660U && value <= 0x0669U) ||
           (value >= 0x06f0U && value <= 0x06f9U) ||
           (value >= 0x0966U && value <= 0x096fU) ||
           (value >= 0xff10U && value <= 0xff19U);
}

bool is_space(uint32_t value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
           value == '\f' || value == '\v' || value == 0x00a0U ||
           value == 0x1680U || (value >= 0x2000U && value <= 0x200aU) ||
           value == 0x2028U || value == 0x2029U || value == 0x202fU ||
           value == 0x205fU || value == 0x3000U;
}

bool is_line_break(uint32_t value) {
    return value == '\n' || value == '\r';
}

bool starts_with_case_insensitive(const std::vector<Codepoint> & points,
                                  size_t position,
                                  const std::string & suffix) {
    if (position + suffix.size() > points.size()) {
        return false;
    }
    for (size_t index = 0; index < suffix.size(); ++index) {
        uint32_t expected = static_cast<unsigned char>(suffix[index]);
        uint32_t actual = points[position + index].value;
        if (expected >= 'A' && expected <= 'Z') expected += 'a' - 'A';
        if (actual >= 'A' && actual <= 'Z') actual += 'a' - 'A';
        if (actual != expected) return false;
    }
    return true;
}

size_t contraction_length(const std::vector<Codepoint> & points, size_t position) {
    if (position >= points.size() || points[position].value != '\'') {
        return 0;
    }
    if (starts_with_case_insensitive(points, position, "'s") ||
        starts_with_case_insensitive(points, position, "'t") ||
        starts_with_case_insensitive(points, position, "'m") ||
        starts_with_case_insensitive(points, position, "'d")) {
        return 2;
    }
    if (starts_with_case_insensitive(points, position, "'re") ||
        starts_with_case_insensitive(points, position, "'ve") ||
        starts_with_case_insensitive(points, position, "'ll")) {
        return 3;
    }
    return 0;
}

std::vector<std::pair<size_t, size_t>> pretokenize(
    const std::string & text,
    bool include_marks) {
    const std::vector<Codepoint> points = decode_utf8(text);
    std::vector<std::pair<size_t, size_t>> pieces;
    const auto is_word_character = [include_marks](uint32_t value) {
        return is_letter(value) || (include_marks && is_mark(value));
    };
    size_t position = 0;
    while (position < points.size()) {
        const size_t begin = position;
        const size_t contraction = contraction_length(points, position);
        if (contraction != 0) {
            pieces.emplace_back(points[position].begin,
                                points[position + contraction - 1].end);
            position += contraction;
            continue;
        }

        // qwen2 uses L+ and qwen35 uses [L M]+. In both cases the optional
        // prefix is consumed together with the following word-character run.
        if ((!is_line_break(points[position].value) &&
             !is_number(points[position].value)) &&
            (is_word_character(points[position].value) ||
             (position + 1 < points.size() &&
              is_word_character(points[position + 1].value)))) {
            ++position;
            while (position < points.size() &&
                   is_word_character(points[position].value)) {
                ++position;
            }
            pieces.emplace_back(points[begin].begin, points[position - 1].end);
            continue;
        }

        // The Qwen2 regex treats each numeric codepoint as a separate piece.
        if (is_number(points[position].value)) {
            ++position;
            pieces.emplace_back(points[begin].begin, points[position - 1].end);
            continue;
        }

        // ?[^\s\p{L}\p{N}]+[\r\n]*: an ASCII space may prefix a
        // punctuation run, but other whitespace cannot.
        const bool has_space_prefix = points[position].value == ' ';
        size_t punctuation = position + (has_space_prefix ? 1 : 0);
        if (punctuation < points.size() &&
            !is_space(points[punctuation].value) &&
            !is_word_character(points[punctuation].value) &&
            !is_number(points[punctuation].value)) {
            position = punctuation;
            while (position < points.size() &&
                   !is_space(points[position].value) &&
                   !is_word_character(points[position].value) &&
                   !is_number(points[position].value)) {
                ++position;
            }
            while (position < points.size() && is_line_break(points[position].value)) {
                ++position;
            }
            pieces.emplace_back(points[begin].begin, points[position - 1].end);
            continue;
        }

        size_t whitespace_count = 0;
        size_t last_line_break_end = 0;
        while (position + whitespace_count < points.size() &&
               is_space(points[position + whitespace_count].value)) {
            const uint32_t value = points[position + whitespace_count].value;
            if (is_line_break(value)) {
                last_line_break_end = position + whitespace_count + 1;
            }
            ++whitespace_count;
        }
        if (last_line_break_end != 0) {
            position = last_line_break_end;
            pieces.emplace_back(points[begin].begin, points[position - 1].end);
            continue;
        }
        // \s+(?!\S): when a run of multiple spaces is followed by text,
        // leave its final space for the next letter/punctuation match.
        if (whitespace_count > 1 && position + whitespace_count < points.size()) {
            position += whitespace_count - 1;
            pieces.emplace_back(points[begin].begin, points[position - 1].end);
            continue;
        }
        if (whitespace_count > 0) {
            position += whitespace_count;
            pieces.emplace_back(points[begin].begin, points[position - 1].end);
            continue;
        }

        // Keep malformed or otherwise unclassified codepoints lossless.
        ++position;
        pieces.emplace_back(points[begin].begin, points[position - 1].end);
    }
    return pieces;
}

std::vector<uint32_t> to_codepoints(const std::string & text) {
    std::vector<uint32_t> result;
    for (const Codepoint & point : decode_utf8(text)) result.push_back(point.value);
    return result;
}

std::string from_codepoints(const std::vector<uint32_t> & points) {
    std::string result;
    for (uint32_t point : points) append_utf8(point, result);
    return result;
}

std::string merge_key(const std::vector<uint32_t> & first,
                      const std::vector<uint32_t> & second) {
    std::string result = from_codepoints(first);
    // GPT-2 byte-encoded symbols never contain an embedded NUL, so it is a
    // safe, allocation-free separator for the two BPE operands.
    result.push_back('\0');
    result += from_codepoints(second);
    return result;
}

std::vector<uint32_t> byte_encode(const std::string & text) {
    std::vector<int> bytes;
    for (int value = 33; value <= 126; ++value) bytes.push_back(value);
    for (int value = 161; value <= 172; ++value) bytes.push_back(value);
    for (int value = 174; value <= 255; ++value) bytes.push_back(value);

    std::vector<uint32_t> codepoints;
    codepoints.reserve(256);
    for (int value : bytes) codepoints.push_back(static_cast<uint32_t>(value));
    int extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
            bytes.push_back(value);
            codepoints.push_back(static_cast<uint32_t>(256 + extra++));
        }
    }

    std::vector<uint32_t> result;
    result.reserve(text.size());
    for (unsigned char byte : text) {
        const auto it = std::find(bytes.begin(), bytes.end(), static_cast<int>(byte));
        if (it == bytes.end()) throw std::runtime_error("byte encoder lookup failed");
        result.push_back(codepoints[static_cast<size_t>(it - bytes.begin())]);
    }
    return result;
}

class QwenTokenizer {
public:
    QwenTokenizer(const std::vector<std::string> & vocabulary,
                  const std::vector<std::string> & merges,
                  const std::string & tokenizer_pre)
        : vocabulary_(vocabulary),
          include_marks_(tokenizer_pre == "qwen35") {
        for (size_t index = 0; index < vocabulary.size(); ++index) {
            vocabulary_ids_.emplace(vocabulary[index], static_cast<int32_t>(index));
            if (vocabulary[index].size() >= 4 && vocabulary[index].compare(0, 2, "<|") == 0 &&
                vocabulary[index].compare(vocabulary[index].size() - 2, 2, "|>") == 0) {
                special_tokens_.emplace_back(vocabulary[index], static_cast<int32_t>(index));
                special_ids_.insert(static_cast<int32_t>(index));
            }
        }
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto & left, const auto & right) {
                      return left.first.size() > right.first.size();
                  });

        for (size_t rank = 0; rank < merges.size(); ++rank) {
            const std::string & merge = merges[rank];
            const size_t separator = merge.find(' ', 1);
            if (separator == std::string::npos || separator == 0 || separator + 1 >= merge.size()) {
                continue;
            }
            const std::vector<uint32_t> first = to_codepoints(merge.substr(0, separator));
            const std::vector<uint32_t> second = to_codepoints(merge.substr(separator + 1));
            if (first.empty() || second.empty()) continue;
            merge_ranks_.emplace(merge_key(first, second), rank);
        }
        byte_decoder_ = make_byte_decoder();
    }

    std::vector<int32_t> encode(const std::string & text) const {
        std::vector<int32_t> result;
        size_t position = 0;
        while (position < text.size()) {
            const auto special = special_at(text, position);
            if (special.first != -1) {
                result.push_back(special.first);
                position += special.second;
                continue;
            }

            size_t next_special = text.size();
            for (const auto & token : special_tokens_) {
                const size_t candidate = text.find(token.first, position + 1);
                if (candidate != std::string::npos) next_special = std::min(next_special, candidate);
            }
            const std::string ordinary = text.substr(position, next_special - position);
            encode_ordinary(ordinary, result);
            position = next_special;
        }
        return result;
    }

    int32_t id(const std::string & token) const {
        const auto it = vocabulary_ids_.find(token);
        return it == vocabulary_ids_.end() ? -1 : it->second;
    }

    std::string decode_piece(int32_t token) const {
        if (token < 0 || static_cast<size_t>(token) >= vocabulary_.size()) return {};
        const std::string & piece = vocabulary_[static_cast<size_t>(token)];
        if (id_is_special(token)) return {};
        std::string result;
        for (const Codepoint & point : decode_utf8(piece)) {
            const auto it = byte_decoder_.find(point.value);
            if (it == byte_decoder_.end()) {
                append_utf8(point.value, result);
            } else {
                result.push_back(static_cast<char>(it->second));
            }
        }
        return result;
    }

    bool is_special(int32_t token) const { return id_is_special(token); }

private:
    static std::unordered_map<uint32_t, uint8_t> make_byte_decoder() {
        std::vector<int> bytes;
        for (int value = 33; value <= 126; ++value) bytes.push_back(value);
        for (int value = 161; value <= 172; ++value) bytes.push_back(value);
        for (int value = 174; value <= 255; ++value) bytes.push_back(value);
        std::vector<uint32_t> codepoints;
        for (int value : bytes) codepoints.push_back(static_cast<uint32_t>(value));
        int extra = 0;
        for (int value = 0; value < 256; ++value) {
            if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
                bytes.push_back(value);
                codepoints.push_back(static_cast<uint32_t>(256 + extra++));
            }
        }
        std::unordered_map<uint32_t, uint8_t> result;
        for (size_t index = 0; index < bytes.size(); ++index) {
            result[codepoints[index]] = static_cast<uint8_t>(bytes[index]);
        }
        return result;
    }

    std::pair<int32_t, size_t> special_at(const std::string & text, size_t position) const {
        for (const auto & token : special_tokens_) {
            if (text.compare(position, token.first.size(), token.first) == 0) {
                return {token.second, token.first.size()};
            }
        }
        return {-1, 0};
    }

    bool id_is_special(int32_t token) const {
        return special_ids_.find(token) != special_ids_.end();
    }

    void encode_ordinary(const std::string & text, std::vector<int32_t> & output) const {
        for (const auto & range : pretokenize(text, include_marks_)) {
            const std::string piece = text.substr(range.first, range.second - range.first);
            encode_piece_with_chunks(piece, output);
        }
    }

    void encode_piece_with_chunks(const std::string & piece,
                                  std::vector<int32_t> & output) const {
        struct Chunk { std::vector<uint32_t> symbols; };
        std::vector<Chunk> chunks;
        for (uint32_t symbol : byte_encode(piece)) chunks.push_back({{symbol}});
        while (chunks.size() > 1) {
            size_t best_index = chunks.size();
            size_t best_rank = std::numeric_limits<size_t>::max();
            for (size_t index = 0; index + 1 < chunks.size(); ++index) {
                const auto rank = merge_ranks_.find(
                    merge_key(chunks[index].symbols, chunks[index + 1].symbols));
                if (rank != merge_ranks_.end() && rank->second < best_rank) {
                    best_rank = rank->second;
                    best_index = index;
                }
            }
            if (best_index == chunks.size()) break;
            chunks[best_index].symbols.insert(
                chunks[best_index].symbols.end(),
                chunks[best_index + 1].symbols.begin(), chunks[best_index + 1].symbols.end());
            chunks.erase(chunks.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
        }

        for (const Chunk & chunk : chunks) {
            const std::string token = from_codepoints(chunk.symbols);
            const auto found = vocabulary_ids_.find(token);
            if (found != vocabulary_ids_.end()) {
                output.push_back(found->second);
                continue;
            }
            // Every byte symbol is part of a GPT-2 vocabulary. This fallback
            // gives a useful error if a malformed/incomplete merge table is
            // encountered while still allowing valid single-byte tokens.
            for (uint32_t symbol : chunk.symbols) {
                std::string one;
                append_utf8(symbol, one);
                const auto byte_token = vocabulary_ids_.find(one);
                if (byte_token == vocabulary_ids_.end()) {
                    throw std::runtime_error("tokenizer vocabulary is missing piece: " + token);
                }
                output.push_back(byte_token->second);
            }
        }
    }

    const std::vector<std::string> & vocabulary_;
    std::unordered_map<std::string, int32_t> vocabulary_ids_;
    std::vector<std::pair<std::string, int32_t>> special_tokens_;
    std::unordered_set<int32_t> special_ids_;
    std::unordered_map<std::string, size_t> merge_ranks_;
    std::unordered_map<uint32_t, uint8_t> byte_decoder_;
    bool include_marks_ = false;
};

struct Options {
    std::string model_path;
    int max_new_tokens = 32;
    std::size_t max_sequence = 0;
    std::size_t expert_cache_count = 8;
    bool raw_prompt = false;
    bool use_gpu = false;
    bool help_requested = false;
    bool profiling_enabled = true;
    bool metal_kernel_profile = false;
    std::string profile_csv_path = "llm_profile.csv";
    std::string system_prompt = "You are a concise and helpful assistant.";
};

void usage(const char * program) {
    std::cerr << "Usage: " << program << " --model MODEL.gguf [options]\n"
              << "  --tokens N       Maximum generated tokens per input, default 32\n"
              << "  --max-sequence N Maximum sequence capacity (backend-specific default)\n"
              << "  --expert-cache N Per-layer routed-expert cache slots, default 8\n"
              << "  --system TEXT    System message used by ChatML\n"
              << "  --raw            Send input directly without ChatML wrapping\n"
              << "  --gpu            Use the Metal GPU backend (no CPU fallback)\n"
              << "  --profile-csv P  Write timing/statistics CSV under output/\n"
              << "                   (default output/llm_profile.csv)\n"
              << "  --profile-log P  Compatibility alias for --profile-csv\n"
              << "  --metal-kernel-profile\n"
              << "                   Record per-kernel Metal GPU timestamps\n"
              << "                   (diagnostic mode; may perturb scheduling)\n"
              << "  --no-profile     Disable timing/statistics collection\n"
              << "  --help           Show this help\n";
}

std::string prepare_profile_output_path(const std::string & requested_path) {
    namespace fs = std::filesystem;

    if (requested_path.empty()) {
        throw std::invalid_argument("Profile output filename cannot be empty");
    }

    fs::path path(requested_path);
    if (!path.is_absolute()) {
        path = path.lexically_normal();
        const auto first = path.begin();
        if (first == path.end() || *first == "..") {
            throw std::invalid_argument(
                "Relative profile output path cannot escape the output directory");
        }
        if (*first != "output") {
            path = fs::path("output") / path;
        }
    }

    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code error;
        fs::create_directories(parent, error);
        if (error) {
            throw std::runtime_error(
                "Cannot create profile output directory " + parent.string() +
                ": " + error.message());
        }
    }
    return path.string();
}

bool parse_options(int argc, char ** argv, Options & options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&](const char * name) -> const char * {
            if (index + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++index];
        };
        if (argument == "--model") {
            options.model_path = value("--model");
        } else if (argument == "--tokens") {
            options.max_new_tokens = std::stoi(value("--tokens"));
        } else if (argument == "--max-sequence") {
            const unsigned long long parsed = std::stoull(
                value("--max-sequence"));
            if (parsed > static_cast<unsigned long long>(
                              std::numeric_limits<std::size_t>::max())) {
                throw std::out_of_range(
                    "--max-sequence does not fit in size_t");
            }
            options.max_sequence = static_cast<std::size_t>(parsed);
        } else if (argument == "--expert-cache") {
            const unsigned long long parsed = std::stoull(
                value("--expert-cache"));
            if (parsed > static_cast<unsigned long long>(
                              std::numeric_limits<std::size_t>::max())) {
                throw std::out_of_range(
                    "--expert-cache does not fit in size_t");
            }
            options.expert_cache_count = static_cast<std::size_t>(parsed);
        } else if (argument == "--system") {
            options.system_prompt = value("--system");
        } else if (argument == "--raw") {
            options.raw_prompt = true;
        } else if (argument == "--gpu") {
            options.use_gpu = true;
        } else if (argument == "--profile-csv") {
            options.profile_csv_path = value("--profile-csv");
        } else if (argument == "--profile-log") {
            // Keep existing scripts working; the selected file is still CSV.
            options.profile_csv_path = value("--profile-log");
        } else if (argument == "--metal-kernel-profile") {
            options.metal_kernel_profile = true;
        } else if (argument == "--no-profile") {
            options.profiling_enabled = false;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            options.help_requested = true;
            return true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (!options.help_requested && (options.model_path.empty() || options.max_new_tokens <= 0)) {
        usage(argv[0]);
        return false;
    }
    return true;
}

std::string chat_prompt(const std::string & system, const std::string & user) {
    return "<|im_start|>system\n" + system +
           "<|im_end|>\n<|im_start|>user\n" + user +
           "<|im_end|>\n<|im_start|>assistant\n";
}

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point begin, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool contains_token(const std::vector<int32_t> & token_ids, int32_t token) {
    return std::find(token_ids.begin(), token_ids.end(), token) != token_ids.end();
}

bool is_stop_token(int32_t token, const std::vector<int32_t> & stop_token_ids) {
    return contains_token(stop_token_ids, token);
}

template <typename Backend>
void validate_generation_request(
    const Backend & backend,
    const std::vector<int32_t> & initial_sequence,
    size_t max_new_tokens) {
    if (initial_sequence.empty()) {
        throw std::invalid_argument("Generation requires at least one initial token");
    }
    if (max_new_tokens == 0) {
        throw std::invalid_argument("Generation requires max_new_tokens > 0");
    }
    if (backend.max_sequence() == 0) {
        throw std::runtime_error("Backend sequence capacity is zero");
    }
    if (initial_sequence.size() >= backend.max_sequence()) {
        throw std::invalid_argument(
            "Initial token sequence leaves no room for generated tokens");
    }
    const auto & config = backend.config();
    if (config.layer_count == 0 || config.vocabulary_size == 0 ||
        config.context_length == 0) {
        throw std::runtime_error("Backend model metadata is incomplete");
    }
}

template <typename Backend>
size_t generation_limit(const Backend & backend,
                        const std::vector<int32_t> & initial_sequence,
                        size_t max_new_tokens) {
    return std::min(
        max_new_tokens, backend.max_sequence() - initial_sequence.size());
}

template <typename Backend>
std::vector<int32_t> make_stop_tokens(const QwenTokenizer & tokenizer,
                                      const Backend & backend,
                                      int32_t im_end,
                                      int32_t slash_s) {
    const auto & config = backend.config();
    std::vector<int32_t> result;
    auto add = [&](int32_t token) {
        if (token >= 0 && !contains_token(result, token)) result.push_back(token);
    };
    add(config.eos_token_id);
    add(im_end);
    add(slash_s);
    for (size_t token = 0; token < config.vocabulary.size(); ++token) {
        const auto id = static_cast<int32_t>(token);
        if (tokenizer.is_special(id)) add(id);
    }
    return result;
}

void print_prompt_tokens(const std::vector<int32_t> & sequence) {
    std::cerr << "[chat] prompt_tokens=" << sequence.size() << " ids=";
    for (int32_t token : sequence) std::cerr << token << ' ';
    std::cerr << '\n';
}

double tokens_per_second(size_t tokens, double milliseconds) {
    if (tokens == 0 || milliseconds <= 0.0) return 0.0;
    return static_cast<double>(tokens) * 1000.0 / milliseconds;
}

double bytes_to_mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_generation_stats(const GenerationStats & stats) {
    std::cerr << std::fixed << std::setprecision(3)
              << "[chat] generation_stats mode="
              << (stats.used_gpu ? "gpu-kv" : "cpu-kv")
              << " prompt_tokens=" << stats.prompt_tokens
              << " generated_tokens=" << stats.generated_tokens
              << " decode_steps=" << stats.decode_steps
              << " prefill_ms=" << stats.prefill_ms
              << " prefill_tokens_per_sec="
              << tokens_per_second(stats.prompt_tokens, stats.prefill_ms)
              << " decode_ms=" << stats.decode_ms
              << " decode_tokens_per_sec="
              << tokens_per_second(stats.decode_steps, stats.decode_ms)
              << " total_ms=" << stats.total_ms
              << " weight_bytes=" << stats.memory.weight_bytes
              << " weight_mib=" << bytes_to_mib(stats.memory.weight_bytes)
              << " kv_cache_bytes=" << stats.memory.kv_cache_bytes
              << " kv_cache_mib=" << bytes_to_mib(stats.memory.kv_cache_bytes)
              << " recurrent_state_bytes="
              << stats.memory.recurrent_state_bytes
              << " recurrent_state_mib="
              << bytes_to_mib(stats.memory.recurrent_state_bytes)
              << " intermediate_bytes=" << stats.memory.intermediate_bytes
              << " intermediate_mib="
              << bytes_to_mib(stats.memory.intermediate_bytes)
              << " intermediate_kind="
              << (stats.memory.intermediate_is_estimate ? "estimated" : "arena")
              << " peak_memory_bytes=" << stats.memory.total_bytes()
              << " peak_memory_mib=" << bytes_to_mib(stats.memory.total_bytes())
              << " stopped=" << (stats.stopped ? "true" : "false")
              << '\n';
}

template <typename Backend>
void run_turn(const Options & options,
              Backend & backend,
              const QwenTokenizer & tokenizer,
              const std::vector<int32_t> & stop_token_ids,
              const std::string & input) {
    const std::string prompt = options.raw_prompt
        ? input
        : chat_prompt(options.system_prompt, input);
    const std::vector<int32_t> sequence = tokenizer.encode(prompt);
    if (sequence.empty()) {
        std::cerr << "[chat] tokenizer produced no tokens\n";
        return;
    }
    if (backend.max_sequence() == 0 ||
        sequence.size() >= backend.max_sequence()) {
        std::cerr << "[chat] prompt exceeds model context length\n";
        return;
    }

    print_prompt_tokens(sequence);
    std::cerr << "[assistant] " << std::flush;
    const TokenSink token_sink = [&tokenizer](int32_t token) {
        std::cout << tokenizer.decode_piece(token) << std::flush;
    };
    const GenerationResult result = run(
        backend, sequence, static_cast<size_t>(options.max_new_tokens),
        stop_token_ids, token_sink);
    std::cout << '\n';
    std::cout.flush();
    print_generation_stats(result.stats);
}

template <typename Backend>
GenerationResult generate(Backend & backend,
                          std::vector<int32_t> initial_sequence,
                          size_t max_new_tokens,
                          const std::vector<int32_t> & stop_token_ids,
                          TokenSink token_sink) {
    validate_generation_request(backend, initial_sequence, max_new_tokens);

    GenerationResult result;
    result.stats.prompt_tokens = initial_sequence.size();
    result.stats.used_gpu = backend.uses_gpu();
    const size_t limit = generation_limit(
        backend, initial_sequence, max_new_tokens);
    const Clock::time_point total_begin = Clock::now();

    // The backend retains its weights and owns the cache across turns.
    // reset() makes that cache logically empty before this prompt.
    backend.reset();

    int32_t next = 0;
    {
        const Clock::time_point prefill_begin = Clock::now();
        next = backend.prefill(initial_sequence);
        result.stats.prefill_ms = elapsed_ms(prefill_begin, Clock::now());
    }

    for (size_t step = 0; step < limit; ++step) {
        if (is_stop_token(next, stop_token_ids)) {
            result.stats.stopped = true;
            break;
        }
        result.generated_tokens.push_back(next);
        ++result.stats.generated_tokens;
        if (token_sink) token_sink(next);
        initial_sequence.push_back(next);
        if (step + 1 >= limit) break;

        const Clock::time_point decode_begin = Clock::now();
        next = backend.decode(next);
        result.stats.decode_ms += elapsed_ms(decode_begin, Clock::now());
        ++result.stats.decode_steps;
    }

    result.stats.total_ms = elapsed_ms(total_begin, Clock::now());
    result.stats.memory = backend.memory_stats();
    return result;
}

std::string backend_description(const llm::CPULLM &) {
    return "CPU";
}

std::string backend_description(const llm::MetalLLM & backend) {
    return "Metal device=" + backend.device_name();
}

std::string backend_description(const llm::MoeLLM & backend) {
    return "Metal MoE device=" + backend.device_name() +
        " expert_cache_per_layer=" +
        std::to_string(backend.expert_cache_count());
}

template <typename Backend>
constexpr bool is_metal_backend =
    std::is_same_v<Backend, llm::MetalLLM> ||
    std::is_same_v<Backend, llm::MoeLLM>;

template <typename Backend>
int run_interactive(const Options & options, Backend & backend) {
    if (options.profiling_enabled) {
        if constexpr (is_metal_backend<Backend>) {
            backend.enable_profiling(
                options.profile_csv_path, options.metal_kernel_profile);
        } else {
            backend.enable_profiling(options.profile_csv_path);
        }
        std::cerr << "[chat] profile_csv=" << options.profile_csv_path << "\n";
        if constexpr (is_metal_backend<Backend>) {
            const std::filesystem::path primary(options.profile_csv_path);
            const std::filesystem::path parent = primary.parent_path();
            const std::string stem = primary.stem().empty()
                ? "llm_profile" : primary.stem().string();
            std::cerr << "[chat] metal_command_csv="
                      << (parent / (stem + "_metal_commands.csv")).string()
                      << "\n[chat] metal_kernel_csv="
                      << (parent / (stem + "_metal_kernels.csv")).string()
                      << "\n[chat] metal_operation_csv="
                      << (parent / (stem + "_metal_ops.csv")).string()
                      << "\n";
        }
    }

    const auto & config = backend.config();
    if (config.vocabulary.empty() || config.merges.empty()) {
        throw std::runtime_error(
            "model does not contain Qwen tokenizer vocabulary/merges");
    }
    QwenTokenizer tokenizer(
        config.vocabulary, config.merges, config.tokenizer_pre);

    const int32_t im_end = tokenizer.id("<|im_end|>");
    if (im_end < 0) {
        throw std::runtime_error(
            "model vocabulary has no <|im_end|> token");
    }
    const int32_t slash_s = tokenizer.id("</s>");

    std::cerr << "[chat] layers=" << config.layer_count
              << " embedding=" << config.embedding_size
              << " vocab=" << config.vocabulary_size
              << " merges=" << config.merges.size()
              << " context=" << config.context_length
              << " capacity=" << backend.max_sequence()
              << " backend=" << backend_description(backend)
              << "\n";
    std::cout << "Enter text (/exit to quit, /clear is accepted):\n";

    const std::vector<int32_t> stop_token_ids = make_stop_tokens(
        tokenizer, backend, im_end, slash_s);

    std::string input;
    while (std::cout << "> " && std::getline(std::cin, input)) {
        if (input == "/exit" || input == "/quit") break;
        if (input == "/clear") {
            backend.reset();
            std::cout << "[conversation cleared]\n";
            continue;
        }
        run_turn(options, backend, tokenizer, stop_token_ids, input);
    }
    std::cout << "\n";
    return 0;
}

} // namespace

GenerationResult run(llm::CPULLM & backend,
                     std::vector<int32_t> initial_sequence,
                     size_t max_new_tokens,
                     const std::vector<int32_t> & stop_token_ids,
                     TokenSink token_sink) {
    return generate(
        backend, std::move(initial_sequence), max_new_tokens,
        stop_token_ids, std::move(token_sink));
}

GenerationResult run(llm::MetalLLM & backend,
                     std::vector<int32_t> initial_sequence,
                     size_t max_new_tokens,
                     const std::vector<int32_t> & stop_token_ids,
                     TokenSink token_sink) {
    return generate(
        backend, std::move(initial_sequence), max_new_tokens,
        stop_token_ids, std::move(token_sink));
}

GenerationResult run(llm::MoeLLM & backend,
                     std::vector<int32_t> initial_sequence,
                     size_t max_new_tokens,
                     const std::vector<int32_t> & stop_token_ids,
                     TokenSink token_sink) {
    return generate(
        backend, std::move(initial_sequence), max_new_tokens,
        stop_token_ids, std::move(token_sink));
}

int run_cli(int argc, char ** argv) {
    try {
        Options options;
        if (!parse_options(argc, argv, options)) return 1;
        if (options.help_requested) return 0;

        if (options.metal_kernel_profile && !options.use_gpu) {
            throw std::invalid_argument(
                "--metal-kernel-profile requires --gpu");
        }
        if (options.metal_kernel_profile && !options.profiling_enabled) {
            throw std::invalid_argument(
                "--metal-kernel-profile cannot be combined with --no-profile");
        }

        if (options.profiling_enabled) {
            options.profile_csv_path = prepare_profile_output_path(
                options.profile_csv_path);
        }

        std::cerr << "[chat] loading model: " << options.model_path << "\n";
        if (options.use_gpu) {
            std::string architecture;
            {
                llm::MetalRawModel probe(options.model_path);
                architecture = probe.config().architecture;
            }
            if (architecture == "qwen35moe") {
                llm::MoeLLM backend(
                    options.model_path, options.max_sequence,
                    {}, options.expert_cache_count);
                return run_interactive(options, backend);
            }
            llm::MetalLLM backend(options.model_path, options.max_sequence);
            return run_interactive(options, backend);
        }
        llm::CPULLM backend(options.model_path, options.max_sequence);
        return run_interactive(options, backend);
    } catch (const std::exception & error) {
        std::cerr << "chat: " << error.what() << "\n";
        return 1;
    }
}

} // namespace chat

int main(int argc, char ** argv) {
    return chat::run_cli(argc, argv);
}

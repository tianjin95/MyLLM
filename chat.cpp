#include "chat.h"

#include "runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

bool is_letter(uint32_t value) {
    if (is_ascii_letter(value)) {
        return true;
    }
    if (is_number(value)) {
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

std::vector<std::pair<size_t, size_t>> pretokenize(const std::string & text) {
    const std::vector<Codepoint> points = decode_utf8(text);
    std::vector<std::pair<size_t, size_t>> pieces;
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

        // [^\r\n\p{L}\p{N}]?\p{L}+: the optional prefix is consumed
        // together with the following letter run.
        if ((!is_line_break(points[position].value) &&
             !is_number(points[position].value)) &&
            (is_letter(points[position].value) ||
             (position + 1 < points.size() && is_letter(points[position + 1].value)))) {
            ++position;
            while (position < points.size() && is_letter(points[position].value)) {
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
            !is_letter(points[punctuation].value) &&
            !is_number(points[punctuation].value)) {
            position = punctuation;
            while (position < points.size() &&
                   !is_space(points[position].value) &&
                   !is_letter(points[position].value) &&
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
                  const std::vector<std::string> & merges)
        : vocabulary_(vocabulary) {
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
        for (const auto & range : pretokenize(text)) {
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
};

struct Options {
    std::string model_path;
    int max_new_tokens = 32;
    bool raw_prompt = false;
    bool help_requested = false;
    std::string system_prompt = "You are a concise and helpful assistant.";
};

void usage(const char * program) {
    std::cerr << "Usage: " << program << " --model MODEL.gguf [options]\n"
              << "  --tokens N       Maximum generated tokens per input, default 32\n"
              << "  --system TEXT    System message used by ChatML\n"
              << "  --raw            Send input directly without ChatML wrapping\n"
              << "  --help           Show this help\n";
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
        } else if (argument == "--system") {
            options.system_prompt = value("--system");
        } else if (argument == "--raw") {
            options.raw_prompt = true;
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

} // namespace

int run(int argc, char ** argv) {
    try {
        Options options;
        if (!parse_options(argc, argv, options)) return 1;
        if (options.help_requested) return 0;

        std::cerr << "[chat] loading model: " << options.model_path << "\n";
        llm::llm_runtime runtime(options.model_path);
        if (runtime.token_vocabulary.empty() || runtime.token_merges.empty()) {
            throw std::runtime_error("model does not contain Qwen tokenizer vocabulary/merges");
        }
        QwenTokenizer tokenizer(runtime.token_vocabulary, runtime.token_merges);

        const int32_t im_end = tokenizer.id("<|im_end|>");
        if (im_end < 0) throw std::runtime_error("model vocabulary has no <|im_end|> token");
        const int32_t slash_s = tokenizer.id("</s>");

        std::cerr << "[chat] layers=" << runtime.layer_count
                  << " embedding=" << runtime.embedding_size
                  << " vocab=" << runtime.vocabulary_size
                  << " merges=" << runtime.token_merges.size()
                  << " context=" << runtime.context_length
                  << " (no KV cache)\n";
        std::cout << "Enter text (/exit to quit, /clear is accepted):\n";

        std::string input;
        while (std::cout << "> " && std::getline(std::cin, input)) {
            if (input == "/exit" || input == "/quit") break;
            if (input == "/clear") {
                std::cout << "[conversation cleared]\n";
                continue;
            }

            const std::string prompt = options.raw_prompt
                ? input
                : chat_prompt(options.system_prompt, input);
            std::vector<int32_t> sequence = tokenizer.encode(prompt);
            if (sequence.empty()) {
                std::cerr << "[chat] tokenizer produced no tokens\n";
                continue;
            }
            if (sequence.size() >= runtime.context_length) {
                std::cerr << "[chat] prompt exceeds model context length\n";
                continue;
            }

            const size_t available = runtime.context_length - sequence.size();
            const size_t generation_limit = std::min(
                available, static_cast<size_t>(options.max_new_tokens));
            std::cerr << "[chat] prompt_tokens=" << sequence.size() << " ids=";
            for (int32_t token : sequence) std::cerr << token << ' ';
            std::cerr << "\n[assistant] " << std::flush;

            for (size_t step = 0; step < generation_limit; ++step) {
                const int32_t next = runtime.forward(sequence);
                if (next == runtime.eos_token_id || next == im_end || next == slash_s ||
                    tokenizer.is_special(next)) {
                    break;
                }
                std::cout << tokenizer.decode_piece(next) << std::flush;
                sequence.push_back(next);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "chat: " << error.what() << "\n";
        return 1;
    }
}

} // namespace chat

int main(int argc, char ** argv) {
    return chat::run(argc, argv);
}

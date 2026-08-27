#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace llm {
class CPULLM;
class MetalLLM;
}

namespace chat {

using TokenSink = std::function<void(std::int32_t)>;

struct GenerationStats {
    // Time spent producing the first next-token logits from the initial
    // sequence: embedding + all layer prefill work + final LM head.
    double prefill_ms = 0.0;

    // Time spent producing logits for subsequent tokens with one-token decode
    // and the per-layer cache.
    double decode_ms = 0.0;

    double total_ms = 0.0;
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t decode_steps = 0;
    bool used_gpu = false;
    bool stopped = false;
};

struct GenerationResult {
    std::vector<std::int32_t> generated_tokens;
    GenerationStats stats;
};

// Generate greedily from an initial token sequence. The selected backend owns
// its weights and K/V cache; run() resets only the logical conversation state
// before prefill while keeping the loaded backend alive for later turns.
// initial_sequence is copied and extended internally; the caller's sequence
// is never modified. stop_token_ids may be empty when no early-stop tokens
// are desired. token_sink, when supplied, is called immediately for each
// emitted token and is useful for streaming output.
GenerationResult run(llm::CPULLM& backend,
                     std::vector<std::int32_t> initial_sequence,
                     std::size_t max_new_tokens,
                     const std::vector<std::int32_t>& stop_token_ids = {},
                     TokenSink token_sink = {});

GenerationResult run(llm::MetalLLM& backend,
                     std::vector<std::int32_t> initial_sequence,
                     std::size_t max_new_tokens,
                     const std::vector<std::int32_t>& stop_token_ids = {},
                     TokenSink token_sink = {});

// Run the interactive native C++ chat entry point.
int run_cli(int argc, char ** argv);

} // namespace chat

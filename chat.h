#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace llm {
class llm_runtime;
}

namespace chat {

using TokenSink = std::function<void(std::int32_t)>;

struct GenerationStats {
    // Time spent producing the first next-token logits from the initial
    // sequence. For run() this is a complete no-KV forward pass; for run_kv()
    // this is embedding + all layer prefill work + final LM head.
    double prefill_ms = 0.0;

    // Time spent producing logits after the first generated token. For run()
    // each step recomputes the complete prefix; for run_kv() each step uses
    // one-token decode with the per-layer cache.
    double decode_ms = 0.0;

    double total_ms = 0.0;
    std::size_t prompt_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t decode_steps = 0;
    bool stopped = false;
};

struct GenerationResult {
    std::vector<std::int32_t> generated_tokens;
    GenerationStats stats;
};

// Generate greedily from an initial token sequence without a KV cache.
// initial_sequence is copied and extended internally; the caller's sequence
// is never modified. stop_token_ids may be empty when no early-stop tokens
// are desired. token_sink, when supplied, is called immediately for each
// emitted token and is useful for streaming output.
GenerationResult run(llm::llm_runtime& runtime,
                     std::vector<std::int32_t> initial_sequence,
                     std::size_t max_new_tokens,
                     const std::vector<std::int32_t>& stop_token_ids = {},
                     TokenSink token_sink = {});

// Generate greedily using one freshly-created K/V cache per invocation. The
// cache is populated by prefill and then consumed by one-token decode steps;
// it is destroyed when this function returns. token_sink has the same
// streaming semantics as run().
GenerationResult run_kv(llm::llm_runtime& runtime,
                        std::vector<std::int32_t> initial_sequence,
                        std::size_t max_new_tokens,
                        const std::vector<std::int32_t>& stop_token_ids = {},
                        TokenSink token_sink = {});

// Run the interactive native C++ chat entry point.
int run_cli(int argc, char ** argv);

} // namespace chat

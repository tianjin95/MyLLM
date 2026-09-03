# Qwen3.6-35B-A3B Q4_K_M: computation and MyLLM compatibility

Analysis date: 2026-09-03

This document describes the text-generation path in the downloaded GGUF. It
uses the model's real GGUF metadata and tensor table, and cross-checks the graph
against llama.cpp's `qwen35moe.cpp` and `delta-net-base.cpp` implementation.

## 1. Download and integrity

- Repository: `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`
- File: `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
- Local path:
  `/Users/zhaotianjin/Library/Mobile Documents/com~apple~CloudDocs/WorkSpace/KVCache/models/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
- File size: `22,285,080,192` bytes = `20.754598 GiB`
- SHA-256:
  `b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`
- Remote LFS size and SHA-256 match the local file.
- GGUF version: 3
- GGUF header/data offset: `10,990,720` bytes
- Tensor count: 753
- Parameter count: `35,505,251,456`
- Tensor payload: `22,274,089,472` bytes = `20.744362 GiB`
- The final tensor ends exactly at the local file's EOF.

The base GGUF contains the language model. Image input additionally needs a
separate `mmproj` file and a vision preprocessing path; neither is part of this
MyLLM compatibility analysis.

## 2. Notation and tensor orientation

- `N`: number of prompt tokens in prefill.
- `L`: valid context length in one decode step, including cached tokens.
- `H = 2048`: hidden width.
- All activation shapes below use the conventional row-major view
  `[tokens, channels]`.
- GGUF prints a linear tensor as `[input, output]`. MyLLM's logical matrix view
  is `[output, input]`. For example, GGUF `2048 x 8192` is a logical
  `W[8192, 2048]` and computes `X[N, 2048] * W^T -> Y[N, 8192]`.
- Unless stated otherwise, current MyLLM uses F32 activations, dequantizes each
  weight block inside the Metal GEMM/GEVM kernel, accumulates in F32, and emits
  F32.

## 3. Model constants

| Property | Value |
| --- | ---: |
| GGUF architecture | `qwen35moe` |
| Main decoder layers | 40 |
| Extra MTP/NextN layers | 1 (`blk.40`) |
| Hidden width | 2048 |
| Vocabulary | 248,320 |
| Maximum context | 262,144 |
| Full-attention interval | 4 |
| Full-attention layers | 3, 7, 11, ..., 39 (10 layers) |
| Gated DeltaNet layers | remaining 30 layers |
| Query heads in full attention | 16 |
| KV heads in full attention | 2 |
| Full-attention head width | 256 |
| Rotary width per head | 64 |
| RoPE theta | 10,000,000 |
| Interleaved MRoPE sections | 11, 11, 10 |
| DeltaNet key heads | 16 |
| DeltaNet value heads | 32 |
| DeltaNet key/value head width | 128 |
| DeltaNet inner width | 4096 |
| Causal depthwise-convolution kernel | 4 |
| Routed experts | 256 |
| Selected experts per token | 8 |
| Routed expert intermediate width | 512 |
| Shared expert intermediate width | 512 |
| RMSNorm epsilon | 1e-6 |
| Tokenizer pre-type | `qwen35` |

The GGUF metadata reports `block_count=41`, but normal autoregressive inference
executes only 40 main layers. `blk.40` is the optional one-layer MTP draft path,
not another main decoder layer.

## 4. Quantization inventory

`Q4_K_M` does not mean every tensor is INT4. It is a mixed quantization recipe.

| GGML type | Tensor count | Stored bytes | Payload share |
| --- | ---: | ---: | ---: |
| Q4_K | 153 | 15,659,237,376 | 70.30% |
| Q6_K | 115 | 5,340,733,440 | 23.98% |
| Q8_0 | 105 | 1,110,769,664 | 4.99% |
| F32 | 368 | 104,624,640 | 0.47% |
| Q5_K | 10 | 57,671,680 | 0.26% |
| BF16 | 2 | 1,052,672 | 0.005% |

The two BF16 tensors are MTP router tensors in `blk.40`. They are unnecessary
for ordinary one-token autoregressive generation if MTP is deliberately
skipped.

Important type assignments in the 40 main layers:

- Token embedding: Q4_K.
- LM head: Q6_K.
- DeltaNet QKV: Q6_K; DeltaNet `z` gate: Q4_K.
- DeltaNet output: Q4_K in 16 layers and Q8_0 in 14 layers.
- Full-attention joint Q+gate: Q4_K in 6 layers and Q6_K in 4 layers.
- Full-attention K and V: Q8_0.
- Full-attention output: Q5_K in all 10 full-attention layers.
- Routed expert gate/up: Q4_K.
- Routed expert down: Q4_K in 20 layers and Q6_K in 20 layers.
- Shared expert gate/up/down: Q6_K in 20 layers and Q8_0 in 20 layers.
- Norms, routers, DeltaNet alpha/beta/conv/state parameters: F32.

## 5. Top-level text graph

For token IDs `int32[N]`:

1. Q4_K embedding gather: `int32[N] -> F32[N, 2048]`.
2. Repeat 40 main layers:
   - RMSNorm: `F32[N, 2048] -> F32[N, 2048]`.
   - Run either Gated DeltaNet or full gated attention.
   - Add the attention/DeltaNet residual: `[N, 2048]`.
   - Post-attention RMSNorm: `[N, 2048]`.
   - Run top-8 routed MoE plus the shared expert: `[N, 2048]`.
   - Add the FFN residual: `[N, 2048]`.
3. Final RMSNorm: `[N, 2048]`.
4. Select the final token row for generation: `[1, 2048]`.
5. Q6_K LM head: `[1, 2048] -> F32[1, 248320]`.
6. Argmax: `F32[248320] -> int32 token_id`.

There are no ordinary Q/K/V projection biases in this architecture. The
DeltaNet time-step branch does have an F32 `ssm_dt.bias[32]`.

## 6. Gated DeltaNet layer (30 layers)

The following starts from normalized hidden state `X[N, 2048]`.

| Step | Weight / operation | Input | Output | Stored type -> activation type |
| --- | --- | --- | --- | --- |
| Joint QKV projection | `Wqkv[8192,2048]` | `[N,2048]` | `[N,8192]` | Q6_K x F32 -> F32 |
| Output-gate projection | `Wz[4096,2048]` | `[N,2048]` | `z[N,4096]` | Q4_K x F32 -> F32 |
| Beta projection | `Wbeta[32,2048]` | `[N,2048]` | `[N,32]` | F32 x F32 -> F32 |
| Beta activation | sigmoid | `[N,32]` | `beta[N,32]` | F32 -> F32 |
| Alpha projection | `Walpha[32,2048]` | `[N,2048]` | `[N,32]` | F32 x F32 -> F32 |
| Decay | `softplus(alpha + dt_bias) * A` | `[N,32]` | `g[N,32]` | F32 -> F32 |
| Causal depthwise Conv1D | kernel `[8192,4]` | `[N,8192]` plus 3-row state | `[N,8192]` | F32 -> F32 |
| Conv activation | SiLU | `[N,8192]` | `[N,8192]` | F32 -> F32 |
| Split | views | `[N,8192]` | Q/K/V below | F32 views |
| Q split | 16 heads x 128 | | `[N,16,128]` | F32 |
| K split | 16 heads x 128 | | `[N,16,128]` | F32 |
| V split | 32 heads x 128 | | `[N,32,128]` | F32 |
| Q/K normalization | L2 norm per 128-vector | Q and K | same shapes | F32 -> F32 |
| Delta recurrence | state update and read | Q/K/V/g/beta | `[N,32,128]` | F32 -> F32 |
| Gated norm | per-head RMSNorm, then `* SiLU(z)` | `[N,32,128]`, z | `[N,4096]` | F32 -> F32 |
| Output projection | `Wo[2048,4096]` | `[N,4096]` | `[N,2048]` | Q4_K/Q8_0 x F32 -> F32 |

Q and K have 16 heads while V/state have 32 heads. Semantically, each Q/K head
is shared by two value heads using an interleaved repeat. An unfused
implementation can repeat Q and K from `[N,16,128]` to `[N,32,128]`; a fused
kernel should map value head `h` to key head `h % 16` without materializing the
repeat.

### Decode recurrence

Each DeltaNet layer persistently owns `S[32,128,128]`. For one value head, the
one-token recurrence is conceptually:

```text
S = exp(g) * S
prediction = S^T * k
delta = beta * (v - prediction)
S = S + outer(k, delta)
o = S^T * q
```

The exact transpose convention depends on storage layout, but every vector is
length 128 and every per-head state is `128 x 128`.

### Prefill recurrence

Prefill must produce the same final recurrent state after all `N` prompt
tokens. A simple correctness implementation may scan tokens serially. llama.cpp
uses a 64-token chunked DeltaNet graph to expose parallel work; implementing
that optimized graph needs chunk-local decay masks, triangular solves, several
small batched matrix products, and a state carry between chunks.

Persistent F32 state for all 30 DeltaNet layers:

- Matrix state: `30 * 32 * 128 * 128 * 4 = 60 MiB`.
- Conv history: `30 * 3 * 8192 * 4 = 2.8125 MiB`.
- Total: `62.8125 MiB`, independent of context length.

## 7. Full gated-attention layer (10 layers)

The following starts from normalized hidden state `X[N, 2048]`.

| Step | Weight / operation | Input | Output | Stored type -> activation type |
| --- | --- | --- | --- | --- |
| Joint Q+gate projection | `Wqg[8192,2048]` | `[N,2048]` | `[N,8192]` | Q4_K/Q6_K x F32 -> F32 |
| Q view | 16 heads x 256 | packed Q+gate | `Q[N,16,256]` | F32 view |
| Gate view | 16 heads x 256 | packed Q+gate | `gate[N,4096]` | F32 view/copy |
| K projection | `Wk[512,2048]` | `[N,2048]` | `K[N,2,256]` | Q8_0 x F32 -> F32 |
| V projection | `Wv[512,2048]` | `[N,2048]` | `V[N,2,256]` | Q8_0 x F32 -> F32 |
| Q RMSNorm | learned gamma `[256]` per head | Q | `[N,16,256]` | F32 -> F32 |
| K RMSNorm | learned gamma `[256]` per head | K | `[N,2,256]` | F32 -> F32 |
| Partial iMRoPE | rotate first 64 dims/head | Q, K | same shapes | F32 -> F32 |
| QK | 16 query heads, GQA ratio 8 | Q and cached K | scores below | F32 x F32 -> F32 |
| Mask | causal | scores | same shape | F32 |
| Softmax | over key positions | scores | probabilities | F32 -> F32 |
| AV | probabilities and cached V | | `[N,16,256]` | F32 x F32 -> F32 |
| Flatten | view | `[N,16,256]` | `[N,4096]` | F32 view |
| Output gate | `attention * sigmoid(gate)` | two `[N,4096]` tensors | `[N,4096]` | F32 -> F32 |
| Output projection | `Wo[2048,4096]` | `[N,4096]` | `[N,2048]` | Q5_K x F32 -> F32 |

The joint projection layout is interleaved per head:
`[Q_head0(256), gate_head0(256), Q_head1(256), gate_head1(256), ...]`.
It is not a contiguous first-half Q plus second-half gate split.

For prefill, stacked scores are logically `[16,N,N]`. MyLLM's present head loop
can reuse one `[N,N]` score slot, so the minimum score arena is `4*N*N` bytes,
not `16*4*N*N` bytes. For decode, one head's score vector is `[1,L]` and uses
`4*L` bytes.

Each KV head is shared by eight Q heads: query head `qh` reads KV head
`floor(qh/8)`.

For text-only prompts, all MRoPE position channels are equal. In that case the
existing partial NeoX-style RoPE arithmetic can be reused with `head_dim=256`
and `rotary_dim=64`. Image/video input needs true four-channel, interleaved
MRoPE positions and is not supported by the current kernel.

## 8. MoE FFN in every main layer

Input is the post-attention normalized `X[N,2048]`.

### Router

1. F32 router: `[N,2048] * Wrouter[256,2048]^T -> logits[N,256]`.
2. Softmax over 256 experts: `[N,256]`.
3. Select top 8 expert IDs: `int32[N,8]` and weights `F32[N,8]`.
4. Renormalize the eight selected probabilities so each token's selected
   weights sum to one.

### Eight routed experts per token

For every selected expert `e`:

```text
gate_e = X[N,2048] * Wgate_e[512,2048]^T -> F32[N,512]  (Q4_K weight)
up_e   = X[N,2048] * Wup_e  [512,2048]^T -> F32[N,512]  (Q4_K weight)
act_e  = SiLU(gate_e) * up_e                     -> F32[N,512]
down_e = act_e * Wdown_e[2048,512]^T             -> F32[N,2048]
```

`Wdown_e` is Q4_K in 20 layers and Q6_K in 20 layers. Real prefill cannot use
one common `N` for every expert because different prompt tokens choose different
experts. It needs token grouping/gather, expert GEMM, and scatter/reduction.

Before weighted reduction, the logical routed output is `[N,8,2048]`.
Multiplying each expert result by `weights[N,8]` and summing over 8 produces
`routed[N,2048]`.

### Shared expert

```text
shared_gate = sigmoid(X[N,2048] * Wscalar[1,2048]^T) -> F32[N,1]
gate        = X * Wshared_gate[512,2048]^T            -> F32[N,512]
up          = X * Wshared_up  [512,2048]^T            -> F32[N,512]
shared      = SiLU(gate) * up                         -> F32[N,512]
shared      = shared * Wshared_down[2048,512]^T       -> F32[N,2048]
shared      = shared * shared_gate                    -> F32[N,2048]
moe_out     = routed + shared                         -> F32[N,2048]
```

The shared expert matrices are Q6_K in 20 layers and Q8_0 in 20 layers. Router
and scalar-gate weights are F32. There are no expert projection biases.

## 9. Decode compute and traffic estimate

Ignoring small elementwise/transcendental costs and counting one multiply-add
as two FLOPs, a one-token main pass is approximately:

- 30 DeltaNet layers, including a direct recurrent update: `2.13 GFLOP`.
- 10 full-attention projection paths, excluding QK/AV over history:
  `0.545 GFLOP`.
- 40 top-8 MoE plus shared-expert paths: `2.31 GFLOP`.
- LM head: `1.017 GFLOP`.
- Full-attention history term across all 10 layers: `163,840 * L` FLOPs.

Therefore:

```text
decode_flops(L) ~= 6.003e9 + 163840 * L
```

| Context `L` | Approximate decode FLOPs/token |
| ---: | ---: |
| 1,024 | 6.17 GFLOP |
| 4,096 | 6.67 GFLOP |
| 8,192 | 7.34 GFLOP |
| 32,768 | 11.37 GFLOP |
| 262,144 | 48.95 GFLOP |

Only 8/256 routed experts are active per token. Including all attention weights,
routers, selected expert slices, shared experts, and the LM head gives
`2,946,429,568` active parameters and about `2,196,040,192` bytes
(`2.045 GiB`) of native quantized/F32 weight reads per decode token in an ideal
implementation. This explains the A3B name and also means short-context decode
will normally be memory-bandwidth dominated.

## 10. Prefill compute and representative activation sizes

A rough, non-fused prefill estimate is:

```text
prefill_flops(N) ~= 4.9855e9 * N + 1.0171e9 + 163840 * N^2
```

The first term is the 40-layer active path, the second is one final-row LM head,
and the last is full QK+AV work for 10 attention layers. This is an estimate:
the exact DeltaNet chunk algorithm and causal/Flash Attention implementation
change the operation count.

| Prompt `N` | Approx. work | Hidden `[N,2048]` | Wide `[N,8192]` | One-head score `[N,N]` | Routed output `[N,8,2048]` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.642 TFLOP | 1 MiB | 4 MiB | 0.0625 MiB | 8 MiB |
| 512 | 2.597 TFLOP | 4 MiB | 16 MiB | 1 MiB | 32 MiB |
| 1,024 | 5.278 TFLOP | 8 MiB | 32 MiB | 4 MiB | 64 MiB |
| 4,096 | 23.170 TFLOP | 32 MiB | 128 MiB | 64 MiB | 256 MiB |

These are individual live-tensor sizes, not a final arena total. A new arena
plan must account for reuse, packed Q+gate views, MoE gather/scatter, and the
chosen DeltaNet prefill algorithm.

## 11. Persistent memory and expert cache

| Component | Bytes | GiB |
| --- | ---: | ---: |
| All 753 GGUF tensor payloads, including MTP | 22,274,089,472 | 20.744 |
| Main graph weights, excluding `blk.40` MTP | 21,376,133,632 | 19.908 |
| Optional MTP block alone | 897,955,840 | 0.836 |
| Routed expert gate/up/down tensors | 19,503,513,600 | 18.164 |
| All non-routed main-graph weights | 1,872,620,032 | 1.743 |
| DeltaNet recurrent + conv state | 65,863,680 | 0.061 |

MyLLM no longer uploads all routed experts. Each layer has an independent LRU
with a runtime-selected number of slots. One slot stores that layer's gate, up,
and down slice. Across all 40 layers, one slot per layer costs 76,185,600 bytes
(72.65625 MiB): 20 Q4_K-down layers use 1,769,472 bytes per slot and 20 Q6_K-down
layers use 2,039,808 bytes per slot. Therefore persistent Metal weight buffers
are:

```text
weight_bytes(N) = 1,872,620,032 + 76,185,600 * N
```

| Per-layer slots `N` | Cached routed bytes | Total resident weight buffers |
| ---: | ---: | ---: |
| 8 | 581.250 MiB | 2,482,104,832 bytes (2.312 GiB) |
| 16 | 1.135 GiB | 3,091,589,632 bytes (2.879 GiB) |
| 32 | 2.271 GiB | 4,310,559,232 bytes (4.015 GiB) |
| 64 | 4.541 GiB | 6,748,498,432 bytes (6.285 GiB) |
| 256 | 18.164 GiB | 21,376,133,632 bytes (19.908 GiB) |

The GGUF is held through a read-only `mmap` (with `pread` fallback). This maps
the full file into virtual address space but does not make its 20.75 GiB payload
part of the persistent Metal working set. Touched pages can remain in the OS
file cache and are reclaimable under memory pressure.

Only the 10 full-attention layers need a conventional KV cache. With current
MyLLM F32 K/V storage, each token requires:

```text
10 layers * 2 (K and V) * 2 KV heads * 256 dims * 4 bytes
= 40,960 bytes/token
```

| Capacity | F32 full-attention KV | 8-slot weights + Delta state + KV |
| ---: | ---: | ---: |
| 1,024 | 40 MiB | 2.412 GiB |
| 4,096 | 160 MiB | 2.529 GiB |
| 8,192 | 320 MiB | 2.686 GiB |
| 16,384 | 640 MiB | 2.998 GiB |
| 32,768 | 1.25 GiB | 3.623 GiB |
| 262,144 | 10 GiB | 12.373 GiB |

The final column excludes the activation arena, tokenizer objects, Metal
runtime allocations, command buffers, program memory, reclaimable file cache,
and macOS. Expert caching removes the previous immediate 20 GiB residency
constraint. Very long contexts are still bounded by KV storage, OS pressure,
and the quadratic full-attention work rather than expert-buffer capacity alone.

## 12. Implemented MyLLM kernel compatibility

### Directly reusable primitives

| Kernel/path | Model use | Status |
| --- | --- | --- |
| Quantized embedding gather | token embedding `[N,2048]` | Implemented |
| F32/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K GEMM/GEVM | ordinary projections | Implemented |
| F32 RMSNorm/L2Norm | hidden and per-head norms | Implemented |
| F32 softmax, top-k, renormalization | attention and 256-way router | Implemented |
| F32 SwiGLU/residual/channel operations | expert and DeltaNet elementwise work | Implemented |
| KV write/QK/AV | 10 full-attention layers | Implemented with per-layer FP32 cache |
| Q+gate split and sigmoid gate | full attention | Implemented |
| Causal Conv1D and history commit | 30 DeltaNet layers | Implemented |
| Stateful Gated DeltaNet scan | prefill and decode | Implemented; serial in token order |
| Indexed Q4_K/Q6_K expert products | top-8 routed experts | Implemented |
| Q6_K LM head + argmax | final token | Implemented |

The 3-D expert tensors are treated as contiguous expert slices. The router first
writes original expert IDs. CPU-side cache management maps each selected expert
to a layer-local slot and copies missing gate/up/down slices from the GGUF. The
IDs are then rewritten to slot IDs, so the indexed expert kernels use the same
byte-offset calculation over a compact cache buffer. The baseline keeps all
eight routes as independent kernels for clarity; it does not group tokens by
expert or fuse gate/up/down.

### Loader and runtime status

| Area | Current status |
| --- | --- |
| Architecture loader | `MetalRawModel` parses `qwen35moe`; `MoeRawModel` maps its schema |
| GGUF scalar type | BF16 row size is recognized; MTP tensors using it are skipped |
| Layer count | Runs 40 main layers and ignores all `blk.40.*` MTP tensors |
| Head dimensions | Reads explicit 256-wide attention heads and 64 rotary dimensions |
| Biases | Full attention is dispatched without synthetic or optional Q/K/V bias |
| Layer state | KV is allocated only for 10 full-attention layers; recurrent/Conv state only for 30 DeltaNet layers |
| Tensor validation | 733 main tensors are consumed; routed tensors are descriptors and 20 optional MTP tensors are explicitly ignored |
| Expert storage | Routed tensors stay in the mapped GGUF; every layer owns a configurable 8-256-slot LRU cache |
| Arena | Chunk-sized prefill arena and persistent one-row decode arena cover all intermediates |
| Command flow | Every layer has a route command, CPU cache update, and expert command; hidden activations stay in the Arena |
| Multimodal RoPE | Not implemented; the current partial iMRoPE path accepts text positions only |
| Tokenizer | Uses the Qwen3.5 `[\\p{L}\\p{M}]+` rule; common combining-mark ranges are supported, while full Unicode-category coverage remains limited |
| Prefill chunking | Chunk size is `floor(expert_cache_slots / 8)`; KV, convolution, and recurrent state continue across chunks |
| Optimized prefill | DeltaNet still scans each chunk serially; llama.cpp's parallel chunk formulation is not implemented |

The existing `metal_rope_heads_f32` is conditionally reusable for text-only
input: after fixing metadata, its split-half pair layout and frequency formula
match the model's partial 64-dimensional RoPE when all MRoPE position channels
are identical. It is not a complete multimodal implementation.

## 13. Current-run result

The standalone target builds without warnings and runs the complete main graph
on a 24 GB, 10-GPU-core Apple M4 with eight expert slots per layer:

```text
[chat] loading model: .../Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf
[chat] layers=40 embedding=2048 vocab=248320 merges=247587
       context=262144 capacity=64 backend=Metal MoE device=Apple M4
       expert_cache_per_layer=8
```

For raw prompt `Hello` (token ID 9419), MyLLM and llama.cpp both greedily produce
the same first two decoded tokens, `",\n\n"`. This covers prefill and the first
stateful decode step. The tested MyLLM run reports:

```text
weight_bytes=2482104832         (2367.120 MiB)
kv_cache_bytes=2621440          (2.500 MiB at capacity 64)
recurrent_state_bytes=65863680  (62.812 MiB)
intermediate_bytes=1181440      (1.127 MiB)
peak_memory_bytes=2551771392    (2433.559 MiB)
```

The first cold eight-slot run measured about 1.60 seconds for one-token prefill
and 1.21 seconds for the first decode, versus about 74-80 and 125 seconds with
all expert buffers resident and severe Metal paging. File-cache warmth changes
these figures substantially. A two-token raw `Hello world` prefill produced the
same `!` token with both eight slots (two one-token chunks) and sixteen slots
(one two-token chunk), checking state continuity across the chunk boundary.

## 14. Remaining work

1. Compare longer prefill/decode sequences and intermediate tensors against
   llama.cpp, including prompts that cross full-attention and recurrent state.
2. Expand the tokenizer's coarse Unicode category tables and consume the
   model's GGUF chat-template semantics instead of assuming ChatML.
3. Group prompt tokens by selected expert, then dispatch each active expert once
   rather than executing eight route-index passes for every token.
4. Replace serial-token DeltaNet prefill with the chunked formulation while
   retaining the simple recurrence as a correctness reference.
5. Replace one-output-thread quantized products with cooperative SIMD-group
   kernels and parallelize the vocabulary argmax.
6. Add cache hit/miss and GGUF page-in telemetry, then evaluate asynchronous
   expert prefetch and double-buffered uploads.

MTP, speculative decoding, and multimodal input remain explicitly disabled.

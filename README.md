# Standalone MyLLM

MyLLM is a self-contained Qwen2 and text-only Qwen3.5/Qwen3.6 MoE inference
demo. It does not call `llama.cpp`, GGML, or another inference runtime. All
execution paths implement prefill and one-token decode directly:

```text
chat
  |-- default ----------------> CPULLM (qwen2)
  `-- --gpu + GGUF architecture
        |-- qwen2 -----------> MetalLLM
        `-- qwen35moe -------> MoeLLM
```

Each backend owns its model weights, persistent sequence state, and current
position, and exposes `reset()`, `prefill()`, and `decode()`. CPU uses
`ModelFile`; both Metal backends use the independent streaming `MetalRawModel`
reader. There is no separate runtime or session layer and `--gpu` does not
silently fall back to CPU. `qwen35moe` is Metal-only.

## Source layout

```text
chat.cpp          console UI, Qwen2/Qwen3.5 tokenizer, autoregressive loop
chat.h            public generation helpers for all three backends
cpu_llm.h/.cpp    CPU tensors, operators, model ownership, KV cache, inference
model.h/.cpp      CPU GGUF parser and FP32 dequantization
metal_model.h/.cpp independent streaming GGUF parser and raw tensor types
metal_llm.h/.mm   Metal model ownership, buffers, graph encoding, and inference
metal_llm.metal   quantized-weight projection and FP32 activation kernels
moe_model.h/.cpp  qwen35moe tensor mapping; excludes optional MTP tensors
moe_llm.h/.mm     text-only MoE/DeltaNet Metal state and inference graph
QWEN36_35B_A3B_COMPATIBILITY.md Qwen3.6 graph, dimensions, and support analysis
METAL_INFERENCE.md detailed kernel, threading, command, prefill/decode guide
memory_stats.h    common weight/KV/intermediate memory accounting interface
profiler.h/.cpp   CSV timing and logical FLOP/traffic statistics
```

See `METAL_INFERENCE.md` for the complete GPU execution walkthrough, including
per-kernel formulas, thread grids, Qwen2.5 tensor shapes, command-buffer
construction, and the path from Chat input to generated text.

The CPU dequantizer remains private to `model.cpp`. The Metal backend does not
include it: `metal_model.cpp` computes GGML block/row byte sizes and reads one
raw tensor at a time. This directory does not include source files from its
parent directory.

## Build

From `cpp/MyLLM`:

```bash
make
```

Object files are written to `build/` and the executable is `./chat`. The build
requires C++17 and, on macOS, links the Metal and Foundation frameworks. The
Metal source is compiled at program startup, so the command-line `metal` tool
is not required.

The program searches for `metal_llm.metal` in the working directory and beside
the executable. Override it when needed:

```bash
MYLLM_METAL_SHADER=/path/to/metal_llm.metal ./chat --gpu --model model.gguf
```

## Run

CPU is selected by default:

```bash
./chat \
  --model ../../models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokens 4
```

Metal must be selected explicitly:

```bash
./chat \
  --model ../../models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokens 4 \
  --gpu
```

The text-only Qwen3.6-35B-A3B path is selected from the GGUF architecture:

```bash
./chat \
  --model ../../models/qwen3.6-35b-a3b/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --gpu --max-sequence 64 --expert-cache 16 --tokens 1 --raw
```

Routed expert tensors remain in the GGUF file. `--expert-cache N` controls the
number of expert bundles cached independently in every layer and must be in
`[8, 256]` for this model. The default is 8. Larger values reduce cache misses
and permit larger prefill chunks at the cost of about 72.656 MiB for every
additional slot across all 40 layers. When `--max-sequence` is omitted,
`MoeLLM` caps its default capacity at 1024 tokens; Qwen2 backends continue to
default to the model context length.

Options:

```text
--tokens N       Maximum generated tokens per input, default 32
--max-sequence N Fixed sequence-state capacity (backend-specific default)
--expert-cache N Routed-expert cache slots per MoE layer, default 8
--system TEXT    Replace the default ChatML system message
--raw            Tokenize input directly without ChatML wrapping
--gpu            Require the Metal backend; never fall back to CPU
--profile-csv P  Write profiling CSV under output/
--profile-log P  Compatibility alias for --profile-csv
--metal-kernel-profile
                 Record per-kernel Metal GPU timestamps (diagnostic mode)
--no-profile     Disable profiling
--help           Show command-line help
```

Interactive commands are `/exit`, `/quit`, and `/clear`. Every normal input is
an independent turn. Before prefill, Chat calls `backend.reset()`; it does not
destroy or recreate the backend.

The Makefile provides the same options:

```bash
make run TOKENS=4
make run MODEL=/path/to/model.gguf TOKENS=8
make run RAW=1 TOKENS=4
make run GPU=1 TOKENS=4
make run MAX_SEQUENCE=2048 TOKENS=4
make run GPU=1 EXPERT_CACHE=16 TOKENS=4
make run PROFILE_CSV=qwen-profile.csv TOKENS=1
make run GPU=1 METAL_KERNEL_PROFILE=1 TOKENS=1
make run NO_PROFILE=1 TOKENS=1
```

`TOKEN` remains an alias for `TOKENS`.

## Backend ownership

### CPU

`CPULLM` dequantizes all GGUF weights to CPU `Matrix` and `Vector` objects in
its constructor and keeps them for its lifetime. It owns a `CPUKVCache` for
each layer. CPU K/V matrices grow by one row during decode. `reset()` discards
their logical contents and position before the next turn.

### Metal

`MetalLLM` initializes the Metal device and pipelines, parses GGUF metadata with
its own loader, and streams each tensor's native bytes once into a persistent
`MTLBuffer`. Quantized matrix weights remain Q5_0, Q8_0, Q4_K, or Q6_K; F32
norm and bias vectors remain F32. The per-tensor CPU byte staging is released
immediately after upload. Inference never looks up or uploads a weight before
an operator.

For the included Qwen2.5-0.5B Q4_K_M file this changes persistent Metal weight
storage from 2,520,669,696 bytes of dequantized FP32 to 485,452,288 bytes
(462.963 MiB) of native GGUF payload.

It allocates one private K buffer and one private V buffer per layer at model
initialization. Each buffer has shape:

```text
[max_sequence, kv_heads * head_dim]
```

The default capacity is the GGUF context length. For Qwen2.5-0.5B this is about
`24 * 2 * 32768 * 128 * 4` bytes, or 768 MiB of FP32 KV storage. `reset()` sets
the sequence length to zero; old bytes are not cleared because prefill
overwrites the new prefix and attention kernels never read beyond the logical
length.

For `MetalLLM`, one command buffer contains the complete Qwen2 prefill or decode
graph. Kernel output buffers are bound directly as following kernel inputs.
Only token IDs enter from CPU and one four-byte greedy argmax token ID returns
after completion.

The Metal backend uses phase-specific private activation Arenas rather than one
Arena sized to the complete context. For a prompt of `N` tokens, prefill plans
all row-shaped slots for `N` rows and one FP32 `[N, N]` score/probability slot.
That Arena is released after the synchronous prefill command completes. The
first decode step lazily allocates an Arena with one activation row and one
FP32 `[1, max_sequence]` score/probability slot; every later decode step in the
turn reuses it. `reset()` releases the prior turn's decode Arena. Operators
within either phase still use fixed offsets in one Arena and do not allocate an
`MTLBuffer` per operator. All intermediate values remain FP32.

### Metal MoE

`MoeLLM` accepts the `qwen35moe` architecture used by Qwen3.5/Qwen3.6 A3B
models. Ordinary Q5_0, Q8_0, Q4_K, Q5_K, and Q6_K weights remain in persistent
Metal buffers. Routed gate/up/down tensors instead remain file-backed: the GGUF
is mapped read-only and every layer owns an explicit LRU cache of `N` expert
bundles in shared Metal buffers. A route command returns top-k IDs, CPU fills
missing slots directly from the mapped expert slices and rewrites IDs to slot
indices, and an expert command resumes the GPU graph. Cached experts survive
`reset()` and can be reused by later turns.

Ten full-attention layers own FP32 K/V caches. The other 30 layers own
persistent FP32 Gated DeltaNet matrices and causal Conv1D history. `reset()`
clears sequence state and releases the previous turn's decode arena. It does not
discard expert LRU contents.

MoE forward uses one command for embedding, two commands per decoder layer
(attention/router, then experts), and one command for the final norm/LM head.
Hidden activations remain in one shared Arena; only route IDs cross the CPU/GPU
boundary. For correctness with finite caches, prefill automatically uses at
most `floor(expert_cache / top_k)` tokens per chunk. Position, full-attention KV
cache, and DeltaNet recurrent state remain continuous across chunks. Only the
last chunk runs the LM head. The implementation uses partial text iMRoPE,
top-8 routed experts, one shared expert, and a correctness-first serial-token
DeltaNet prefill.

For the downloaded model, resident weight-buffer bytes are:

```text
1,872,620,032 + 76,185,600 * expert_cache
```

This is 2,482,104,832 bytes (2367.120 MiB) at 8 slots and 3,091,589,632
bytes (2948.370 MiB) at 16 slots, instead of 21,376,133,632 bytes with all
256 experts resident. The read-only 20.75 GiB file mapping consumes virtual
address space; expert pages touched through it may enter the reclaimable OS file
cache and are not included in `weight_bytes`.

At the end of every turn Chat prints `weight_bytes`, `kv_cache_bytes`,
`recurrent_state_bytes`, `intermediate_bytes`, and their sum as
`peak_memory_bytes`. For Metal,
`intermediate_bytes` is the largest actual Arena allocated during that turn,
even when the prefill Arena has already been released, and
`kv_cache_bytes` is the actual private allocation. For CPU, the KV value is the
current FP32 cache payload and the intermediate value is a shape-based estimate
(`intermediate_kind=estimated`), because the reference CPU path does not use a
single Arena allocator.

## Inference path

For a prompt of `N` tokens, prefill performs:

```text
embedding [N, 896]
  -> 24 x (pre-norm GQA attention + residual)
  -> 24 x (pre-norm SwiGLU FFN + residual)
  -> final RMSNorm
  -> LM head
  -> greedy argmax token id
```

Decode embeds one generated token and runs the same 24 layers. Each layer
appends one rotated K row and one V row, then attends over the valid cached
prefix. Earlier tokens are not projected again.

Linear weights retain their quantized GGUF-native `[output, input]` layout.
Metal prefill computes `F32[M,K] * Quantized[N,K]^T -> F32[M,N]`; decode
computes `Quantized[N,K] * F32[K] -> F32[N]`, decoding each GGML block inside
the kernel and accumulating in FP32. Decode products use one Metal thread per
output channel; each thread decodes and consumes one complete quantized weight
row. Q/K/V use
`gemmtb` in CPU prefill and `gevmtb` in CPU decode because Qwen2.5 has Q/K/V
bias. Attention output, FFN gate/up/down, and the LM head are bias-free. Metal
kernels receive the equivalent transpose flags and dimensions.

For a `qwen35moe` prompt of `N` tokens, `MoeLLM` performs:

```text
embedding [N, 2048]
  -> 30 x (RMSNorm -> gated causal Conv1D DeltaNet -> residual)
  -> 10 x (RMSNorm -> gated GQA full attention -> residual)
  -> every layer: RMSNorm -> top-8 routed MoE + shared expert -> residual
  -> final RMSNorm -> LM head -> greedy argmax token id
```

Full-attention layers are 3, 7, ..., 39. Decode reuses their KV cache and the
DeltaNet recurrent/Conv1D state; it does not recompute earlier tokens.

## Profiling

Profiling is enabled by default. Relative filenames are placed under `output/`,
and the directory is created automatically. The CSV header is:

```text
record_type,forward_index,sequence_tokens,layer_index,stage,calls,time_ms,flops,read_bytes,write_bytes,temporary_bytes,logical_bytes,estimated_bytes_with_temporaries,allocations,gflops,logical_gbps,estimated_gbps,arithmetic_intensity,arithmetic_intensity_with_temporaries
```

The Qwen2 Metal path records `prefill.gpu_resident` and `decode.gpu_resident`;
the MoE path records `prefill.gpu_moe` and `decode.gpu_moe`. Their wall time
includes command encoding, GPU execution, synchronization, and the final token
readback. FLOPs and byte counts are logical shape-derived estimates, not
hardware performance counters.

Metal profiling also creates three sidecar files next to the selected CSV. For
`output/llm_profile.csv`, they are:

```text
output/llm_profile_metal_commands.csv
output/llm_profile_metal_kernels.csv
output/llm_profile_metal_ops.csv
```

The command file is the low-overhead baseline. It separates CPU command
creation/encoding/commit/wait time, `commit -> GPUStartTime` queue latency,
actual `GPUStartTime -> GPUEndTime`, kernel/threadgroup counts, and callback
delivery latency. Callback timestamps describe when the CPU handler ran; they
are not substitutes for `GPUStartTime` when measuring queue delay.

`--metal-kernel-profile` additionally records one row per dispatch and an
operation aggregate. Rows include layer/head, pipeline, M/N/K, dispatch grid,
CPU encoder time, GPU duration, preceding GPU gap, FLOPs, minimum tensor bytes,
quantized weight bytes, and shader load-request estimates. On devices such as
Apple M4, MyLLM uses start/end stage-boundary timestamps because dispatch-
boundary sampling is unavailable. Counter samples are split across multiple
buffers to support large models. This diagnostic mode perturbs scheduling, so
use the command-only run for production latency and the detailed run for the
relative operator breakdown. In particular, sampled inter-kernel gaps can
mostly be counter-sampling overhead.

At the end of each input turn, Chat writes prompt token count, generated token
count, prefill latency, aggregate decode latency, tokens per second, and the
weight/KV/recurrent/intermediate/peak memory fields to `stderr`.

## Metal access

Some managed shells deny the Metal/IOGPU user client. In that environment,
`MTLCreateSystemDefaultDevice()` and `MTLCopyAllDevices()` return no device even
when the Mac has a supported GPU. Because `--gpu` is explicit, MyLLM reports an
error instead of constructing `CPULLM`. Run Metal tests from a normal Terminal
session when the restricted shell cannot expose a device.

## Limitations

CPU weights are FP32 after dequantization. Metal matrix weights stay in their
supported GGUF quantized formats, while activations, KV caches, and DeltaNet
state remain FP32. Generation is greedy; temperature sampling, batching, fused
attention, conversation-history assembly, and production cache scheduling are
not implemented. `MoeLLM` intentionally skips `blk.40` MTP/NextN prediction and
all multimodal inputs; its iMRoPE path currently accepts text positions only.
The code is intended for operator inspection, correctness work, and latency
experiments rather than production throughput.

```bash
make clean
make help
```

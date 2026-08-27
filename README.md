# Standalone MyLLM

MyLLM is a self-contained Qwen2 inference demo. It does not call `llama.cpp`,
GGML, or another inference runtime. Both execution paths use KV-cache prefill
and one-token decode:

```text
chat
  |-- default --> CPULLM
  `-- --gpu  --> MetalLLM
```

`CPULLM` and `MetalLLM` are complete, independent backends. Each constructor
parses the GGUF through the shared `ModelFile` loader, owns its model weights,
owns one K/V cache per transformer layer, tracks the current sequence position,
and exposes `reset()`, `prefill()`, and `decode()`. There is no separate runtime
or session layer and `--gpu` does not silently fall back to CPU.

## Source layout

```text
chat.cpp          console UI, Qwen2 tokenizer, and autoregressive loop
chat.h            public generation helpers for CPULLM and MetalLLM
cpu_llm.h/.cpp    CPU tensors, operators, model ownership, KV cache, inference
model.h/.cpp      GGUF parser and FP32 dequantization shared by both backends
metal_llm.h/.mm   Metal model ownership, buffers, graph encoding, and inference
metal_llm.metal   FP32 Metal kernels
profiler.h/.cpp   CSV timing and logical FLOP/traffic statistics
```

The GGUF reader and F32/F16/Q5_0/Q8_0/Q4_K/Q6_K dequantizers are private to
`model.cpp`. This directory does not include source files from its parent
directory.

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

Options:

```text
--tokens N       Maximum generated tokens per input, default 32
--system TEXT    Replace the default ChatML system message
--raw            Tokenize input directly without ChatML wrapping
--gpu            Require the Metal backend; never fall back to CPU
--profile-csv P  Write profiling CSV under output/
--profile-log P  Compatibility alias for --profile-csv
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
make run PROFILE_CSV=qwen-profile.csv TOKENS=1
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

`MetalLLM` initializes the Metal device and pipelines, parses and dequantizes
the GGUF into temporary CPU tensors, uploads every weight once into persistent
`MTLBuffer` objects, and then releases the temporary CPU tensors. Inference
never looks up a CPU weight or uploads a weight before an operator.

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

One command buffer contains the complete prefill or decode graph. Kernel output
buffers are bound directly as following kernel inputs. Only token IDs enter
from CPU and one four-byte greedy argmax token ID returns after completion.
Transient activation buffers are recycled through a bounded pool.

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

Linear weights retain their GGUF-native `[output, input]` layout. Q/K/V use
`gemmtb` in CPU prefill and `gevmtb` in CPU decode because Qwen2.5 has Q/K/V
bias. Attention output, FFN gate/up/down, and the LM head are bias-free. Metal
kernels receive the equivalent transpose flags and dimensions.

## Profiling

Profiling is enabled by default. Relative filenames are placed under `output/`,
and the directory is created automatically. The CSV header is:

```text
record_type,forward_index,sequence_tokens,layer_index,stage,calls,time_ms,flops,read_bytes,write_bytes,temporary_bytes,logical_bytes,estimated_bytes_with_temporaries,allocations,gflops,logical_gbps,estimated_gbps,arithmetic_intensity,arithmetic_intensity_with_temporaries
```

The Metal path records `prefill.gpu_resident` and `decode.gpu_resident` as
end-to-end scopes. Their wall time includes command encoding, GPU execution,
synchronization, and the final token readback. FLOPs and byte counts are logical
shape-derived estimates, not hardware performance counters.

At the end of each input turn, Chat writes prompt token count, generated token
count, prefill latency, aggregate decode latency, and tokens per second to
`stderr`.

## Metal access

Some managed shells deny the Metal/IOGPU user client. In that environment,
`MTLCreateSystemDefaultDevice()` and `MTLCopyAllDevices()` return no device even
when the Mac has a supported GPU. Because `--gpu` is explicit, MyLLM reports an
error instead of constructing `CPULLM`. Run Metal tests from a normal Terminal
session when the restricted shell cannot expose a device.

## Limitations

Weights and KV caches are FP32 after GGUF dequantization. Generation is greedy;
temperature sampling, batching, fused attention, conversation-history assembly,
and production cache scheduling are not implemented. The code is intended for
operator inspection, correctness work, and latency experiments.

```bash
make clean
make help
```

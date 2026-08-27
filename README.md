# Standalone `llm` Chat Runtime

This directory contains the standalone Qwen2 runtime and its interactive text
entry point. It can be compiled without the top-level CMake project and without
`llama.cpp`, GGML, or any dynamic inference library. The chat executable
supports both the original complete-prefix path and an explicit per-turn
KV-cache prefill/decode path.

## Build

Run the following commands from this directory:

```bash
make
```

The Makefile compiles:

```text
chat.cpp              interactive entry point and Qwen2 tokenizer
llm.cpp               Vector/Matrix operators and Qwen2 attention/FFN operators
model.cpp             GGUF reader, model loading, and FP32 weight conversion
runtime.cpp           complete no-KV-cache forward and greedy next-token output
profiler.cpp          operator timing and logical FLOP/traffic statistics
metal_llm.h           Metal backend interface
metal_llm.mm          Metal device, buffer, pipeline, and forward integration
metal_llm.metal       FP32 Metal GEMM/GEVM kernels
```

The GGUF reader and the F32/F16/Q5_0/Q8_0/Q4_K/Q6_K dequantizers are private
implementation code inside `model.cpp`. The standalone target does not include
or compile any source or header from the parent `cpp` directory.

Object files are placed in `build/`; the executable is `./chat`.

The same build can be launched from the project root:

```bash
make -C cpp/MyLLM
```

`CXX`, `CPPFLAGS`, `CXXFLAGS`, `LDFLAGS`, and `LDLIBS` can be overridden for a
different compiler or target platform. The default compiler is `c++` and the
required language standard is C++17.

On macOS, the target links the system Metal and Foundation frameworks. The
Metal library is compiled from `metal_llm.metal` at runtime, so the standalone
build does not require the command-line `metal` tool. The executable searches
for the shader in its working directory and next to the executable; an explicit
location can be supplied with `MYLLM_METAL_SHADER=/path/to/metal_llm.metal`.

After the GGUF weights have been loaded, model-side FP32 matrices and bias
vectors used by linear layers are uploaded once into shared Metal buffers. Each
GEMM dispatch uses a 16x16 tiled kernel, while GEVM uses one thread per output
element. The existing `Matrix` type is intentionally unchanged, so transient
activation matrices are flattened before dispatch and copied back after the
command buffer completes. This first backend step synchronizes per linear
operator; it is intended for correctness and kernel experiments rather than a
fully fused, device-resident graph.

Metal is selected automatically when a device and valid shader are available.
If initialization fails, the program reports the reason and uses the original
CPU operators. Set `MYLLM_DISABLE_METAL=1` to force that CPU reference path for
token and timing comparisons.

## Run

From `cpp/MyLLM/`, use the model path relative to this directory:

```bash
./chat \
  --model ../../models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokens 4
```

Then type a line and press Enter. The input is wrapped as:

```text
<|im_start|>system
You are a concise and helpful assistant.<|im_end|>
<|im_start|>user
<input><|im_end|>
<|im_start|>assistant
```

The tokenizer reads `tokenizer.ggml.tokens` and
`tokenizer.ggml.merges` from the GGUF metadata. It performs Qwen2's
special-token splitting, Unicode pre-tokenization, GPT-2 byte mapping, full
pair BPE merge ranking, and byte-level decoding. Each generated token piece is
written and flushed immediately.

Interactive commands:

```text
/exit or /quit   Exit the program
/clear           Print a new-turn marker
```

Useful options:

```text
--tokens N       Maximum generated tokens, default 32
--system TEXT    Replace the default ChatML system message
--raw            Do not add ChatML; tokenize the input directly
--kv             Use per-conversation KV-cache prefill/decode
--profile-csv P  Write operator statistics CSV under output/
--profile-log P  Compatibility alias for --profile-csv
--no-profile     Disable profiling and do not create a profile CSV
--help           Print command-line help
```

Profiling is enabled by default for `chat`. It truncates the selected CSV when
the process starts. Each record is one flat row, so the file can be opened
directly in a spreadsheet or imported into a data-analysis tool. A one-token
run is enough to collect a complete prompt forward:

```bash
printf '你好\n/exit\n' | ./chat \
  --model ../../models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokens 1 \
  --profile-csv llm_profile.csv
```

Relative profile filenames are placed under `output/`; the directory and any
requested subdirectories are created automatically. For example, the command
above writes `output/llm_profile.csv`. An absolute path remains an explicit
override.

The header is:

```text
record_type,forward_index,sequence_tokens,layer_index,stage,calls,time_ms,flops,read_bytes,write_bytes,temporary_bytes,logical_bytes,estimated_bytes_with_temporaries,allocations,gflops,logical_gbps,estimated_gbps,arithmetic_intensity,arithmetic_intensity_with_temporaries
```

`record_type` is one of `forward_total`, `forward_stage`, `layer_total`,
`layer_stage`, or `global`. Context columns are empty when they do not apply.
Each row reports `calls`, `time_ms`, estimated `flops`, logical
`read_bytes`/`write_bytes`, `temporary_bytes`, `allocations`, `gflops`,
`logical_gbps`, and `arithmetic_intensity`. The main compute/memory indicators
are derived from tensor shapes and wall-clock time. These are software
estimates, not DRAM hardware counters.

The detailed stages include embedding, attention RMSNorm/QKV projection/head
split/RoPE/QK/mask/Softmax/AV/concat/output projection/residual, FFN
RMSNorm/gate projection/up projection/SwiGLU/down projection/residual, and the
final RMSNorm, LM-head GEMV, and argmax. Compare the `time_ms` and `gflops` of
the projection stages to find compute-kernel bottlenecks; a high
`logical_gbps` together with low arithmetic intensity indicates a memory or
allocation-heavy stage. `attention.total` and `ffn.total` are inclusive timing
scopes, so their child stage times must not be summed with the total again.
The FLOP and byte fields on `forward.total` and `layer.total` are sums of leaf
stages with those inclusive parent scopes excluded.

The Makefile also provides a convenience target:

```bash
make run TOKENS=4
make run MODEL=/path/to/model.gguf TOKENS=8
make run RAW=1 TOKENS=4
make run KV=1 TOKENS=4
make run PROFILE_CSV=qwen-profile.csv TOKENS=1
make run NO_PROFILE=1 TOKENS=1
```

`TOKENS` is the canonical Make variable. `TOKEN` is also accepted as an
alias, so `make run KV=1 TOKEN=16384` passes `--tokens 16384` to the program.

The default executable path is the complete-prefix no-KV implementation. To
run the layer-level KV-cache path, pass `--kv`:

```bash
printf '你好\n/exit\n' | ./chat \
  --model ../../models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokens 4 --kv --no-profile
```

At the end of every conversation turn, chat writes a timing summary to
`stderr`. `prefill_ms` covers the initial prompt and first LM-head prediction;
`decode_ms` covers subsequent generated-token predictions. In `--kv` mode,
the latter uses one-token `decode()` calls and the former fills one K/V matrix
per layer. In the default mode, both stages call `forward()`, so decode timing
includes recomputation of the complete prefix. The cache is local to one turn
and is released before the next prompt is read.

When Metal is enabled, the `--kv` path also routes Q/K/V projections, attention
QK/AV products, output projections, FFN projections, and the vocabulary GEVM
through `MetalLLM`; the cache itself remains in the existing CPU `Matrix`
objects in this first step.

## Forward path

For a prompt with `N` tokens, `llm_runtime::forward()` executes:

```text
embedding [N, 896]
  -> 24 x (pre-norm GQA attention + residual)
  -> 24 x (pre-norm SwiGLU FFN + residual)
  -> final RMSNorm
  -> LM head GEMV
  -> greedy argmax token id
```

The runtime keeps linear weights in their GGUF-native `[output, input]` form.
Q/K/V projections use `gemmtb` during full-sequence processing and `gevmtb`
during one-token decode because Qwen2.5 defines those three bias tensors.
Attention output, FFN gate/up/down, and the vocabulary LM head are bias-free and
use `gemmt` or `gevmt`. The loader requires every Q/K/V bias tensor and does not
keep optional output, FFN, or LM-head bias fields. Qwen2's 14 query heads share
2 KV heads through GQA. The current implementation uses the complete sequence
on every runtime call; after a token is appended, all previous K/V and FFN work
is recomputed. It is intended for operator inspection and correctness
experiments, not throughput benchmarking.

`llm.h` exposes layer-level `prefill()` and `decode()` functions. `prefill()`
fills one layer's `[position, kv_head * head_dim]` K/V matrices, storing K
after per-head RoPE. `decode()` appends one new K/V row, applies the absolute
RoPE position to the new Q/K, and computes attention over the cached prefix.
`chat --kv` owns one K/V pair per transformer layer for exactly one turn and
uses the runtime's layer wrappers around these APIs; no cache is retained across
turns.

## Memory and limitations

The model is fully dequantized to `float` while loading. The Qwen2.5-0.5B
model therefore needs substantially more memory than its approximately 468 MiB
Q4_K_M file. Generation is greedy only; temperature sampling, batching,
device-resident activation graphs, fused attention, conversation history, and
production-grade cache scheduling are not implemented. The layer-level
`prefill()`/`decode()` APIs and the `chat --kv` path are intended for correctness
and latency experiments.

The model file must exist before running. From `cpp/MyLLM/`, the default expected
location is:

```text
../../models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

## Clean and troubleshoot

```bash
make clean
make help
```

The model loader is self-contained in `model.cpp`; no parent-directory include
path or external inference library is required.

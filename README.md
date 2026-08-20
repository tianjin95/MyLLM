# Standalone `llm` Chat Runtime

This directory contains the FP32, no-KV-cache Qwen2 runtime and its interactive
text entry point. It can be compiled without the top-level CMake project and
without `llama.cpp`, GGML, or any dynamic inference library.

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
```

The GGUF reader and the F32/F16/Q5_0/Q8_0/Q4_K/Q6_K dequantizers are private
implementation code inside `model.cpp`. The standalone target does not include
or compile any source or header from the parent `cpp` directory.

Object files are placed in `build/`; the executable is `./chat`.

The same build can be launched from the project root:

```bash
make -C cpp/llm
```

`CXX`, `CPPFLAGS`, `CXXFLAGS`, `LDFLAGS`, and `LDLIBS` can be overridden for a
different compiler or target platform. The default compiler is `c++` and the
required language standard is C++17.

## Run

From `cpp/llm/`, use the model path relative to this directory:

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
--help           Print command-line help
```

The Makefile also provides a convenience target:

```bash
make run TOKENS=4
make run MODEL=/path/to/model.gguf TOKENS=8
make run RAW=1 TOKENS=4
```

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

The runtime stores linear weights in transposed `[input, output]` form so the
existing GEMM/GEMV operators can be called directly. Qwen2's 14 query heads
share 2 KV heads through GQA. The current implementation uses the complete
sequence on every call; after a token is appended, all previous K/V and FFN
work is recomputed. It is intended for operator inspection and correctness
experiments, not throughput benchmarking.

## Memory and limitations

The model is fully dequantized to `float` while loading. The Qwen2.5-0.5B
model therefore needs substantially more memory than its approximately 468 MiB
Q4_K_M file. Generation is greedy only; temperature sampling, batching,
accelerator kernels, conversation history, and KV cache are not implemented.

The model file must exist before running. From `cpp/llm/`, the default expected
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

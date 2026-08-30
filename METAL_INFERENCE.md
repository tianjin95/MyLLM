# MyLLM Metal GPU 推理流程详解

本文对应当前 `metal_llm.metal`、`metal_llm.mm` 和 `chat.cpp` 的真实实现，
从单个 GPU kernel 的数学计算与线程切分开始，一直说明到 Chat 如何取得并
打印下一个 token。

当前实现的核心边界是：

```text
chat.cpp                  CPU: 文本、分词、自回归控制、打印
    |
    v
MetalLLM::prefill/decode  CPU: 创建并编码一个 MTLCommandBuffer
    |
    v
encode_* helpers          CPU: 分配 Buffer、绑定参数、记录 kernel dispatch
    |
    v
metal_llm.metal           GPU: 执行 FP32 kernel
    |
    v
argmax token id           GPU -> CPU: 每一步只读取 4 字节 token id
```

这里必须区分三个概念：

1. `.metal` 中的 `kernel` 是 GPU 要执行的函数。
2. `.mm` 中的 `encode_*` 函数在 CPU 上运行，它们不直接做矩阵计算，而是
   把某个 kernel、Buffer 和网格大小记录进 command buffer。
3. `finish_async()` 才提交整个 command buffer，并等待 GPU 完成。

所以并不是每调用一个算子就进行一次 CPU/GPU 往返。一个完整的 prefill 或
decode 只提交一个 command buffer，但这个 command buffer 内包含大量 kernel
dispatch。

## 1. 本模型的固定参数

Qwen2.5-0.5B GGUF 中与本文有关的配置如下：

| 符号 | 含义 | 数值 |
|---|---|---:|
| `L` | Transformer 层数 | 24 |
| `H` | Hidden Size / `d_model` | 896 |
| `I` | FFN Intermediate Size | 4864 |
| `h_q` | Query Head 数量 | 14 |
| `h_kv` | Key/Value Head 数量 | 2 |
| `d` | 每个 Head 的维度 | 64 |
| `D_q` | Q 投影总宽度，`h_q * d` | 896 |
| `D_kv` | K/V 投影总宽度，`h_kv * d` | 128 |
| `Vocab` | 词表大小 | 151936 |
| `S_max` | 默认最大 Sequence | 32768 |

GQA 的分组大小为：

```text
group_size = h_q / h_kv = 14 / 2 = 7
```

因此：

```text
Q Head 0..6   -> KV Head 0
Q Head 7..13  -> KV Head 1
```

所有模型权重在 GGUF 中可能是 Q4_K_M 等量化格式，但进入当前 Metal 路径前，
都会先反量化为 FP32。当前所有 GPU 矩阵、激活与 KV Cache 都使用 FP32；
token id 使用 `int32`，argmax 输出使用 `uint32`。

## 2. GPU Buffer 与张量描述符

### 2.1 DeviceMatrix 和 DeviceVector

`DeviceMatrix`、`DeviceVector` 是 CPU 侧的小型描述符：

```text
DeviceMatrix = MTLBuffer + rows + cols + stride + offset_elements
DeviceVector = MTLBuffer + length + offset_elements
```

它们本身不存储数据。真实数据位于 `MTLBuffer`。`offset_elements` 的单位是
FP32 元素，绑定到 Metal encoder 时再乘 `sizeof(float)` 转换成字节偏移。

这个设计允许不复制数据就建立视图，例如：

- prefill 最后只取 `[N, H]` 的最后一行作为 `[H]`。
- Attention 不创建 14 个 Q Buffer，而是对同一个 `[N, 896]` Buffer 使用
  `head * 64` 的起始偏移。
- AV kernel 直接写入最终 `[N, 896]` 的对应 Head 区域，不需要 concat kernel。

### 2.2 Buffer 的存储模式

当前实现中：

| 数据 | Storage Mode | 生命周期 |
|---|---|---|
| 模型权重 | `MTLResourceStorageModeShared` | `MetalLLM` 整个生命周期 |
| 激活 Arena | `MTLResourceStorageModePrivate` | `MetalLLM` 整个生命周期，按固定 slot 复用 |
| Token IDs | `MTLResourceStorageModeShared` | 一次 command |
| Argmax 结果 | `MTLResourceStorageModeShared` | CPU 读取后回收 |
| K/V Cache | `MTLResourceStorageModePrivate` | `MetalLLM` 整个生命周期 |

Apple Silicon 使用统一内存，因此 Shared Buffer 不代表每个 kernel 前都从 CPU
复制到独立显存。它代表 CPU 和 GPU 都能访问同一块系统内存。当前流程仍然有
两类显式 CPU 写入：

- 初始化时把反量化权重 `memcpy` 到持久化 Metal Buffer。
- 每个 prefill/decode 开始时写入 token id。

算子之间只传递 Buffer，不会把中间矩阵转换回 CPU `Matrix`。

KV Cache 使用 Private Buffer，CPU 没有映射它的内容。每层分别持有：

```text
K_cache: [S_max, 128]
V_cache: [S_max, 128]
```

## 3. 通用线程派发方式

### 3.1 一维 Flat Dispatch

Embedding、SwiGLU、Residual、RoPE、KV 写入、QK 和 AV 都使用
`dispatch_flat()`：

```text
threads_per_threadgroup = min(256, pipeline 支持的最大线程数)
threadgroup_count       = ceil(total_elements / threads_per_threadgroup)
```

kernel 使用 `thread_position_in_grid` 得到全局一维 `index`，再把它还原为
`row`、`col`、`head dim` 等逻辑坐标。最后一个 threadgroup 可能有多余线程，
所以每个 kernel 都先检查 `index >= total`。

### 3.2 逐行 Reduction Dispatch

RMSNorm 和 Softmax 使用：

```text
threadgroup_count       = rows
threads_per_threadgroup = 256
```

一个 threadgroup 独占一行。组内 256 个线程先各自处理：

```text
col = local_index, local_index + 256, local_index + 512, ...
```

然后通过 `threadgroup` 内存和树形 reduction 合并结果。这里只能同步同一个
threadgroup，不能同步整个 GPU 网格。

### 3.3 GEMM 二维 Tile Dispatch

GEMM 使用固定的 `16 x 16` threadgroup：

```text
threads_per_threadgroup = [16, 16, 1] = 256 threads
threadgroup_grid        = [ceil(N / 16), ceil(M / 16), 1]
```

一个 threadgroup 计算输出矩阵的一个 `[16, 16]` Tile；一个线程负责 Tile
中的一个输出元素。

## 4. 每个 Kernel 的计算过程

### 4.1 Embedding: `metal_embedding_f32`

输入与输出：

```text
token_ids: [N]             int32
embedding: [Vocab, H]      FP32
output:    [N, H]          FP32
```

数学计算：

```text
output[position, col] = embedding[token_ids[position], col]
```

线程切分：

```text
total threads = N * H
一个线程负责一个 output[position, col]
```

对于当前模型，`H=896`，因此 N 个 token 产生 `N*896` 个逻辑线程。
同一个 token 的 896 个线程读取相同 token id，并读取 Embedding 的一整行。

### 4.2 GEMM: `metal_gemm_f32`

通用公式：

```text
C[m, n] = scale * sum(k, A[m, k] * B[k, n]) + bias[n]
```

Bias 由 `has_bias` 控制；Qwen2.5 只在 Q/K/V 投影中使用 Bias。

权重物理布局保持 GGUF 原生的 `[output, input]`。例如 Q 投影：

```text
A 物理/逻辑形状: [N, 896]
Wq 物理形状:     [896, 896]
逻辑计算:         A * Wq^T
输出:             [N, 896]
```

host 因此传入 `rhs_transposed=true`。kernel 的 `load_rhs()` 会读取：

```text
Wq[col, k]
```

而不是先创建一份真正转置后的权重。

Tile 计算步骤：

1. `group_id` 决定输出 `[16,16]` Tile 的位置。
2. `local_id.y` 决定 Tile 内输出行，`local_id.x` 决定输出列。
3. 每个线程从 A 加载一个元素到 `lhs_tile[16][16]`。
4. 每个线程从 B 加载一个元素到 `rhs_tile[16][16]`。
5. 第一次 `threadgroup_barrier` 保证整个 Tile 已加载完成。
6. 每个线程对当前 16 个 K 元素执行 16 次乘加。
7. 第二次 barrier 防止下一轮 K Tile 覆盖仍在使用的共享数据。
8. 重复直到遍历完整 K。
9. 对累加值应用 scale、Bias，并写回一个输出元素。

边缘 Tile 中超出 M、N 或 K 的加载会填零。

以 Q 投影为例：

```text
M = N(prompt tokens)
N = 896(output channels)
K = 896(input channels)
threadgroup grid = [56, ceil(prompt_tokens/16), 1]
K Tile 次数 = 56
```

FFN Gate/Up 输出宽度 4864，所以 X 方向有 `4864/16=304` 个 threadgroup。
FFN Down 的 K 为 4864，因此每个输出线程要遍历 304 个 K Tile。

当前 GEMM 是可读性优先的基础 Tile 实现，还没有使用 simdgroup matrix、
Metal Performance Shaders 或更复杂的寄存器分块。

### 4.3 GEVM/GEMV: `metal_gevm_f32`

Decode 的输入只有一个 token，因此 Dense 投影退化为矩阵向量乘法：

```text
y[out] = scale * sum(k, x[k] * W[out, k]) + bias[out]
```

线程切分：

```text
一个线程负责一个输出通道 out
每个线程内部串行遍历全部 input_size
```

没有跨线程 reduction。对于 `[output,input]` 权重和
`matrix_transposed=true`，每个线程连续读取 `W[out, :]`，这对单个线程的
内存访问是连续的。

当前模型中常见的派发规模，按每组 256 线程计算：

| 投影 | Output | Input | Threadgroups |
|---|---:|---:|---:|
| Q | 896 | 896 | 4 |
| K/V | 128 | 896 | 1 |
| Attention Output | 896 | 896 | 4 |
| FFN Gate/Up | 4864 | 896 | 19 |
| FFN Down | 896 | 4864 | 4 |
| LM Head | 151936 | 896 | 594 |

这种设计让输出通道提供并行性，但每个线程有一条较长的串行 FMA 依赖链。

### 4.4 RMSNorm: `metal_rmsnorm_f32`

每一行独立计算：

```text
mean_square = sum(i, x[i]^2) / H
inverse_rms = rsqrt(mean_square + epsilon)
y[i]        = x[i] * inverse_rms * gamma[i]
```

一个 threadgroup 负责一行，步骤为：

1. 256 个线程分段计算各自的平方和。
2. 每个线程把局部和写入 `threadgroup float partial[256]`。
3. 依次以 128、64、32、...、1 为 stride 做树形求和。
4. 所有线程读取 `partial[0]` 得到同一个 `inverse_rms`。
5. 256 个线程再次分段遍历该行，完成归一化和 Gamma 乘法。

Prefill 输入 `[N,896]`，会派发 N 个 threadgroup。Decode 输入 `[896]`，
等价于一行，只派发一个 threadgroup。

### 4.5 Softmax: `metal_softmax_f32`

Softmax 同样逐行独立：

```text
maximum = max(x)
sum_exp = sum(exp(x[i] - maximum))
y[i]    = exp(x[i] - maximum) / sum_exp
```

一个 threadgroup 负责一个 Attention Score 行，进行三次行遍历：

1. 求局部最大值，再做树形 max reduction。
2. 计算局部 `exp` 和，再做树形 sum reduction。
3. 再计算一次 `exp` 并写出概率。

`-INFINITY` 表示 Causal Mask，被显式转换为概率 0。当前实现为简单起见，
第二、第三阶段会重复计算 `exp`。

### 4.6 SwiGLU: `metal_swiglu_f32`

数学计算：

```text
SiLU(g) = g * sigmoid(g)
output  = SiLU(gate) * up
```

一个线程负责一个 `[row, ffn_col]` 元素，不需要同步。Sigmoid 根据正负值
使用两个等价公式，避免对很大的正数执行不稳定的 `exp(+x)`。

Prefill 派发 `N*4864` 个逻辑线程；Decode 派发 4864 个逻辑线程。

### 4.7 Residual: `metal_residual_f32`

```text
output[index] = left[index] + right[index]
```

一个线程负责一个 FP32 元素。Attention 后和 FFN 后各执行一次。

### 4.8 RoPE: `metal_rope_heads_f32`

Q/K 仍保持打包布局：

```text
Q: [rows, 14 * 64] = [rows, 896]
K: [rows,  2 * 64] = [rows, 128]
```

kernel 根据列坐标计算所在 Head：

```text
head_start = floor(col / head_dim) * head_dim
local_col  = col - head_start
```

当前使用 NeoX 风格的前后半区配对。`rotary_dimension=64` 时：

```text
pair 0:  local 0  <-> local 32
pair 1:  local 1  <-> local 33
...
pair 31: local 31 <-> local 63
```

频率和旋转为：

```text
frequency = theta^(-2*pair/rotary_dimension)
angle     = (start_position + row) * frequency
a'        = a*cos(angle) - b*sin(angle)
b'        = a*sin(angle) + b*cos(angle)
```

一个线程负责一个输出标量，所以同一对 `(a,b)` 会由两个线程分别计算
`a'` 和 `b'`。这两个线程都会计算 sin/cos，当前没有预计算 RoPE 表。

V 不执行 RoPE。K 在写入 KV Cache 前完成 RoPE，因此历史 K 不需要重复旋转。

### 4.9 KV Cache Write: `metal_kv_cache_write_f32`

输入 K/V 是连续的：

```text
source K/V: [source_rows, 128]
cache K/V:  [S_max, 128]
```

每个线程复制同一个 `(row,col)` 的 K 和 V：

```text
destination = (position + row) * 128 + col
K_cache[destination] = K_source[row,col]
V_cache[destination] = V_source[row,col]
```

Prefill 中 `position=0`、`source_rows=N`；Decode 中 `position=P`、
`source_rows=1`。

### 4.10 QK 与 Causal Mask: `metal_kv_cache_qk_f32`

host 每次只处理一个 Query Head，并通过 Buffer Offset 指向该 Head 的第一个
元素。输出形状为：

```text
scores: [query_rows, key_length]
```

一个线程负责一个 `(query_row,key_row)` Score，在线程内部串行计算 64 维
点积：

```text
score = sum(dim=0..63, Q[query_row,dim] * K_cache[key_row,kv_head,dim])
score = score / sqrt(64)
```

Causal Mask 已融合进 QK kernel：

```text
absolute_query_position = query_position + query_row
if key_row > absolute_query_position:
    score = -INFINITY
```

因此当前 Metal 路径没有单独的 Mask kernel。

GQA 映射在 CPU host 循环中确定：

```text
kv_head = query_head / 7
```

Prefill 每个 Q Head 派发 `N*N` 个逻辑线程，每个线程做 64 维点积。
Decode 每个 Q Head 派发 `P+1` 个逻辑线程。

### 4.11 Attention AV: `metal_kv_cache_av_f32`

Softmax 后：

```text
context[query_row,dim] =
    sum(key_row, probability[query_row,key_row] * V_cache[key_row,kv_head,dim])
```

一个线程负责一个 `(query_row, head_dim)` 输出，并在线程内部串行遍历
`key_length`。

14 个 Head 共用同一个最终 Attention Buffer `[query_rows,896]`。host 给每个
Head 设置：

```text
output_offset = query_head * 64
output_stride = 896
```

因此每个 AV kernel 直接写入自己的列区间，不需要 Split Heads 或 Concat
Buffer。

### 4.12 Argmax: `metal_argmax_f32`

输入是 `[151936]` FP32 Logits，输出是一个 `uint32 token_id`。

当前 kernel 只派发一个 GPU 线程，该线程串行扫描完整词表：

```text
best = 0
for candidate in 1..151935:
    if logits[candidate] > logits[best]:
        best = candidate
```

优点是 CPU 最终只读取 4 字节，而不是读取约 594 KiB Logits。缺点是 Argmax
没有并行化，是明确的基础实现；更高性能的版本应使用 threadgroup 局部归约
加第二阶段全局归约。

## 5. Host 如何把一个算子变成 Command

以 GEMM 为例，`encode_matrix_product()` 在 CPU 上依次执行：

1. 根据 transpose flag 验证 M、N、K。
2. 从启动时规划好的激活 Arena 取得输出 slot 视图，不申请新的激活 `MTLBuffer`。
3. 填充 `MetalMatmulParamsHost`。
4. 从 command buffer 创建 `MTLComputeCommandEncoder`。
5. 设置 `metal_gemm_f32` 对应的 Pipeline State。
6. 使用 `setBuffer` 绑定 LHS、RHS、Bias、Output。
7. 使用 `setBytes` 把 36 字节参数复制进 command 数据。
8. 使用 `dispatchThreadgroups` 记录二维线程网格。
9. `endEncoding()` 结束这个 encoder。
10. 立即返回一个指向 Output Buffer 的 `DeviceMatrix` 描述符。

此时 GEMM 还没有执行。下一个 `encode_*` 可以立刻把这个 Output Buffer
绑定为输入。所有 encoder 都属于同一个 command buffer，并按记录顺序执行。

不同 kernel 之间不需要 `threadgroup_barrier`。该 barrier 只能同步一个
threadgroup，不能同步不同 dispatch。跨 kernel 的顺序和资源可见性由同一个
command buffer 的编码顺序及 Metal Resource Hazard Tracking 保证。

## 6. 一层 Prefill 的完整流程

设 Prompt 长度为 `N`，某层输入为 `hidden [N,896]`。

| 步骤 | Kernel | 输入 | 输出 |
|---:|---|---|---|
| 1 | RMSNorm | `[N,896]`, gamma `[896]` | `A [N,896]` |
| 2 | GEMM + Bias | `A`, `Wq [896,896]` | `Q [N,896]` |
| 3 | GEMM + Bias | `A`, `Wk [128,896]` | `K [N,128]` |
| 4 | GEMM + Bias | `A`, `Wv [128,896]` | `V [N,128]` |
| 5 | RoPE | `Q [N,896]` | `rotated_Q [N,896]` |
| 6 | RoPE | `K [N,128]` | `rotated_K [N,128]` |
| 7 | KV Write | rotated K, V | Cache 行 `[0,N)` |
| 8 | QK，14 次 | 每个 Q Head 和对应 KV Head | 14 个 `[N,N]` Score |
| 9 | Softmax，14 次 | 14 个 Score | 14 个 `[N,N]` Probability |
| 10 | AV，14 次 | Probability 和 V Cache | 打包 `C [N,896]` |
| 11 | GEMM | `C`, `Wo [896,896]` | `O [N,896]` |
| 12 | Residual | hidden, O | `X [N,896]` |
| 13 | RMSNorm | X, FFN gamma | `F [N,896]` |
| 14 | GEMM | F, Gate Weight `[4864,896]` | `G [N,4864]` |
| 15 | GEMM | F, Up Weight `[4864,896]` | `U [N,4864]` |
| 16 | SwiGLU | G, U | `S [N,4864]` |
| 17 | GEMM | S, Down Weight `[896,4864]` | `D [N,896]` |
| 18 | Residual | X, D | 下一层 Hidden `[N,896]` |

步骤 8 到 10 由 host 的 14 次循环编码。每个 Head 编码三个 kernel：

```text
QK -> Softmax -> AV
```

所以一层的 kernel dispatch 数为：

```text
Attention 前半: RMSNorm 1 + QKV 3 + RoPE 2 + KV Write 1 = 7
14 个 Head:     14 * (QK 1 + Softmax 1 + AV 1)          = 42
Attention 后半: Output Projection 1 + Residual 1        = 2
FFN:            Norm 1 + Gate/Up 2 + SwiGLU 1
                + Down 1 + Residual 1                   = 6
每层合计:                                                   57
```

24 层就是 `24*57=1368` 次 dispatch。

完整 prefill 还包括：

```text
Embedding 1 + 24 层 1368 + Final RMSNorm 1
+ LM Head GEVM 1 + Argmax 1 = 1372 dispatches
```

### Prefill 最后一段

24 层完成后：

1. 当前实现对整个 `[N,896]` Hidden 执行 Final RMSNorm。
2. 通过 Buffer Offset 建立最后一行 `[896]` 的视图，不复制数据。
3. 使用 LM Head GEVM：`[151936,896] x [896] -> [151936]`。
4. Argmax 得到第一个生成 token。

这里只需要最后一行，但当前 Final RMSNorm 仍处理全部 N 行，这是一个可以继续
优化的点。

## 7. 一层 Decode 的完整流程

假设当前缓存已有 `P` 个 token，Decode 输入是刚刚生成、准备加入序列的一个
token id。

```text
position   = P
key_length = P + 1
```

Embedding 产生 `[1,896]`，host 直接把它视为 `[896]` Vector。每层执行：

| 步骤 | Kernel | 输出形状 |
|---:|---|---|
| 1 | RMSNorm | `[896]` |
| 2 | Q GEVM + Bias | `[896]` |
| 3 | K GEVM + Bias | `[128]` |
| 4 | V GEVM + Bias | `[128]` |
| 5 | Q RoPE，Position=P | `[896]` |
| 6 | K RoPE，Position=P | `[128]` |
| 7 | KV Write | 写 Cache 第 P 行 |
| 8 | QK，14 次 | 每个 Head `[P+1]` |
| 9 | Softmax，14 次 | 每个 Head `[P+1]` |
| 10 | AV，14 次 | 直接写入 `[896]` |
| 11 | Output GEVM | `[896]` |
| 12 | Attention Residual | `[896]` |
| 13 | FFN RMSNorm | `[896]` |
| 14 | Gate/Up GEVM | 两个 `[4864]` |
| 15 | SwiGLU | `[4864]` |
| 16 | Down GEVM | `[896]` |
| 17 | FFN Residual | 下一层 `[896]` |

Decode 仍然是每层 57 个 kernel、完整一步 1372 个 dispatch，只是 GEMM 变为
GEVM，Score 从 `[N,N]` 变为 `[P+1]`。

Decode 的 QK 公式仍包含 Causal 判断，但 `key_length=P+1`，不存在大于当前位置
的 Key，因此不会屏蔽有效元素。

## 8. Command Buffer 的完整生命周期

一次 `MetalLLM::prefill()` 或 `decode()` 的时序是：

```text
CPU: begin_async()
     `- 从 MTLCommandQueue 创建 1 个 MTLCommandBuffer

CPU: encode_embedding()
CPU: for layer in 0..23
       encode layer 中的 57 个 kernel
CPU: encode_final_rmsnorm()
CPU: encode_lm_head()
CPU: encode_argmax()

CPU: finish_async()
     |- commandBuffer.commit()
     |- commandBuffer.waitUntilCompleted()
     `- 检查 Metal error/status

CPU: 从 Shared Argmax Buffer 读取 uint32 token id
CPU: 释放本次 command 的 token-id/argmax IO 引用；激活仍留在 Arena
CPU: 更新 sequence_length
```

虽然函数名中有 `async`，对外的 `prefill()` 和 `decode()` 仍是同步 API，因为
它们必须等到 Argmax 后才能把 token 返回给 Chat。`async` 表示内部编码阶段
不会在每个 kernel 后等待。

### 8.1 激活 Arena 生命周期

`MetalLLM::Impl::allocate_activation_arena()` 在模型加载完成后调用一次。它
根据 `max_sequence` 为每个逻辑 slot 计算字节偏移和容量，然后申请一个
`MTLResourceStorageModePrivate` 的大 `MTLBuffer`。之后每个 `encode_*` 只构造
带有 `offset_elements` 的 `DeviceMatrix`/`DeviceVector` 视图；中间激活不会再
触发逐算子 `newBufferWithLength:`。

当前 slot 的主要复用关系是：

```text
HiddenA <-> HiddenB       层间 ping-pong
Norm                       Attention RMSNorm / FFN RMSNorm / Final RMSNorm
AttentionScores            每个 Head、每个 Layer 的 QK 后 Softmax 原地覆盖
Projection                 Attention output projection / FFN down projection
Gate                       SwiGLU 原地写回 gate
```

一个 command buffer 内的 encoder 按顺序执行。前一个 kernel 完成写入后，后续
kernel 才读取同一 slot；Softmax 和 SwiGLU 的同址读写只对每个元素写回，且
Softmax 在写出前已完成行内 reduction，因此这些别名在当前 kernel 语义下是
安全的。不同层之间复用 slot 也安全，因为同一 command 中层序是串行依赖的。

Arena 的容量是一次性按 `max_sequence` 预留，不做大小分桶。当前非融合 Prefill
Attention 需要一个 `[max_sequence, max_sequence]` FP32 Score/Probability 区域，
所以默认使用模型上下文长度时内存可能很大；这也是 `--max-sequence` 存在的
原因。该参数不能超过 GGUF 的 context length。

## 9. MetalLLM 初始化过程

`chat --gpu` 只构造一次 `MetalLLM`：

1. `MTLCreateSystemDefaultDevice()` 获取 GPU。
2. 创建 `MTLCommandQueue`。
3. 读取 `metal_llm.metal` 源码。
4. `newLibraryWithSource` 在运行时编译 Metal Library。
5. 为每个 kernel 创建一个 `MTLComputePipelineState`。
6. `ModelFile` 读取 GGUF Metadata 和 Tensor。
7. GGUF 权重临时反量化为 CPU FP32 `Matrix/Vector`。
8. `upload_model()` 把所有权重一次性放入持久化 Metal Buffer。
9. CPU 临时权重离开作用域并释放。
10. 为 24 层分配固定容量的 Private K/V Buffer。
11. 按 `max_sequence` 计算所有激活 slot 的偏移，并分配一个 Private
    activation Arena。

之后所有对话共享同一套权重和已分配 KV Buffer。`reset()` 只把
`sequence_length` 设为 0，不清零 GPU Buffer，因为后续 kernel 只读取逻辑有效
范围，而新 Prefill 会覆盖从第 0 行开始的 Cache。

## 10. 从 Chat 输入到生成文本

`chat.cpp` 的 GPU 调用链如下：

```text
run_cli()
  |- 解析 --gpu
  |- 构造 MetalLLM(model_path)
  `- run_interactive()

用户输入字符串
  |- 添加 ChatML system/user/assistant 包装
  |- QwenTokenizer.encode()
  `- 得到 initial_sequence: vector<int32_t>

generate()
  |- backend.reset()
  |- next = backend.prefill(initial_sequence)
  |- 检查 next 是否为停止 token
  |- tokenizer.decode_piece(next)
  |- 立即打印字符串片段
  `- next = backend.decode(next)，循环直到停止或达到 token 数量
```

这里有一个容易混淆的 Sequence 时序：

1. Prefill 把 Prompt 的 N 个 token 写入 KV Cache，返回第一个生成 token `t0`。
2. `t0` 此时还没有进入 KV Cache。
3. Chat 先打印 `t0`。
4. `decode(t0)` 在 Position N 计算并写入 `t0` 的 K/V，然后预测 `t1`。
5. Chat 打印 `t1`，再调用 `decode(t1)`。

每一轮控制台输入都会调用 `reset()`，所以当前 Chat 不会自动把上一行对话历史
拼回新 Prompt。跨轮上下文如果需要保留，应由 Chat 层构造完整历史 token 序列，
而不是依赖旧 KV Buffer 中未清零的字节。

## 11. 当前实现没有做什么

为了准确理解性能，还需要明确当前没有以下优化：

- 没有 Flash Attention，QK Score 和 Softmax Probability 会显式落到 Buffer。
- 没有将多个 Q Head 合并成一次 QK/Softmax/AV dispatch。
- 没有融合 QKV 投影。
- 没有融合 RMSNorm 与后续 Projection。
- 没有融合 Bias、Residual、SwiGLU 与相邻 GEMM。
- 没有并行 Argmax。
- 没有 simdgroup matrix 或 MPS GEMM。
- 没有 FP16/BF16/INT8/INT4 Metal 计算。
- 没有使用 Flash Attention，因此 Score/Probability slot 仍按二维矩阵预留。
- 没有把多个 decode token 放入同一 command，因为下一个 token 依赖上一步
  Argmax。

当前实现已经消除了逐算子的 CPU 中间结果回传，但“大量小 kernel dispatch”
仍然存在。对于 Decode，很多工作量很小的 kernel 只使用一个 threadgroup，
dispatch/encoder 开销可能接近甚至超过计算本身。后续优化应优先依据 GPU 时间线
验证以下方向：合并 Head dispatch、并行 Argmax、融合逐元素算子、优化 GEVM，
最后再考虑更复杂的 Attention 融合。

## 12. 代码导航

| 目标 | 文件/函数 |
|---|---|
| GPU kernel | `metal_llm.metal` |
| Metal 参数结构、Pipeline、Arena | `metal_llm.mm`, `MetalLLM::Impl` |
| 单层 Prefill 图 | `MetalLLM::Impl::encode_prefill_layer` |
| 单层 Decode 图 | `MetalLLM::Impl::encode_decode_layer` |
| 完整 Prefill Command | `MetalLLM::prefill` |
| 完整 Decode Command | `MetalLLM::decode` |
| GGUF 加载和权重上传 | `MetalLLM::Impl::load_model` / `upload_model` |
| GPU KV Cache 分配 | `MetalLLM::Impl::allocate_kv_cache` |
| 自回归循环 | `chat.cpp`, `generate` |
| 控制台后端选择 | `chat.cpp`, `run_cli` |

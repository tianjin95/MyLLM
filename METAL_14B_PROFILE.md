# Qwen2.5-14B Metal Decode 性能分析

测试日期：2026-09-01

测试环境：24 GB Apple M4 Mac mini，10-core GPU。模型为
`Qwen2.5-14B-Instruct-Q4_K_M`，`max_sequence=256`，Batch=1，输入为一个 raw
token。Prompt 长度 1 的 Prefill 不代表常规 Prefill 性能，本文只分析稳定
Decode。

## 1. 两类采样

无逐 kernel counter 的低扰动结果：

```text
output/profile_14b_command_metal_commands.csv
```

逐 kernel stage-boundary timestamp 结果：

```text
output/profile_14b_detail_chunked_metal_commands.csv
output/profile_14b_detail_chunked_metal_kernels.csv
output/profile_14b_detail_chunked_metal_ops.csv
```

详细模式会人为增加 kernel 间隙。真实 token latency、CPU 下发和 command queue
等待必须读取低扰动结果；详细结果只用于分解 kernel 本体。

## 2. 真实 Decode 时间线

低扰动模式连续两个 Decode 的平均结果：

| 指标 | 平均值 | 占同步调用时间 |
|---|---:|---:|
| CPU 编码整张图 | 2.747 ms | 1.42% |
| `commit()` | 0.007 ms | <0.01% |
| Commit 到 GPUStart | 0.303 ms | 0.16% |
| GPUStart 到 GPUEnd | 190.037 ms | 98.34% |
| 同步调用总时间 | 193.248 ms | 100% |
| Decode 速度 | 约 5.17 token/s | - |

每个 Decode 只有一个 command buffer，但内部包含 6484 个 compute dispatch、
20776 个 threadgroup。5858 个 kernel 只有一个 threadgroup，占 90.35%。

结论：CPU 编码、`commit()` 和 command queue 排队都不是当前主瓶颈。不能通过
把一个 Decode 再合成“一个大 command buffer”获得明显收益，因为现在已经只有
一个 command buffer。大量小 kernel 的问题发生在 GPU command 内部。

## 3. GPU Kernel 分解

详细模式中，GPU command 为 301.613 ms，其中 kernel 本体合计 200.268 ms，
stage-boundary 采样引入的间隙合计 101.345 ms。Kernel 本体只比低扰动 GPU
时间高约 5.4%，可用于算子占比分析；101.345 ms 的 gap 不能当作真实运行间隙。

| 类别 | Kernel 时间 | Kernel 占比 | 量化权重读取 | 权重占比 |
|---|---:|---:|---:|---:|
| FFN | 103.024 ms | 51.44% | 6.171 GB | 72.25% |
| Attention | 69.529 ms | 34.72% | 1.731 GB | 20.27% |
| LM Head/Argmax 等输出 | 27.715 ms | 13.84% | 0.639 GB | 7.48% |
| 合计 | 200.268 ms | 100% | 8.541 GB | 100% |

主要算子：

| 算子 | 调用数 | GPU 时间 |
|---|---:|---:|
| FFN Gate Q4_K GEVM | 48 | 28.308 ms |
| FFN Up Q4_K GEVM | 48 | 28.784 ms |
| FFN Down Q4_K GEVM | 24 | 14.070 ms |
| FFN Down Q6_K GEVM | 24 | 30.865 ms |
| Attention Q/K/V/O GEVM 合计 | 192 | 38.439 ms |
| 每 Head QK | 1920 | 18.398 ms |
| 每 Head Softmax | 1920 | 6.776 ms |
| 每 Head AV | 1920 | 4.746 ms |
| LM Head Q6_K GEVM | 1 | 17.706 ms |
| 串行 Argmax | 1 | 9.995 ms |

## 4. 算力还是访存

每个 token 至少流过约 8.541 GB 量化权重。按低扰动 GPU 时间计算，端到端有效
权重带宽约为 44.9 GB/s；相对基础 M4 的标称 120 GB/s 统一内存带宽约为 37%。
只统计量化 GEVM kernel，本次采样约为 54.0 GB/s：Q4_K 约 62.2 GB/s，Q6_K
约 38.7 GB/s。

Q4_K/Q6_K Decode 的模型乘加算术强度只有约 2.4 到 3.6 FLOP/byte，远低于
GPU 的计算/带宽平衡点，因此它首先是权重流式读取敏感的工作负载。另一方面，
当前带宽没有接近 120 GB/s，说明它也不是“内存控制器已完全打满”的纯带宽
瓶颈。标量 block 解码、位提取/scale 计算、访存合并、占用率和小矩阵派发共同
限制了可达到的带宽。模型 FLOPs 没有统计反量化指令，所以较低的模型 GFLOPS
不能单独证明 ALU 空闲。

更准确的结论是：当前主瓶颈是量化 GEVM 的权重流式读取与在线反量化效率，
不是 FP32 峰值算力；第二个瓶颈是 Attention 的极细粒度派发；第三个明确问题是
单线程 Argmax。

## 5. 优化优先级

1. 优化 Q4_K/Q6_K GEVM，尤其是 Q6_K FFN Down 和 LM Head。目标是提高合并
   访存、并行输出数和 occupancy，同时控制寄存器与 reduction 开销。此前“四个
   输出/一个 SIMD Group”的实现降低了输出并行度并增加 reduction，实测没有收益。
2. 把同一层 40 个 Q Head 的 QK、Softmax、AV 分别合成按 Head 维派发的 kernel。
   这可把每 token 的 5760 个 Head kernel 降到 144 个；若进一步融合 Attention，
   可降到 48 个。
3. 将 Argmax 改成 threadgroup/SIMD 分层 reduction。当前一个线程扫描 152064
   个 logit，单独消耗约 10 ms。
4. 在前三项完成后，再评估 RMSNorm、Residual、RoPE、SwiGLU 等融合。它们的
   kernel 本体合计很小，不应先做。
5. 不要优先拆分 command buffer、增加 CPU 并发提交或优化 command queue；现有
   相关开销不到 2%。

## 6. Q4_K Decode GEVM Packed-Byte 复用

测试日期：2026-09-02。

第一版实验将 Q4_K 的内层循环按 4 元素展开，并使用 4 个独立 FP32 累加器；
Q6_K 则把一次循环天然产生的 4 个乘积分配给 4 个累加器。该版本没有收益：
低扰动 GPU Decode 只变化约 0.45%，详细采样中 Q4_K FFN Gate/Up 反而慢约
3.7%，Q6_K 基本不变。说明编译器已经处理了简单展开，而额外寄存器可能降低了
occupancy，因此该版本已撤销。

最终保留的优化利用 Q4_K 的实际布局：一个 packed byte 的低、高潮 nibble 分别
对应相邻两个 32 元素 activation half。原实现分两轮读取同一组 packed byte；新
实现一次读取后同时计算两个 half，并使用两条独立累加链。该实现只用于 Decode
GEVM，Prefill GEMM 继续使用原始点积，避免把 Decode 调优结果强加给不同派发
形态的 Prefill。

同一设备、同一模型、同一输入、各 7 个 Decode 的低扰动 A/B 结果：

| 指标 | 原始 Q4_K GEVM | Packed-byte 复用 | 变化 |
|---|---:|---:|---:|
| GPU 时间/token | 187.931 ms | 170.046 ms | -9.52% |
| 同步调用时间/token | 190.893 ms | 173.103 ms | -9.32% |
| Decode 速度 | 5.239 token/s | 5.777 token/s | +10.27% |

另一轮最终版本的 3 个 Decode 平均 GPU 时间为 170.255 ms，结果可复现。详细
kernel 采样中，受影响的 FFN 算子变化如下：

| 算子 | 原始时间 | 优化后时间 | 变化 |
|---|---:|---:|---:|
| FFN Gate Q4_K | 28.308 ms | 24.069 ms | -14.97% |
| FFN Up Q4_K | 28.784 ms | 24.624 ms | -14.45% |
| FFN Down Q4_K | 14.070 ms | 11.660 ms | -17.13% |
| 三项合计 | 71.163 ms | 60.352 ms | -15.19% |

未修改的 Q6_K Down 基本不变。0.5B CPU/GPU 对照以及 14B 原始/优化 A/B 均生成
相同 token 序列；重新分组的 FP32 累加顺序没有改变本次贪心生成结果。

原始和优化 A/B 数据分别位于：

```text
output/profile_14b_ab_baseline_command_metal_commands.csv
output/profile_14b_ab_q4_nibble_reuse_command_metal_commands.csv
```

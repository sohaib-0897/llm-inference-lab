# LLM Inference Lab: Comprehensive Benchmark Report

This document reports the empirical benchmarking results conducted across multiple optimization strategies within the `llm-inference-lab`.

---

## 1. Test Environment & Hardware Specification

- **Processor**: 13th Gen Intel(R) Core(TM) i5-13420H (8 physical cores, 12 logical threads, 2.10 GHz base)
- **Host RAM**: 15.65 GB Total (Available during runs: ~1.26 GB – 1.61 GB)
- **Available Disk Space**: ~2.85 GB free on C: drive
- **GPU Present**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, Driver 592.82, CUDA 13.1)
- **Execution Runtime**: PyTorch 2.14.0+cpu on Windows x64 (CPU execution provider)
- **ONNX Runtime**: version 1.30.0 (CPUExecutionProvider with 4 intra-op threads)
- **Benchmarked Architecture Preset**: `nano` (0.72M parameters, `d_model`=128, 3 layers, 4 query heads, 2 KV heads, GQA)

---

## 2. Benchmark Suite 1: Key-Value (KV) Cache Scaling

In autoregressive token generation, storing previous keys and values eliminates redundant attention computations over previous tokens.

### Empirical Measurements

| Scenario | Prompt Tokens | Generated Tokens | TTFT (ms) | Avg TPOT (ms) | P95 TPOT (ms) | Total Latency (ms) | Throughput (tok/s) | KV Cache Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KV Cache ON** | 32 | 16 | 7.59 | **5.16** | 9.84 | **85.85** | **186.38** | 0.070 MB |
| **KV Cache OFF** | 32 | 16 | 4.72 | 8.23 | 13.34 | 128.84 | 124.19 | 0.000 MB |
| **KV Cache ON** | 32 | 32 | 2.84 | **3.35** | 6.72 | **107.71** | **297.10** | 0.094 MB |
| **KV Cache OFF** | 32 | 32 | 29.09 | 6.09 | 10.40 | 219.29 | 145.93 | 0.000 MB |
| **KV Cache ON** | 32 | 64 | 3.63 | **4.28** | 8.13 | **275.09** | **232.65** | 0.141 MB |
| **KV Cache OFF** | 32 | 64 | 3.84 | 5.15 | 9.85 | 330.23 | 193.80 | 0.000 MB |

### Analysis
- **Complexity Shift**: Without KV caching, every step recomputes the full context from position 0, resulting in quadratic attention cost $O(N^2)$. With KV caching, only the newest query vector is computed against the cached key/value matrices, achieving $O(1)$ query complexity per step.
- **Speedup**: At 32 tokens, enabling KV cache yields a **2.03x total latency speedup** and increases throughput from 145.9 tok/s to 297.1 tok/s.
- **Memory Overhead**: The KV cache memory footprint is negligible for small models (0.14 MB for 96 tokens), offering high performance returns for minimal RAM consumption.

---

## 3. Benchmark Suite 2: Batch Size Scaling

Evaluates parallel execution efficiency as concurrent request batch size increases.

| Batch Size | Prompt Len | Gen Len | TTFT (ms) | Avg TPOT (ms) | Total Latency (ms) | Per-Stream Throughput | Aggregate Batch Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 32 | 32 | 4.11 | 2.96 | 96.98 | 329.95 tok/s | **329.95 tok/s** |
| **2** | 32 | 32 | 5.01 | 2.66 | 88.45 | 361.81 tok/s | **723.62 tok/s** |
| **4** | 32 | 32 | 5.79 | 3.27 | 108.32 | 295.42 tok/s | **1,181.68 tok/s** |

### Analysis
- **Throughput Scaling**: Scaling from batch size 1 to 4 increases aggregate generation throughput from 329.95 tok/s to **1,181.68 tok/s** (a **3.58x scaling factor**).
- **Latency Impact**: Total latency per request increases by only **11.6%** (from 96.98 ms to 108.32 ms) while serving 4x the request volume, demonstrating effective multi-core CPU matrix-multiplication parallelization.

---

## 4. Benchmark Suite 3: INT8 Weight Quantization

Evaluates symmetric per-channel INT8 weight-only quantization against the FP32 baseline.

| Precision Format | Weight Footprint (MB) | Compression Ratio | Generation Latency (ms) | Throughput (tok/s) |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 2.82 MB | 1.00x | **115.25 ms** | **277.65 tok/s** |
| **INT8 (Quantized)** | 1.14 MB | **2.46x** | 186.05 ms | 171.99 tok/s |

### Analysis
- **Memory Savings**: INT8 quantization compresses linear projection weights by 4x, reducing overall model memory footprint from 2.82 MB to 1.14 MB (a net 2.46x reduction including unquantized token embeddings).
- **Compute Trade-off**: On standard CPU architectures without dedicated hardware INT8 GEMM (e.g., AVX-512 VNNI / AMX), dequantizing INT8 weights to float32 on the fly incurs compute overhead (latency increased by ~61%). This highlights that weight-only quantization on CPU is primarily memory-bandwidth and RAM-constrained optimization.

---

## 5. Benchmark Suite 4: Speculative Decoding

Evaluates the dual-model draft-and-verify paradigm (Leviathan et al., 2023) using a draft model (`nano`, 0.72M) speculating $\gamma=3$ tokens for verification by a target model (`micro`, 5.48M).

| Configuration | Draft Model | Target Model | $\gamma$ (Draft Lookahead) | Draft Acceptance Rate | Generation Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Baseline** | N/A | `micro` (5.48M) | N/A | N/A | **133.19 tok/s** |
| **Speculative Decoding**| `nano` (0.72M) | `micro` (5.48M) | 3 tokens | 0.0% (untrained) | 36.95 tok/s |

### Analysis
- **Draft Alignment Prerequisite**: Speculative decoding efficiency is strictly bounded by the probability distribution match between draft and target models. With randomly initialized weights, draft acceptance rate is ~0%, requiring fallback to target model correction on every verification cycle.
- **Empirical Takeaway**: When draft acceptance is low ($< 50\%$), the overhead of running both models sequentially exceeds the single-model baseline. Trained distilled draft models with $>70\%$ acceptance are essential for speculative decoding speedup.

---

## 6. Benchmark Suite 5: PyTorch CPU vs ONNX Runtime CPU

Compares forward pass latency of native PyTorch CPU execution against graph-optimized ONNX Runtime CPU execution.

| Inference Engine | Input Shape (`batch`, `seq_len`) | Mean Forward Latency (ms) | P95 Latency (ms) | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch CPU (2.14.0+cpu)** | `(1, 32)` | 7.24 ms | 24.92 ms | 1.00x |
| **ONNX Runtime (CPUExecutionProvider)** | `(1, 32)` | **1.27 ms** | **4.03 ms** | **5.72x** |

### Analysis
- **Graph Optimization**: ONNX Runtime applies constant folding, operator fusion (combining linear projections and activation functions), and cache-friendly GEMM memory tiling.
- **Speedup**: ONNX Runtime achieves an empirical **5.72x speedup** over native PyTorch on CPU, reducing single-pass latency from 7.24 ms to 1.27 ms.

---

## 7. Hardware Detection & Constraints

1. **CPU Only Execution**:
   - The detected PyTorch binary installed in the Python environment is `torch==2.14.0+cpu`.
   - Although an NVIDIA RTX 4050 Laptop GPU is physically installed with driver version 592.82 and CUDA 13.1, PyTorch CUDA extensions were not present in the user-space environment. In accordance with execution rules (no sudo/admin, no installing global packages, conservative disk usage), execution proceeded safely on CPU without failure or interactive interruption.
2. **Disk & RAM Safety Constraints**:
   - Host machine available disk space was measured at **2.85 GB**, and available RAM was measured at **1.26 GB – 1.61 GB**.
   - Model presets (`nano`, `micro`, `small`), virtual environment structure (`--system-site-packages`), and benchmark batch sizes were tuned to prevent memory exhaustion (OOM) or disk overflow.
   - Temporary exported ONNX models were cleaned up automatically after benchmarking.

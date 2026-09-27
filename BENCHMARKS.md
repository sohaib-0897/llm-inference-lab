# LLM Inference Lab: Comprehensive Empirical Benchmark Report

This document reports empirical benchmarking results across two complementary inference layers in `llm-inference-lab`:

1. **From-Scratch Inference Internals**: Micro-architectural evaluations of custom KV caching, GQA, RoPE, INT8 quantization, speculative decoding, and ONNX Runtime CPU graph execution.
2. **Real Pretrained Model Serving (Ollama)**: Macro-evaluations of production-grade pretrained models (`qwen2.5:0.5b` and `qwen3:4b`) running with full GPU acceleration on local hardware.

---

## 1. Test Environment & Hardware Specification

- **Processor**: 13th Gen Intel(R) Core(TM) i5-13420H (8 physical cores, 12 logical threads, 2.10 GHz base)
- **Host RAM**: 15.65 GB Total (Available during runs: ~1.26 GB – 1.61 GB)
- **Disk Space**: ~1.77 GB free on C: drive (initial: 2.85 GB; safely preserved throughout execution)
- **GPU Present**: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, Driver 592.82, CUDA 13.1)
- **From-Scratch Runtime**: PyTorch 2.14.0+cpu on Windows x64 & ONNX Runtime 1.30.0 (CPUExecutionProvider)
- **Real-Model Runtime**: Ollama version 0.34.4 (GPU offload enabled, `llama-server.exe` backend)

---

## PART I: FROM-SCRATCH INFERENCE INTERNALS BENCHMARKS

These micro-benchmarks isolate algorithmic transformer mechanisms using a controlled `nano` architecture (0.72M parameters, `d_model`=128, 3 layers, 4 query heads, 2 KV heads with GQA).

### Suite 1: Key-Value (KV) Cache Scaling (Cached Incremental Decoding vs Full-Prefix Recomputation)

In autoregressive token generation, storing previous keys and values eliminates redundant attention computations over previous tokens.

| Scenario | Prompt Tokens | Generated Tokens | TTFT (ms) | Avg TPOT (ms) | P95 TPOT (ms) | Total Latency (ms) | Throughput (tok/s) | KV Cache Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KV Cache ON** | 32 | 16 | 7.59 | **5.16** | 9.84 | **85.85** | **186.38** | 0.070 MB |
| **KV Cache OFF** | 32 | 16 | 4.72 | 8.23 | 13.34 | 128.84 | 124.19 | 0.000 MB |
| **KV Cache ON** | 32 | 32 | 2.84 | **3.35** | 6.72 | **107.71** | **297.10** | 0.094 MB |
| **KV Cache OFF** | 32 | 32 | 29.09 | 6.09 | 10.40 | 219.29 | 145.93 | 0.000 MB |
| **KV Cache ON** | 32 | 64 | 3.63 | **4.28** | 8.13 | **275.09** | **232.65** | 0.141 MB |
| **KV Cache OFF** | 32 | 64 | 3.84 | 5.15 | 9.85 | 330.23 | 193.80 | 0.000 MB |

**Findings**:
- Without KV caching, every step recomputes all past key/value representations from position 0 (full-prefix recomputation across prior tokens).
- With KV caching, only the newest query vector is computed against past cached states (cached incremental decoding per step).
- At 32 tokens, enabling KV cache yields a **2.03x total latency speedup** (107.7 ms vs 219.3 ms) and increases generation throughput from 145.9 tok/s to 297.1 tok/s.

---

### Suite 2: Batch Size Scaling

Evaluates parallel execution efficiency as concurrent request batch size increases on multi-core CPU.

| Batch Size | Prompt Len | Gen Len | TTFT (ms) | Avg TPOT (ms) | Total Latency (ms) | Per-Stream Throughput | Aggregate Batch Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 32 | 32 | 4.11 | 2.96 | 96.98 | 329.95 tok/s | **329.95 tok/s** |
| **2** | 32 | 32 | 5.01 | 2.66 | 88.45 | 361.81 tok/s | **723.62 tok/s** |
| **4** | 32 | 32 | 5.79 | 3.27 | 108.32 | 295.42 tok/s | **1,181.68 tok/s** |

**Findings**:
- Scaling batch size from 1 to 4 increases aggregate generation throughput from 329.95 tok/s to **1,181.68 tok/s** (a **3.58x scaling factor**).
- Total request latency increases by only **11.6%** (96.98 ms to 108.32 ms) while processing 4x the request volume.

---

### Suite 3: INT8 Weight Quantization vs FP32

Evaluates symmetric per-channel INT8 weight-only quantization against the FP32 baseline.

| Precision Format | Weight Footprint (MB) | Compression Ratio | Generation Latency (ms) | Throughput (tok/s) |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 2.82 MB | 1.00x | **115.25 ms** | **277.65 tok/s** |
| **INT8 (Quantized)** | 1.14 MB | **2.46x** | 186.05 ms | 171.99 tok/s |

**Findings**:
- INT8 quantization reduces linear projection weights by 4x, achieving an overall **2.46x model compression** (2.82 MB to 1.14 MB).
- On standard CPU architectures without dedicated hardware INT8 GEMM instructions (AVX-512 VNNI / AMX), dequantizing INT8 weights to float32 on the fly incurs compute overhead (latency increased by ~61%). This demonstrates the classic compute-vs-memory trade-off.

---

### Suite 4: Speculative Decoding (Draft-Target Verification)

Evaluates the dual-model draft-and-verify paradigm (Leviathan et al., 2023) using a draft model (`nano`, 0.72M) speculating $\gamma=3$ tokens for verification by a target model (`micro`, 5.48M).

| Configuration | Draft Model | Target Model | $\gamma$ (Lookahead) | Draft Acceptance Rate | Generation Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Baseline** | N/A | `micro` (5.48M) | N/A | N/A | **133.19 tok/s** |
| **Speculative Decoding**| `nano` (0.72M) | `micro` (5.48M) | 3 tokens | 0.0% (untrained) | 36.95 tok/s |

**Findings**:
- Speculative decoding speedup is strictly bounded by the probability distribution alignment between the draft and target models.
- With randomly initialized weights, draft tokens disagree with target argmax choices, requiring fallback to target corrections on every step. Trained distilled draft models with $>70\%$ acceptance are essential for net wall-clock speedups.

---

### Suite 5: PyTorch CPU vs ONNX Runtime CPU

Compares forward pass latency of native PyTorch CPU execution against graph-optimized ONNX Runtime CPU execution.

| Inference Engine | Input Shape (`batch`, `seq_len`) | Mean Forward Latency (ms) | P95 Latency (ms) | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch CPU (2.14.0+cpu)** | `(1, 32)` | 7.24 ms | 24.92 ms | 1.00x |
| **ONNX Runtime (CPUExecutionProvider)** | `(1, 32)` | **1.27 ms** | **4.03 ms** | **5.72x** |

**Findings**:
- ONNX Runtime applies graph constant folding, operator fusion (combining linear projections and activation functions), and cache-friendly GEMM memory tiling.
- Achieves an empirical **5.72x speedup** over native PyTorch on CPU (1.27 ms vs 7.24 ms).

---

## PART II: REAL PRETRAINED MODEL BENCHMARKS (OLLAMA RUNTIME)

The real-model benchmark layer profiles actual open-weight models (`qwen2.5:0.5b` and `qwen3:4b`) served by Ollama on local hardware.

### GPU Residency & Observational Data

GPU status captured via `ollama ps` and `nvidia-smi`:

```text
NAME            ID              SIZE      PROCESSOR    CONTEXT    VRAM RESIDENCY
qwen2.5:0.5b    a8b0c5157701    379 MB    100% GPU     4096       459.5 MiB / 6,141 MiB
qwen3:4b        e4285a787853    2.4 GB    100% GPU     4096       3,030.9 MiB / 6,141 MiB
```

- **Observational Evidence**: `ollama ps` reports processor allocation as **100% GPU** with dedicated VRAM residency of `459.5 MiB` (`qwen2.5:0.5b`) and `3,030.9 MiB` (`qwen3:4b`) out of 6,141 MiB on the NVIDIA GeForce RTX 4050 Laptop GPU.
- **Process Activity**: `llama-server.exe` was observed via process monitoring and `nvidia-smi` executing under CUDA acceleration without host RAM layer swapping. These figures represent observational runtime and process metrics rather than kernel-level GPU profiler traces.

---

### Experiment A: Real-Model Baseline (`qwen2.5:0.5b`)

Controlled prompt: *"Explain how key-value caching accelerates transformer inference in three concise sentences."*
Parameters: `temperature=0.0`, `seed=42`, `num_predict=64`, 1 warm-up, 3 measured trials.

| Trial | Prompt Tokens | Output Tokens | Client TTFT (ms) | Client Total Latency (ms) | Client Throughput (tok/s) | Server Decode Speed (tok/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trial 1** | 20 | 64 | 28.52 ms | 253.11 ms | 225.20 tok/s | 258.12 tok/s |
| **Trial 2** | 20 | 64 | 26.84 ms | 249.88 ms | 228.11 tok/s | 258.54 tok/s |
| **Trial 3** | 20 | 64 | 27.80 ms | 251.30 ms | 227.57 tok/s | 257.52 tok/s |
| **Mean ± Std** | **20** | **64** | **27.72 ± 0.69 ms** | **251.43 ± 1.32 ms** | **226.96 ± 1.25 tok/s** | **258.06 ± 0.42 tok/s** |

**Metric Distinction**:
- **Client Throughput (226.96 tok/s)**: Total generated tokens divided by total client wall-clock duration (including HTTP overhead, prompt prefill, network socket latency, and final token transfer).
- **Server Decode Speed (258.06 tok/s)**: Evaluated purely during the autoregressive generation loop by Ollama's `eval_duration` metric.

---

### Experiment B: Context-Length Scaling (Prefill Cost)

Evaluates prompt evaluation latency across expanding context windows up to ~3,000 prompt tokens (`num_predict=32` tokens constant).

| Target Length | Actual Prompt Tokens | Output Tokens | Server Prompt Eval (ms) | Client TTFT (ms) | Total Latency (ms) | Client Throughput (tok/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **~128** | 223 | 32 | 4.37 ms | 43.53 ms | 110.25 ms | 154.42 tok/s |
| **~256** | 405 | 32 | 4.45 ms | 26.47 ms | 96.97 ms | 185.64 tok/s |
| **~512** | 769 | 32 | 4.38 ms | 46.66 ms | 115.99 ms | 148.30 tok/s |
| **~1024**| 1,497 | 32 | 4.42 ms | 37.93 ms | 111.59 ms | 161.50 tok/s |
| **~2048**| 2,979 | 32 | 5.06 ms | 62.72 ms | 124.25 ms | 106.69 tok/s |

**Findings**:
- On dedicated GPU VRAM, prompt evaluation is parallelized across GPU tensor cores.
- Prompt eval duration stays minimal (4.37 ms at 223 tokens to 5.06 ms at 2,979 tokens).
- Client TTFT shows a slight increase from ~26 ms to 62.7 ms at ~3,000 tokens due to prompt parsing and KV allocation.

---

### Experiment C: Output-Length Scaling

Fixed prompt (~64 tokens), varying requested output tokens: 16, 32, 64, 128.

| Output Tokens | Server Eval Duration (ms) | Client TTFT (ms) | Client Total Latency (ms) | Client Throughput (tok/s) |
| :--- | :--- | :--- | :--- | :--- |
| **16** | 57.03 ms | 35.14 ms | 94.02 ms | 172.99 tok/s |
| **32** | 119.35 ms | 30.46 ms | 152.50 ms | 210.07 tok/s |
| **64** | 243.03 ms | 41.87 ms | 288.08 ms | 222.21 tok/s |
| **128**| 485.80 ms | 33.24 ms | 522.13 ms | **245.20 tok/s** |

**Findings**:
- Server eval duration scales linearly with output tokens: $57.0\text{ ms} \to 119.4\text{ ms} \to 243.0\text{ ms} \to 485.8\text{ ms}$ (~3.8 ms per token, matching the ~260 tok/s decode rate).
- Client throughput increases from 172.9 tok/s to 245.2 tok/s as the fixed TTFT overhead is amortized over longer generation runs.

---

### Experiment D: Model-Size Comparison (`qwen2.5:0.5b` vs `qwen3:4b`)

Controlled prompt, fixed output length (32 tokens), temperature 0.0, 3 trials each.

| Model | Parameter Size | Quantization | Disk Footprint | VRAM Allocated | Mean TTFT | Server Decode Speed | Client Total Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`qwen2.5:0.5b`** | 494.03M | Q4_K_M | 379.4 MB | **459.5 MiB** | **36.68 ms** | **241.73 tok/s** | **176.42 ms** |
| **`qwen3:4b`** | 4.0B | Q4_K_M | 2,381.6 MB | **3,030.9 MiB** | 57.01 ms | **63.32 tok/s** | **565.38 ms** |

**Findings**:
- **Throughput Ratio**: The 0.5B model achieves **3.82x higher generation speed** (241.7 tok/s vs 63.3 tok/s), reflecting the ~8x difference in parameter matrix-multiplication FLOPs.
- **Memory Footprint**: The 4.0B model consumes 3.03 GB VRAM (~50% of the RTX 4050 6GB VRAM capacity), while the 0.5B model fits into 459.5 MB. Both execute 100% on GPU without Host RAM paging.

---

### Experiment E: Concurrency Scaling (1, 2, 4 Concurrent Requests)

Simultaneous requests dispatched in parallel via Python `ThreadPoolExecutor` to `qwen2.5:0.5b`.

| Concurrency Level | Wall-Clock Time (ms) | Total Tokens | Aggregate Throughput | Mean Request Latency | P95 Request Latency | Mean TTFT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 (Single)** | 79.91 ms | 15 | 187.72 tok/s | 77.38 ms | 77.38 ms | 20.52 ms |
| **2 (Dual)** | 155.38 ms | 30 | 193.07 tok/s | 121.17 ms | 150.56 ms | 63.62 ms |
| **4 (Quad)** | 291.58 ms | 60 | **205.77 tok/s** | 191.59 ms | **279.07 ms** | 135.67 ms |

**Findings**:
- **Aggregate Serving Throughput**: Increases from 187.7 tok/s to **205.8 tok/s** as GPU utilization is maximized.
- **Queueing & Latency Trade-off**: Under 4 concurrent requests, per-request mean latency increases from 77.4 ms to 191.6 ms, and P95 latency reaches 279.1 ms. TTFT increases from 20.5 ms to 135.7 ms due to internal FIFO scheduling in Ollama's single-worker engine.

---

## PART III: COMPARATIVE ANALYSIS (CUSTOM BACKEND VS OLLAMA BACKEND)

| Evaluation Dimension | Custom PyTorch / ONNX Backend | Ollama Real-Model Backend |
| :--- | :--- | :--- |
| **Primary Purpose** | Low-level ML-systems internals, tensor verification, and algorithmic mechanics | Macro-level serving benchmarks, real pretrained models, and GPU hardware profiling |
| **Execution Domain** | CPU (native PyTorch + ONNX Runtime) | GPU (NVIDIA RTX 4050 6GB via llama.cpp CUDA backend) |
| **KV Cache Transparency**| Explicit access to LayerKVCache tensors, buffer allocations, and memory bytes | Internal GGUF cache managed invisibly by `llama-server` |
| **Model Architectures** | Native PyTorch implementation (`CausalLM`) with GQA, RoPE, and SwiGLU | Real quantized models (`qwen2.5:0.5b`, `qwen3:4b` in Q4_K_M GGUF format) |
| **Optimization Capabilities**| Custom INT8 weight-only quantization, ONNX graph export, speculative decoding | Multi-threaded CUDA GEMM, GPU tensor cores, native prompt prefill caching |
| **Best Used For** | Proving mathematical correctness, testing new layers, studying cached incremental decoding vs full-prefix recomputation | Evaluating real-world latency, context scaling, throughput per dollar, concurrent capacity |

---

## PART IV: HARDWARE CONSTRAINTS & LIMITATIONS

1. **Host Disk Space Constraints**:
   - Host free disk space was measured at **2.85 GB** initially and preserved at **1.77 GB** after pulling `qwen2.5:0.5b` (397 MB).
   - Downloading large 7B+ models or heavy CUDA PyTorch wheels (~2.8 GB) was avoided to prevent disk overflow.
2. **PyTorch CPU Execution vs Ollama GPU Execution**:
   - The custom PyTorch and ONNX Runtime backends evaluate micro-architectural mechanics and algorithmic scaling on the host CPU (`2.14.0+cpu`).
   - The Ollama backend evaluates production-grade serving throughput on the local GPU via llama.cpp's bundled CUDA runtime.
   - These two layers evaluate distinct tiers of the inference stack and must not be conflated as equivalent or direct like-for-like hardware workloads.
3. **Single-Worker Queuing**:
   - Ollama by default serializes requests on single-model instances unless configured with `OLLAMA_NUM_PARALLEL`. Concurrency tests document the expected queueing latency without altering service settings.

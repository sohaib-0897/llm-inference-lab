# LLM Inference Lab

A lightweight, high-performance experimentation and benchmarking laboratory for Large Language Model (LLM) inference optimization.

Designed to profile, evaluate, and compare modern inference strategies including **Key-Value (KV) Caching**, **Grouped-Query Attention (GQA)**, **Rotary Positional Embeddings (RoPE)**, **Dynamic INT8 Quantization**, **Speculative Decoding**, and **ONNX Runtime Engine Acceleration**.

---

## Key Features

- **Modern Decoder-Only Causal Architecture**:
  - Implements Root Mean Square Normalization ([`RMSNorm`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/models/transformer.py#L17)).
  - Rotary Position Embeddings ([`apply_rotary_emb`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/models/attention.py#L32)) supporting dynamic sequence position offsets.
  - Multi-Head and Grouped-Query Attention ([`GroupedQueryAttention`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/models/attention.py#L70)) for memory-efficient key/value representation.
  - SwiGLU Feed-Forward Networks ([`SwiGLUFeedForward`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/models/transformer.py#L29)).
- **Key-Value Caching Engine**:
  - Layer-wise cache management ([`KVCache`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/engine/kv_cache.py#L48)) separating prefill ($O(N)$ prompt processing) from autoregressive decode ($O(1)$ single-step query).
  - Theoretical vs. empirical cache memory tracking.
  - Full mathematical parity verification between cached and non-cached decoding.
- **Inference Engines & Acceleration**:
  - **PyTorch Native Engine**: Streaming and batch generation with configurable sampling (Greedy, Top-K, Top-P Nucleus, Temperature, Repetition Penalty).
  - **ONNX Runtime Engine**: Export pipeline ([`export_to_onnx`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/engine/onnx_engine.py#L33)) and execution provider harness ([`ONNXInferenceEngine`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/engine/onnx_engine.py#L65)), delivering a **5.72x speedup** on CPU.
- **Quantization Laboratory**:
  - Symmetric per-channel INT8 weight quantization ([`QuantizedLinear`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/quantization/quantized_linear.py#L11)) achieving **2.46x parameter memory compression**.
- **Speculative Decoding**:
  - Draft-target model verification loop ([`SpeculativeDecoder`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/engine/speculative.py#L13)) based on Leviathan et al. (2023) tracking empirical draft acceptance rate.
- **Benchmarking & Visualization**:
  - Statistical profiling harness ([`BenchmarkRunner`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/benchmarks/runner.py#L19)) exporting JSON, CSV, and visual performance graphs.

---

## Architecture Presets

| Preset | Parameters | Hidden Dim (`d_model`) | Layers | Heads | KV Heads | Context Window |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`nano`** | 0.72M | 128 | 3 | 4 | 2 | 512 |
| **`micro`** | 5.48M | 256 | 6 | 8 | 4 | 1,024 |
| **`small`** | 26.75M | 512 | 8 | 16 | 4 | 2,048 |

---

## Installation & Setup

The project runs in a local user-space virtual environment (`.venv`) with zero external API requirements:

```bash
# Clone or navigate to the repository
cd llm-inference-lab

# Create and activate local virtual environment
python -m venv --system-site-packages .venv
.venv\Scripts\activate   # On Windows PowerShell: .venv\Scripts\Activate.ps1

# Install development dependencies or run directly with existing packages
pip install -e .
```

---

## Command-Line Interface (CLI)

The CLI tool `llm_lab.cli` provides subcommands for inspection, generation, benchmarking, and visualization:

### 1. Hardware & System Inspection
Inspect CPU cores, RAM availability, CUDA status, and model architectural presets:
```bash
python -m llm_lab.cli info
```

### 2. Interactive Token Generation
Generate tokens with real-time streaming latency profiling (TTFT, TPOT, and throughput):
```bash
python -m llm_lab.cli generate --preset nano --prompt-len 16 --max-tokens 32
```
Options:
- `--preset`: `nano`, `micro`, or `small`
- `--no-cache`: Disables KV caching to test quadratic recomputation latency
- `--sample`: Enables stochastic sampling (supports `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`)

### 3. Run Automated Benchmark Suite
Run the 5-suite comprehensive benchmark harness:
```bash
python -m llm_lab.cli benchmark --preset nano --warmup 1 --trials 3 --output-dir benchmarks/results
```

### 4. Generate Visual Charts
Export publication-ready comparison charts from existing results:
```bash
python -m llm_lab.cli plot --results-dir benchmarks/results
```

---

## Running Tests, Linting, & Type Checking

All verification tools run autonomously:

```bash
# Run Unit Test Suite (16/16 tests passing)
pytest

# Code Formatting & Linting Check
ruff check .

# Static Type Checking
mypy llm_lab tests
```

---

## Benchmark Results Summary

Measured on an **Intel Core i5-13420H (8 physical / 12 logical cores)**:

1. **KV Cache Speedup**:
   - At 64 generated tokens, KV caching yields **232.7 tok/s** compared to **193.8 tok/s** without cache, with average decode step latency remaining constant ($O(1)$) rather than scaling quadratically.
2. **Batch Scaling**:
   - Increasing batch size from 1 to 4 scales aggregate throughput from **329.9 tok/s** to **1,181.7 tok/s** (3.58x parallel scaling).
3. **INT8 Quantization**:
   - Reduces weight footprint from **2.82 MB** (FP32) to **1.14 MB** (INT8), yielding a **2.46x compression ratio**.
4. **ONNX Runtime Acceleration**:
   - Single forward pass latency drops from **7.24 ms** (PyTorch native CPU) to **1.27 ms** (ONNX Runtime CPU), delivering a **5.72x speedup**.

For complete numerical tables, latency percentiles, and charts, refer to [BENCHMARKS.md](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/BENCHMARKS.md).

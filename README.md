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
  - Layer-wise cache management ([`KVCache`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/engine/kv_cache.py#L48)) separating prefill prompt processing from cached incremental single-token decoding.
  - Theoretical vs. empirical cache memory tracking.
  - Full mathematical parity verification between cached and non-cached decoding.
- **Inference Engines & Acceleration**:
  - **PyTorch Native Engine**: Streaming and batch generation with configurable sampling (Greedy, Top-K, Top-P Nucleus, Temperature, Repetition Penalty).
  - **ONNX Runtime Engine**: Export pipeline ([`export_to_onnx`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/engine/onnx_engine.py#L33)) and execution provider harness ([`ONNXInferenceEngine`](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/llm_lab/engine/onnx_engine.py#L65)), demonstrating a **5.7x forward speedup** on CPU.
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

### 3. Run Automated Benchmark Suite (From-Scratch Internals)
Run the 5-suite comprehensive benchmark harness:
```bash
python -m llm_lab.cli benchmark --preset nano --warmup 1 --trials 3 --output-dir benchmarks/results
```

### 4. Generate Visual Charts (From-Scratch Internals)
Export publication-ready comparison charts from existing results:
```bash
python -m llm_lab.cli plot --results-dir benchmarks/results
```

### 5. Inspect Local Ollama Models & GPU Memory Offload
```bash
python -m llm_lab.cli ollama-info
```

### 6. Interactive Generation with Real Pretrained Models (Ollama)
```bash
python -m llm_lab.cli ollama-gen --model qwen2.5:0.5b --max-tokens 48
```

### 7. Run Real-Model Benchmark Suite (Experiments A–E)
```bash
python -m llm_lab.cli ollama-bench --model qwen2.5:0.5b --output-dir benchmarks/results/ollama
```

---

## Real-Model Benchmarks with Ollama

While the custom PyTorch engine isolates low-level mechanisms (KV cache dynamics, GQA, RoPE, and INT8 compression) on CPU, the **Ollama Real-Model Benchmark Layer** evaluates production-grade pretrained LLMs running on actual local hardware with full GPU acceleration (NVIDIA RTX 4050 Laptop GPU, 6GB VRAM via Ollama's bundled CUDA runtime).

### Two Complementary Layers

| Layer | Implementation | Focus & Capabilities |
| :--- | :--- | :--- |
| **From-Scratch Internals** | Custom PyTorch & ONNX | • Explicit KV cache tensor updates and memory tracking<br>• Mathematical equivalence validation (cached incremental decoding vs full-prefix recomputation)<br>• Speculative decoding draft/target verification loops<br>• Custom INT8 weight-only quantization<br>• PyTorch vs ONNX Runtime graph execution |
| **Real Pretrained Serving** | Local Ollama Runtime | • Pretrained models (`qwen2.5:0.5b`, `qwen3:4b`)<br>• Full GPU VRAM offloading (100% GPU via llama-server)<br>• Realistic prompt prefill scaling up to 3,000 tokens<br>• Concurrent request serving & queue latency profiling<br>• Model size & throughput trade-offs |

### Strongest Measured Ollama Results

Measured on **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)** with Ollama 0.34.4:

| Benchmark Experiment | Model | Key Measurement | Observed Behavior |
| :--- | :--- | :--- | :--- |
| **Model Size Trade-off** | `qwen2.5:0.5b` vs `qwen3:4b` | **241.7 tok/s** vs **63.3 tok/s** | 3.82x throughput advantage for 0.5B; 4B requires 3.03 GB VRAM |
| **Output Scaling** | `qwen2.5:0.5b` (16 to 128 tok) | **172.99 tok/s** → **245.20 tok/s** | Sustained high throughput during continuous token generation |
| **Context Prefill Scaling** | `qwen2.5:0.5b` (223 to 2,979 tok) | Prompt Eval: **4.37 ms** → **5.06 ms** | GPU tensor cores process large prompt blocks with low prefill latency |
| **Concurrency Scaling** | `qwen2.5:0.5b` (1 to 4 concurrent) | Aggregate: **187.7 tok/s** → **205.8 tok/s** | P95 latency scales from 77.4 ms to 279.1 ms as requests queue |

Artifacts and charts:
- Data exports: [benchmarks/results/ollama/](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/benchmarks/results/ollama/)
- Visual plots: [context_scaling.png](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/benchmarks/results/ollama/context_scaling.png), [output_scaling.png](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/benchmarks/results/ollama/output_scaling.png), [model_comparison.png](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/benchmarks/results/ollama/model_comparison.png), [concurrency.png](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/benchmarks/results/ollama/concurrency.png)

---

## Running Tests, Linting, & Type Checking

All verification tools run autonomously:

```bash
# Run Complete Unit & Integration Test Suite (24/24 tests passing)
pytest

# Code Formatting & Linting Check
ruff check .

# Static Type Checking
mypy llm_lab tests
```

---

## Benchmark Results Summary

### 1. From-Scratch Internals (Intel Core i5-13420H CPU)
- **KV Cache Speedup**: At 64 generated tokens, KV caching yields **232.7 tok/s** compared to **193.8 tok/s** without cache, maintaining cached incremental decoding rather than full-prefix recomputation.
- **Batch Scaling**: Increasing batch size from 1 to 4 scales aggregate throughput from **329.9 tok/s** to **1,181.7 tok/s** (3.58x parallel scaling).
- **INT8 Quantization**: Reduces weight footprint from **2.82 MB** (FP32) to **1.14 MB** (INT8), yielding a **2.46x compression ratio**.
- **ONNX Runtime Acceleration**: Single forward pass latency drops from **7.24 ms** (PyTorch native CPU) to **1.27 ms** (ONNX Runtime CPU), delivering a **5.7x forward speedup**.

### 2. Real-Model Serving (NVIDIA RTX 4050 Laptop GPU via Ollama)
- **Model Size Scaling**: `qwen2.5:0.5b` delivers **241.7 tok/s** decode speed vs **63.3 tok/s** for `qwen3:4b` (3.82x throughput advantage).
- **Output Sustained Speed**: Generation throughput scales from **173.0 tok/s** (16 tokens) to **245.2 tok/s** (128 tokens).
- **Prefill Scalability**: Evaluating 2,979 prompt tokens adds only ~0.69 ms of prompt eval time on GPU tensor cores.
- **Concurrent Capacity**: Aggregate throughput peaks at **205.8 tok/s** under 4 concurrent requests.

For full numerical breakdowns, latency percentiles, and hardware constraints, refer to [BENCHMARKS.md](file:///C:/Users/Sohaib/Desktop/llm-inference-lab/BENCHMARKS.md).

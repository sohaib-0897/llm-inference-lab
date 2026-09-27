# LLM Inference Lab

A reproducible systems project for studying how large language model inference behaves under different execution strategies, model sizes, context lengths, output lengths, and serving conditions.

The project combines:

- a custom decoder-only transformer implementation
- KV-cached vs full-prefix decoding
- batching and context-length experiments
- INT8 weight-only quantization
- ONNX Runtime comparisons
- speculative decoding mechanics
- real-model inference through Ollama
- local GPU serving measurements
- an interactive benchmark visualizer

The goal is not to build another chat application.

The goal is to measure and understand what actually happens during inference.

---

## What this project studies

### Decoder performance

The custom transformer implementation is used to examine:

- incremental decoding
- KV-cache reuse
- full-prefix recomputation
- token generation latency
- throughput
- batching behavior
- output-length scaling
- context-length scaling

### Memory and quantization

The project compares FP32 weights with a custom per-channel INT8 weight-only representation.

Measured example:

| Representation | Model size | Latency | Throughput |
|---|---:|---:|---:|
| FP32 | 2.82 MB | 115.25 ms | 277.65 tok/s |
| INT8 | 1.14 MB | 186.05 ms | 171.99 tok/s |

The INT8 representation reduced model footprint by approximately **2.46×**, but the measured CPU implementation was slower.

This is intentionally documented as a tradeoff rather than assuming quantization automatically improves latency.

---

## KV caching

One of the main experiments compares cached incremental decoding with repeated full-prefix recomputation.

| Strategy | Generation time | Throughput |
|---|---:|---:|
| KV-cached decoding | 107.71 ms | 297.10 tok/s |
| Full recomputation | 219.29 ms | 145.93 tok/s |

In this experiment, KV caching reduced measured generation time by approximately **2×**.

The important distinction is that cached decoding reuses previously computed key/value states instead of repeatedly processing the entire prefix.

---

## ONNX Runtime

The project also compares the custom PyTorch CPU path with ONNX Runtime for the same small-model workload.

| Runtime | Mean latency |
|---|---:|
| PyTorch CPU | 7.24 ms |
| ONNX Runtime | 1.27 ms |

Measured speedup:

**5.72×**

These numbers apply to the benchmark configuration in this repository and should not be interpreted as a universal PyTorch-vs-ONNX result.

---

# Real-model inference

Synthetic experiments are useful for understanding mechanics, but the project also includes measurements from real pretrained models served through Ollama.

Tested models:

- `qwen2.5:0.5b`
- `qwen3:4b`

Both were run locally using Q4_K_M quantization.

The machine reported full GPU processor allocation through Ollama during the measured runs.

---

## Local hardware

Real-model experiments were performed on:

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
- **VRAM:** 6141 MiB
- **CPU:** Intel Core i5-13420H
- **System RAM:** approximately 15.6 GB
- **Ollama:** 0.34.4

Observed GPU residency:

| Model | Approx. VRAM residency |
|---|---:|
| Qwen2.5 0.5B | 459.5 MiB |
| Qwen3 4B | 3030.9 MiB |

The larger model used roughly **6.6×** more observed VRAM.

---

# Real-model benchmark results

## Qwen2.5 0.5B baseline

Representative measurements:

- TTFT: **27.72 ± 0.69 ms**
- total latency: **251.43 ± 1.32 ms**
- client-observed throughput: **226.96 ± 1.25 tok/s**
- Ollama-reported decode throughput: **258.06 ± 0.42 tok/s**

Client-observed timing and server-reported timing are kept separate throughout the benchmark outputs.

---

## Context-length scaling

Prompt size was increased while generated output was held constant.

Tested prompt lengths:

- 223 tokens
- 405 tokens
- 769 tokens
- 1497 tokens
- 2979 tokens

Observed prompt-evaluation times remained relatively small across most of the tested range, while TTFT increased for the largest context.

The raw results are preserved in the benchmark artifacts rather than summarized only through charts.

---

## Output-length scaling

Generated output was varied across:

- 16 tokens
- 32 tokens
- 64 tokens
- 128 tokens

Observed server evaluation time increased approximately with generated sequence length.

Representative server evaluation times:

| Output tokens | Server evaluation |
|---:|---:|
| 16 | 57.03 ms |
| 32 | 119.35 ms |
| 64 | 243.03 ms |
| 128 | 485.80 ms |

This makes output length one of the clearest drivers of total generation time in the measured configuration.

---

## Model-size comparison

With a 32-token output:

| Metric | Qwen2.5 0.5B | Qwen3 4B |
|---|---:|---:|
| TTFT | 36.68 ms | 57.01 ms |
| Server decode | 241.73 tok/s | 63.32 tok/s |
| Total latency | 176.42 ms | 565.38 ms |
| Observed VRAM | 459.5 MiB | 3030.9 MiB |

In this benchmark, the smaller model produced tokens approximately **3.82× faster** during generation.

This is a serving-performance comparison only. It is not a model-quality comparison.

---

# Concurrency

The Ollama benchmark also measures multiple simultaneous requests.

| Concurrency | Aggregate throughput | Mean latency | P95 latency |
|---:|---:|---:|---:|
| 1 | 187.72 tok/s | 77.38 ms | 77.38 ms |
| 2 | 193.07 tok/s | 121.17 ms | 150.56 ms |
| 4 | 205.77 tok/s | 191.59 ms | 279.07 ms |

The measured result shows a classic serving tradeoff:

aggregate throughput increased modestly while request latency, particularly tail latency, increased substantially.

---

# Architecture

The repository contains two complementary benchmark paths.

## 1. Custom inference engine

Used for controlled systems experiments.

Includes:

- decoder-only transformer
- rotary positional embeddings
- grouped-query attention
- RMSNorm
- SwiGLU
- KV caching
- streaming generation
- batching
- speculative decoding mechanics
- INT8 weight-only quantization
- ONNX export/runtime path

This path uses small custom models so individual inference mechanisms can be isolated and measured.

## 2. Ollama real-model benchmarks

Used for observing actual pretrained-model serving behavior.

Measures:

- TTFT
- prompt evaluation
- decode throughput
- total latency
- context-length scaling
- output-length scaling
- model-size differences
- concurrency
- observed GPU residency

The two benchmark paths are reported separately because they measure different systems.

---

# Interactive visualizer

The repository also includes a frontend under:

```text
frontend/

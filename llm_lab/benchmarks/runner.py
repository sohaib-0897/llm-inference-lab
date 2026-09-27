from __future__ import annotations

import csv
import json
import os
import time
from typing import List

import torch

from llm_lab.benchmarks.metrics import RunMetrics, compute_metrics
from llm_lab.config import BenchmarkConfig, GenerationConfig, ModelConfig
from llm_lab.engine.generation import generate, stream_generate
from llm_lab.engine.onnx_engine import ONNXInferenceEngine, export_to_onnx
from llm_lab.engine.speculative import SpeculativeDecoder
from llm_lab.models.transformer import CausalLM
from llm_lab.quantization.quantized_linear import calculate_model_weight_bytes, quantize_model


class BenchmarkRunner:
    """Executes systematic benchmarks on LLM inference strategies and saves empirical results."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.results: List[RunMetrics] = []

    def run_all(self, model_preset: str = "nano") -> List[RunMetrics]:
        """Runs the complete suite of inference experiments."""
        print(f"=== Starting LLM Inference Lab Benchmark Suite (Model Preset: {model_preset}) ===")

        # Initialize base model
        if model_preset == "micro":
            model_cfg = ModelConfig.micro()
        elif model_preset == "small":
            model_cfg = ModelConfig.small()
        else:
            model_cfg = ModelConfig.nano()

        base_model = CausalLM(model_cfg).to(self.config.device).eval()

        # 1. KV-Cache Scaling Suite
        self.benchmark_kv_cache_scaling(base_model)

        # 2. Batch Size Scaling Suite
        self.benchmark_batch_size_scaling(base_model)

        # 3. Quantization Suite
        self.benchmark_quantization(model_cfg)

        # 4. Speculative Decoding Suite
        self.benchmark_speculative_decoding()

        # 5. ONNX Runtime Engine Suite
        self.benchmark_onnx_runtime(model_cfg)

        # Export results to JSON and CSV
        self.save_results()
        print("=== Benchmark Suite Complete ===")
        return self.results

    def benchmark_kv_cache_scaling(self, model: CausalLM) -> None:
        """Suite 1: Compare Cached vs Non-Cached decoding across different token lengths."""
        print("\n--- Running Suite 1: KV-Cache vs Non-Cached Scaling ---")
        prompt_len = 32
        for gen_len in [16, 32, 64]:
            for use_cache in [True, False]:
                scenario_name = f"kv_cache_{'enabled' if use_cache else 'disabled'}"
                input_ids = torch.randint(
                    0, model.config.vocab_size, (1, prompt_len), device=self.config.device
                )
                gen_cfg = GenerationConfig(
                    max_new_tokens=gen_len,
                    do_sample=False,
                    use_cache=use_cache,
                )

                # Warmup
                for _ in range(self.config.warmup_steps):
                    _ = generate(model, input_ids, gen_cfg)

                # Profiling trials
                trial_metrics = []
                for _ in range(self.config.num_trials):
                    t0 = time.perf_counter()
                    tokens = []
                    decode_latencies = []
                    ttft = 0.0

                    for tok, step_time, is_prefill in stream_generate(model, input_ids, gen_cfg):
                        tokens.append(tok)
                        if is_prefill:
                            ttft = step_time
                        else:
                            decode_latencies.append(step_time)

                    total_time = time.perf_counter() - t0
                    kv_mem = (
                        model.allocate_kv_cache().theoretical_memory_mb(
                            1,
                            model.config.n_layers,
                            model.config.n_kv_heads or model.config.n_heads,
                            prompt_len + gen_len,
                            model.config.head_dim,
                        )
                        if use_cache
                        else 0.0
                    )

                    m = compute_metrics(
                        scenario=scenario_name,
                        batch_size=1,
                        prompt_len=prompt_len,
                        gen_len=gen_len,
                        use_cache=use_cache,
                        ttft_s=ttft,
                        tpot_latencies_s=decode_latencies,
                        total_time_s=total_time,
                        kv_cache_mb=kv_mem,
                    )
                    trial_metrics.append(m)

                # Pick median run
                trial_metrics.sort(key=lambda x: x.total_latency_ms)
                best_run = trial_metrics[len(trial_metrics) // 2]
                self.results.append(best_run)
                print(
                    f"[{scenario_name}] Gen {gen_len} tokens | "
                    f"TTFT: {best_run.ttft_ms:.1f}ms | "
                    f"Avg TPOT: {best_run.avg_tpot_ms:.1f}ms | "
                    f"Throughput: {best_run.tokens_per_second:.1f} tok/s"
                )

    def benchmark_batch_size_scaling(self, model: CausalLM) -> None:
        """Suite 2: Measure latency and throughput across batch sizes."""
        print("\n--- Running Suite 2: Batch Size Scaling ---")
        prompt_len = 32
        gen_len = 32
        for bs in self.config.batch_sizes:
            input_ids = torch.randint(
                0, model.config.vocab_size, (bs, prompt_len), device=self.config.device
            )
            gen_cfg = GenerationConfig(max_new_tokens=gen_len, do_sample=False, use_cache=True)

            # Warmup
            for _ in range(self.config.warmup_steps):
                _ = generate(model, input_ids, gen_cfg)

            trial_metrics = []
            for _ in range(self.config.num_trials):
                t0 = time.perf_counter()
                tokens = []
                decode_latencies = []
                ttft = 0.0

                for tok, step_time, is_prefill in stream_generate(model, input_ids, gen_cfg):
                    tokens.append(tok)
                    if is_prefill:
                        ttft = step_time
                    else:
                        decode_latencies.append(step_time)

                total_time = time.perf_counter() - t0
                m = compute_metrics(
                    scenario="batch_scaling",
                    batch_size=bs,
                    prompt_len=prompt_len,
                    gen_len=gen_len,
                    use_cache=True,
                    ttft_s=ttft,
                    tpot_latencies_s=decode_latencies,
                    total_time_s=total_time,
                )
                trial_metrics.append(m)

            trial_metrics.sort(key=lambda x: x.total_latency_ms)
            best_run = trial_metrics[len(trial_metrics) // 2]
            self.results.append(best_run)
            print(
                f"[Batch {bs}] Total: {best_run.total_latency_ms:.1f}ms | "
                f"Per-Stream: {best_run.tokens_per_second:.1f} tok/s | "
                f"Aggregate: {best_run.aggregate_throughput:.1f} tok/s"
            )

    def benchmark_quantization(self, model_cfg: ModelConfig) -> None:
        """Suite 3: Dynamic INT8 Weight Quantization vs FP32 Baseline."""
        print("\n--- Running Suite 3: INT8 Quantization vs FP32 Baseline ---")
        fp32_model = CausalLM(model_cfg).to(self.config.device).eval()
        fp32_bytes, param_count = calculate_model_weight_bytes(fp32_model)

        # Clone and quantize
        q_model = CausalLM(model_cfg).to(self.config.device).eval()
        q_model.load_state_dict(fp32_model.state_dict())
        quantize_model(q_model)
        int8_bytes, _ = calculate_model_weight_bytes(q_model)

        compression_ratio = fp32_bytes / max(int8_bytes, 1)
        print(f"FP32 Weights: {fp32_bytes / (1024*1024):.2f} MB | INT8 Weights: {int8_bytes / (1024*1024):.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")

        prompt_len = 32
        gen_len = 32
        input_ids = torch.randint(
            0, model_cfg.vocab_size, (1, prompt_len), device=self.config.device
        )
        gen_cfg = GenerationConfig(max_new_tokens=gen_len, do_sample=False, use_cache=True)

        for name, m_instance, w_bytes in [
            ("precision_fp32", fp32_model, fp32_bytes),
            ("precision_int8", q_model, int8_bytes),
        ]:
            # Warmup
            for _ in range(self.config.warmup_steps):
                _ = generate(m_instance, input_ids, gen_cfg)

            _, run_perf = generate(m_instance, input_ids, gen_cfg)

            metric = RunMetrics(
                scenario=name,
                batch_size=1,
                prompt_len=prompt_len,
                gen_len=gen_len,
                use_cache=True,
                ttft_ms=round(run_perf["ttft_ms"], 3),
                avg_tpot_ms=round(run_perf["avg_tpot_ms"], 3),
                p50_tpot_ms=round(run_perf["avg_tpot_ms"], 3),
                p95_tpot_ms=round(run_perf["avg_tpot_ms"], 3),
                total_latency_ms=round(run_perf["total_latency_ms"], 3),
                tokens_per_second=round(run_perf["tokens_per_second"], 2),
                aggregate_throughput=round(run_perf["tokens_per_second"], 2),
                kv_cache_mb=0.0,
                process_memory_mb=round(w_bytes / (1024 * 1024), 2),
                extra={"model_weight_mb": round(w_bytes / (1024 * 1024), 2)},
            )
            self.results.append(metric)
            print(
                f"[{name}] Latency: {metric.total_latency_ms:.1f}ms | "
                f"Throughput: {metric.tokens_per_second:.1f} tok/s | "
                f"Weight Size: {metric.process_memory_mb:.2f} MB"
            )

    def benchmark_speculative_decoding(self) -> None:
        """Suite 4: Speculative Decoding (Draft + Target Model)."""
        print("\n--- Running Suite 4: Speculative Decoding ---")
        # Target model: micro (8.5M), Draft model: nano (1.2M)
        target_cfg = ModelConfig.micro()
        draft_cfg = ModelConfig.nano()
        # Align vocab size for speculative pair
        draft_cfg.vocab_size = target_cfg.vocab_size

        target_model = CausalLM(target_cfg).to(self.config.device).eval()
        draft_model = CausalLM(draft_cfg).to(self.config.device).eval()

        prompt_len = 16
        gen_len = 32
        input_ids = torch.randint(
            0, target_cfg.vocab_size, (1, prompt_len), device=self.config.device
        )
        gen_cfg = GenerationConfig(max_new_tokens=gen_len, do_sample=False)

        # Baseline: Target model autoregressive standard
        for _ in range(self.config.warmup_steps):
            _ = generate(target_model, input_ids, gen_cfg)

        _, base_perf = generate(target_model, input_ids, gen_cfg)

        base_metric = RunMetrics(
            scenario="speculative_baseline_target",
            batch_size=1,
            prompt_len=prompt_len,
            gen_len=gen_len,
            use_cache=False,
            ttft_ms=round(base_perf["ttft_ms"], 3),
            avg_tpot_ms=round(base_perf["avg_tpot_ms"], 3),
            p50_tpot_ms=round(base_perf["avg_tpot_ms"], 3),
            p95_tpot_ms=round(base_perf["avg_tpot_ms"], 3),
            total_latency_ms=round(base_perf["total_latency_ms"], 3),
            tokens_per_second=round(base_perf["tokens_per_second"], 2),
            aggregate_throughput=round(base_perf["tokens_per_second"], 2),
            kv_cache_mb=0.0,
            process_memory_mb=0.0,
            extra={},
        )
        self.results.append(base_metric)

        # Speculative Decoder with gamma=3
        spec_decoder = SpeculativeDecoder(target_model=target_model, draft_model=draft_model, gamma=3)
        for _ in range(self.config.warmup_steps):
            _ = spec_decoder.generate(input_ids, gen_cfg)

        _, spec_perf = spec_decoder.generate(input_ids, gen_cfg)

        spec_metric = RunMetrics(
            scenario="speculative_decoding_gamma3",
            batch_size=1,
            prompt_len=prompt_len,
            gen_len=gen_len,
            use_cache=False,
            ttft_ms=0.0,
            avg_tpot_ms=0.0,
            p50_tpot_ms=0.0,
            p95_tpot_ms=0.0,
            total_latency_ms=round(spec_perf["total_latency_ms"], 3),
            tokens_per_second=round(spec_perf["tokens_per_second"], 2),
            aggregate_throughput=round(spec_perf["tokens_per_second"], 2),
            kv_cache_mb=0.0,
            process_memory_mb=0.0,
            extra={
                "acceptance_rate": round(spec_perf["acceptance_rate"], 3),
                "target_evaluations": spec_perf["target_evaluations"],
            },
        )
        self.results.append(spec_metric)
        print(
            f"[Speculative] Baseline Throughput: {base_metric.tokens_per_second:.1f} tok/s | "
            f"Speculative: {spec_metric.tokens_per_second:.1f} tok/s | "
            f"Acceptance Rate: {spec_perf['acceptance_rate']*100:.1f}%"
        )

    def benchmark_onnx_runtime(self, model_cfg: ModelConfig) -> None:
        """Suite 5: PyTorch CPU vs ONNX Runtime CPU forward pass latency."""
        print("\n--- Running Suite 5: PyTorch CPU vs ONNX Runtime CPU ---")
        model = CausalLM(model_cfg).to(self.config.device).eval()
        onnx_path = os.path.join(self.config.output_dir, "model_temp.onnx")

        try:
            print("Exporting model to ONNX...")
            export_to_onnx(model, onnx_path, dummy_seq_len=32)
            engine = ONNXInferenceEngine(onnx_path)

            dummy_input = torch.randint(0, model_cfg.vocab_size, (1, 32), device=self.config.device)

            # PyTorch warmup & bench
            for _ in range(self.config.warmup_steps):
                with torch.no_grad():
                    _ = model(dummy_input)

            pt_times = []
            for _ in range(10):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(dummy_input)
                pt_times.append((time.perf_counter() - t0) * 1000.0)

            ort_results = engine.benchmark_forward(batch_size=1, seq_len=32, num_runs=10)

            pt_mean = float(torch.tensor(pt_times).mean())
            ort_mean = ort_results["mean_ms"]

            self.results.append(
                RunMetrics(
                    scenario="engine_pytorch_cpu",
                    batch_size=1,
                    prompt_len=32,
                    gen_len=0,
                    use_cache=False,
                    ttft_ms=round(pt_mean, 3),
                    avg_tpot_ms=round(pt_mean, 3),
                    p50_tpot_ms=round(pt_mean, 3),
                    p95_tpot_ms=round(float(torch.tensor(pt_times).quantile(0.95)), 3),
                    total_latency_ms=round(pt_mean, 3),
                    tokens_per_second=0.0,
                    aggregate_throughput=0.0,
                    kv_cache_mb=0.0,
                    process_memory_mb=0.0,
                    extra={"framework": "pytorch_cpu"},
                )
            )

            self.results.append(
                RunMetrics(
                    scenario="engine_onnxruntime_cpu",
                    batch_size=1,
                    prompt_len=32,
                    gen_len=0,
                    use_cache=False,
                    ttft_ms=round(ort_mean, 3),
                    avg_tpot_ms=round(ort_mean, 3),
                    p50_tpot_ms=round(ort_mean, 3),
                    p95_tpot_ms=round(ort_results["max_ms"], 3),
                    total_latency_ms=round(ort_mean, 3),
                    tokens_per_second=0.0,
                    aggregate_throughput=0.0,
                    kv_cache_mb=0.0,
                    process_memory_mb=0.0,
                    extra={"framework": "onnxruntime_cpu"},
                )
            )

            print(
                f"[Runtime Comparison] PyTorch CPU Forward: {pt_mean:.2f}ms | "
                f"ONNX Runtime CPU: {ort_mean:.2f}ms | "
                f"Speedup: {pt_mean / max(ort_mean, 1e-4):.2f}x"
            )
        finally:
            # Clean up temporary ONNX model to respect disk space constraints
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def save_results(self) -> None:
        """Export benchmark results to structured JSON and CSV files."""
        json_path = os.path.join(self.config.output_dir, "benchmark_results.json")
        csv_path = os.path.join(self.config.output_dir, "benchmark_results.csv")

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

        # CSV
        if self.results:
            keys = [
                "scenario",
                "batch_size",
                "prompt_len",
                "gen_len",
                "use_cache",
                "ttft_ms",
                "avg_tpot_ms",
                "p50_tpot_ms",
                "p95_tpot_ms",
                "total_latency_ms",
                "tokens_per_second",
                "aggregate_throughput",
                "kv_cache_mb",
                "process_memory_mb",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                for r in self.results:
                    writer.writerow(r.to_dict())

        print(f"Results exported successfully to:\n  {json_path}\n  {csv_path}")

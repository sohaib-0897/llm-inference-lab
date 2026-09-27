from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import time
from typing import Any, List

import numpy as np

from llm_lab.backends.base import GenerationRequest, GenerationResult
from llm_lab.backends.ollama_backend import OllamaBackend


class OllamaBenchmarkSuite:
    """Automated benchmark experiments for real pretrained LLMs via Ollama."""

    def __init__(
        self,
        backend: OllamaBackend | None = None,
        output_dir: str = "benchmarks/results/ollama",
    ) -> None:
        self.backend = backend or OllamaBackend()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_data(self, base_name: str, records: List[dict[str, Any]]) -> None:
        """Export benchmark results to JSON and CSV formats."""
        json_path = os.path.join(self.output_dir, f"{base_name}.json")
        csv_path = os.path.join(self.output_dir, f"{base_name}.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        if records:
            # Flatten or select standard keys
            keys = list(records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                for r in records:
                    # format nested dicts if any
                    row = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()}
                    writer.writerow(row)

    # --------------------------------------------------------------------------
    # Experiment A: Real-Model Baseline
    # --------------------------------------------------------------------------
    def run_experiment_a_baseline(
        self,
        model_name: str = "qwen2.5:0.5b",
        num_warmup: int = 1,
        num_trials: int = 3,
        num_predict: int = 64,
    ) -> List[dict[str, Any]]:
        print(f"\n=== Experiment A: Real-Model Baseline ({model_name}) ===")
        prompt = (
            "Explain how key-value caching accelerates transformer inference in three concise sentences."
        )
        req = GenerationRequest(
            prompt=prompt,
            max_tokens=num_predict,
            temperature=0.0,
            seed=42,
        )

        # Warmup
        for i in range(num_warmup):
            print(f"  Warm-up run {i+1}/{num_warmup}...")
            _ = self.backend.generate(req, model_name=model_name)

        trials_results: List[GenerationResult] = []
        for i in range(num_trials):
            print(f"  Measured trial {i+1}/{num_trials}...")
            res = self.backend.generate(req, model_name=model_name)
            trials_results.append(res)

        ttfts = [r.client_ttft_ms for r in trials_results]
        latencies = [r.client_total_latency_ms for r in trials_results]
        throughputs = [r.client_tokens_per_sec for r in trials_results]
        server_throughputs = [
            r.server_eval_tokens_per_sec
            for r in trials_results
            if r.server_eval_tokens_per_sec is not None
        ]

        summary = {
            "experiment": "experiment_a_baseline",
            "model": model_name,
            "prompt": prompt,
            "trials_count": num_trials,
            "mean_ttft_ms": round(float(np.mean(ttfts)), 2),
            "median_ttft_ms": round(float(np.median(ttfts)), 2),
            "std_ttft_ms": round(float(np.std(ttfts)), 2),
            "mean_total_latency_ms": round(float(np.mean(latencies)), 2),
            "median_total_latency_ms": round(float(np.median(latencies)), 2),
            "p95_total_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "mean_client_tokens_per_sec": round(float(np.mean(throughputs)), 2),
            "mean_server_tokens_per_sec": round(float(np.mean(server_throughputs)), 2)
            if server_throughputs
            else None,
            "trials": [r.to_dict() for r in trials_results],
        }

        print(
            f"  Result -> TTFT: {summary['mean_ttft_ms']}ms | "
            f"Latency: {summary['mean_total_latency_ms']}ms | "
            f"Client Throughput: {summary['mean_client_tokens_per_sec']} tok/s | "
            f"Server Throughput: {summary['mean_server_tokens_per_sec']} tok/s"
        )

        self._save_data("baseline", [summary])
        return [summary]

    # --------------------------------------------------------------------------
    # Experiment B: Context-Length Scaling
    # --------------------------------------------------------------------------
    def run_experiment_b_context_scaling(
        self,
        model_name: str = "qwen2.5:0.5b",
        target_lengths: List[int] | None = None,
        num_predict: int = 32,
        trials: int = 2,
    ) -> List[dict[str, Any]]:
        print(f"\n=== Experiment B: Context-Length Scaling ({model_name}) ===")
        if target_lengths is None:
            target_lengths = [128, 256, 512, 1024, 2048]

        base_unit = (
            "Large language model inference requires significant computational and memory bandwidth resources. "
            "Autoregressive generation processes tokens sequentially using key-value cache mechanisms. "
        )

        records: List[dict[str, Any]] = []

        for target_tokens in target_lengths:
            # Estimate ~18 tokens per repetition
            repeat_count = max(1, target_tokens // 18)
            synthetic_prompt = f"Summarize the following context in 15 words: {' '.join([base_unit] * repeat_count)}"

            req = GenerationRequest(
                prompt=synthetic_prompt,
                max_tokens=num_predict,
                temperature=0.0,
                seed=42,
            )

            # Warmup
            _ = self.backend.generate(req, model_name=model_name)

            trial_ttft = []
            trial_prompt_eval_ms = []
            trial_total_ms = []
            trial_thru = []
            actual_prompt_tokens = 0

            for _ in range(trials):
                res = self.backend.generate(req, model_name=model_name)
                trial_ttft.append(res.client_ttft_ms)
                if res.server_prompt_eval_duration_ms:
                    trial_prompt_eval_ms.append(res.server_prompt_eval_duration_ms)
                trial_total_ms.append(res.client_total_latency_ms)
                trial_thru.append(res.client_tokens_per_sec)
                actual_prompt_tokens = res.prompt_tokens

            rec = {
                "target_approx_tokens": target_tokens,
                "actual_prompt_tokens": actual_prompt_tokens,
                "output_tokens": num_predict,
                "mean_prompt_eval_duration_ms": round(float(np.mean(trial_prompt_eval_ms)), 2)
                if trial_prompt_eval_ms
                else 0.0,
                "mean_client_ttft_ms": round(float(np.mean(trial_ttft)), 2),
                "mean_total_latency_ms": round(float(np.mean(trial_total_ms)), 2),
                "mean_tokens_per_sec": round(float(np.mean(trial_thru)), 2),
            }
            records.append(rec)
            print(
                f"  Prompt ~{target_tokens} tok (actual: {actual_prompt_tokens}) -> "
                f"Prompt Eval: {rec['mean_prompt_eval_duration_ms']}ms | "
                f"TTFT: {rec['mean_client_ttft_ms']}ms | "
                f"Total: {rec['mean_total_latency_ms']}ms"
            )

        self._save_data("context_scaling", records)
        return records

    # --------------------------------------------------------------------------
    # Experiment C: Output-Length Scaling
    # --------------------------------------------------------------------------
    def run_experiment_c_output_scaling(
        self,
        model_name: str = "qwen2.5:0.5b",
        output_lengths: List[int] | None = None,
        trials: int = 2,
    ) -> List[dict[str, Any]]:
        print(f"\n=== Experiment C: Output-Length Scaling ({model_name}) ===")
        if output_lengths is None:
            output_lengths = [16, 32, 64, 128]

        fixed_prompt = (
            "Write a detailed technological overview of GPU tensor cores, parallel memory hierarchies, "
            "and their utilization in modern artificial intelligence workloads."
        )

        records: List[dict[str, Any]] = []

        for out_len in output_lengths:
            req = GenerationRequest(
                prompt=fixed_prompt,
                max_tokens=out_len,
                temperature=0.0,
                seed=42,
            )

            # Warmup
            _ = self.backend.generate(req, model_name=model_name)

            trial_ttft = []
            trial_eval_duration = []
            trial_total = []
            trial_thru = []
            actual_gen_tokens = 0

            for _ in range(trials):
                res = self.backend.generate(req, model_name=model_name)
                trial_ttft.append(res.client_ttft_ms)
                if res.server_eval_duration_ms:
                    trial_eval_duration.append(res.server_eval_duration_ms)
                trial_total.append(res.client_total_latency_ms)
                trial_thru.append(res.client_tokens_per_sec)
                actual_gen_tokens = res.output_tokens

            rec = {
                "requested_output_tokens": out_len,
                "actual_output_tokens": actual_gen_tokens,
                "mean_client_ttft_ms": round(float(np.mean(trial_ttft)), 2),
                "mean_eval_duration_ms": round(float(np.mean(trial_eval_duration)), 2)
                if trial_eval_duration
                else 0.0,
                "mean_total_latency_ms": round(float(np.mean(trial_total)), 2),
                "mean_tokens_per_sec": round(float(np.mean(trial_thru)), 2),
            }
            records.append(rec)
            print(
                f"  Gen Length {out_len} tok -> "
                f"Eval Duration: {rec['mean_eval_duration_ms']}ms | "
                f"Total Latency: {rec['mean_total_latency_ms']}ms | "
                f"Throughput: {rec['mean_tokens_per_sec']} tok/s"
            )

        self._save_data("output_scaling", records)
        return records

    # --------------------------------------------------------------------------
    # Experiment D: Model-Size Comparison
    # --------------------------------------------------------------------------
    def run_experiment_d_model_comparison(
        self,
        models: List[str] | None = None,
        num_predict: int = 32,
        trials: int = 3,
    ) -> List[dict[str, Any]]:
        print("\n=== Experiment D: Model-Size Comparison ===")
        if models is None:
            models = ["qwen2.5:0.5b", "qwen3:4b"]

        prompt = "Explain in three bullet points why deep learning requires high memory bandwidth."
        records: List[dict[str, Any]] = []

        installed = {m["name"]: m for m in self.backend.list_models()}

        for m_name in models:
            if m_name not in installed and f"{m_name}:latest" not in installed:
                print(f"  Warning: Model '{m_name}' not found locally. Skipping comparison for it.")
                continue

            # Model metadata
            meta = installed.get(m_name, installed.get(f"{m_name}:latest", {}))
            size_mb = meta.get("size", 0) / (1024 * 1024)
            param_size = meta.get("details", {}).get("parameter_size", "unknown")
            quant_level = meta.get("details", {}).get("quantization_level", "unknown")

            req = GenerationRequest(
                prompt=prompt,
                max_tokens=num_predict,
                temperature=0.0,
                seed=42,
            )

            # Warmup
            _ = self.backend.generate(req, model_name=m_name)

            ttfts = []
            prompt_evals = []
            eval_thrus = []
            client_thrus = []
            totals = []

            for _ in range(trials):
                res = self.backend.generate(req, model_name=m_name)
                ttfts.append(res.client_ttft_ms)
                if res.server_prompt_eval_duration_ms:
                    prompt_evals.append(res.server_prompt_eval_duration_ms)
                if res.server_eval_tokens_per_sec:
                    eval_thrus.append(res.server_eval_tokens_per_sec)
                client_thrus.append(res.client_tokens_per_sec)
                totals.append(res.client_total_latency_ms)

            # Inspect active VRAM usage via /api/ps
            running = self.backend.get_running_models()
            running_info = next((r for r in running if r.get("name") == m_name), {})
            vram_mb = running_info.get("size_vram", 0) / (1024 * 1024)
            processor = running_info.get("processor", "unknown")

            rec = {
                "model_name": m_name,
                "param_size": param_size,
                "quantization": quant_level,
                "disk_size_mb": round(size_mb, 1),
                "vram_size_mb": round(vram_mb, 1) if vram_mb else None,
                "processor": processor,
                "mean_ttft_ms": round(float(np.mean(ttfts)), 2),
                "mean_prompt_eval_ms": round(float(np.mean(prompt_evals)), 2) if prompt_evals else 0.0,
                "mean_eval_tokens_per_sec": round(float(np.mean(eval_thrus)), 2) if eval_thrus else 0.0,
                "mean_client_tokens_per_sec": round(float(np.mean(client_thrus)), 2),
                "mean_total_latency_ms": round(float(np.mean(totals)), 2),
            }
            records.append(rec)
            print(
                f"  Model {m_name} ({param_size}) -> "
                f"Processor: {processor} | "
                f"TTFT: {rec['mean_ttft_ms']}ms | "
                f"Server Throughput: {rec['mean_eval_tokens_per_sec']} tok/s | "
                f"Total Latency: {rec['mean_total_latency_ms']}ms"
            )

        self._save_data("model_comparison", records)
        return records

    # --------------------------------------------------------------------------
    # Experiment E: Concurrency
    # --------------------------------------------------------------------------
    def run_experiment_e_concurrency(
        self,
        model_name: str = "qwen2.5:0.5b",
        concurrency_levels: List[int] | None = None,
        num_predict: int = 32,
    ) -> List[dict[str, Any]]:
        print(f"\n=== Experiment E: Concurrency Scaling ({model_name}) ===")
        if concurrency_levels is None:
            concurrency_levels = [1, 2, 4]

        prompt = "Write a haiku about artificial intelligence."
        req = GenerationRequest(
            prompt=prompt,
            max_tokens=num_predict,
            temperature=0.0,
            seed=42,
        )

        # Warmup
        _ = self.backend.generate(req, model_name=model_name)

        records: List[dict[str, Any]] = []

        for c_level in concurrency_levels:
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=c_level) as executor:
                futures = [
                    executor.submit(self.backend.generate, req, model_name)
                    for _ in range(c_level)
                ]
                results = [f.result() for f in futures]
            wall_time = time.perf_counter() - t0

            total_generated_tokens = sum(r.output_tokens for r in results)
            latencies = [r.client_total_latency_ms for r in results]
            ttfts = [r.client_ttft_ms for r in results]

            aggregate_thru = total_generated_tokens / wall_time if wall_time > 0 else 0.0

            rec = {
                "concurrency": c_level,
                "wall_time_ms": round(wall_time * 1000.0, 2),
                "total_tokens_generated": total_generated_tokens,
                "aggregate_tokens_per_sec": round(aggregate_thru, 2),
                "mean_latency_ms": round(float(np.mean(latencies)), 2),
                "median_latency_ms": round(float(np.median(latencies)), 2),
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
                "mean_ttft_ms": round(float(np.mean(ttfts)), 2),
                "per_request_mean_throughput": round(
                    float(np.mean([r.client_tokens_per_sec for r in results])), 2
                ),
            }
            records.append(rec)
            print(
                f"  Concurrency {c_level} -> "
                f"Wall Time: {rec['wall_time_ms']}ms | "
                f"Aggregate Throughput: {rec['aggregate_tokens_per_sec']} tok/s | "
                f"Mean Latency: {rec['mean_latency_ms']}ms | "
                f"P95: {rec['p95_latency_ms']}ms"
            )

        self._save_data("concurrency", records)
        return records

    def run_all(self, model_name: str = "qwen2.5:0.5b") -> dict[str, Any]:
        """Execute the entire suite of 5 real-model experiments."""
        print(f"=== Starting Ollama Real-Model Benchmark Suite (Primary: {model_name}) ===")
        exp_a = self.run_experiment_a_baseline(model_name=model_name)
        exp_b = self.run_experiment_b_context_scaling(model_name=model_name)
        exp_c = self.run_experiment_c_output_scaling(model_name=model_name)
        exp_d = self.run_experiment_d_model_comparison()
        exp_e = self.run_experiment_e_concurrency(model_name=model_name)

        print("\n=== Ollama Real-Model Benchmark Suite Complete ===")
        return {
            "baseline": exp_a,
            "context_scaling": exp_b,
            "output_scaling": exp_c,
            "model_comparison": exp_d,
            "concurrency": exp_e,
        }

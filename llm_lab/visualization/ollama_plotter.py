from __future__ import annotations

import json
import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np


def generate_ollama_plots(
    results_dir: str = "benchmarks/results/ollama",
) -> List[str]:
    """Generates visual charts from Ollama experiment JSON artifacts."""
    generated: List[str] = []

    # 1. Context Scaling Plot
    ctx_path = os.path.join(results_dir, "context_scaling.json")
    if os.path.exists(ctx_path):
        with open(ctx_path, "r", encoding="utf-8") as f:
            data: List[dict[str, Any]] = json.load(f)
        if data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            x_vals = [d.get("actual_prompt_tokens", d.get("target_approx_tokens")) for d in data]
            prompt_eval_ms = [d.get("mean_prompt_eval_duration_ms", 0.0) for d in data]
            ttft_ms = [d.get("mean_client_ttft_ms", 0.0) for d in data]

            ax1.plot(x_vals, prompt_eval_ms, marker="o", color="#1f77b4", linewidth=2)
            ax1.set_xlabel("Prompt Tokens (Context Length)")
            ax1.set_ylabel("Prompt Eval Duration (ms)")
            ax1.set_title("Ollama: Prompt Prefill Duration vs Context Length")
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2.plot(x_vals, ttft_ms, marker="s", color="#ff7f0e", linewidth=2)
            ax2.set_xlabel("Prompt Tokens (Context Length)")
            ax2.set_ylabel("Client-Observed TTFT (ms)")
            ax2.set_title("Ollama: Time to First Token (TTFT) vs Context Length")
            ax2.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            out_file = os.path.join(results_dir, "context_scaling.png")
            plt.savefig(out_file, dpi=200)
            plt.close()
            generated.append(out_file)

    # 2. Output Scaling Plot
    out_path = os.path.join(results_dir, "output_scaling.json")
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            gen_lens = [d["requested_output_tokens"] for d in data]
            eval_durations = [d["mean_eval_duration_ms"] for d in data]
            total_latencies = [d["mean_total_latency_ms"] for d in data]
            throughputs = [d["mean_tokens_per_sec"] for d in data]

            ax1.plot(gen_lens, eval_durations, marker="o", color="#2ca02c", label="Server Eval Duration")
            ax1.plot(gen_lens, total_latencies, marker="^", color="#1f77b4", label="Client Total Latency")
            ax1.set_xlabel("Generated Tokens")
            ax1.set_ylabel("Time (ms)")
            ax1.set_title("Ollama: Generation Duration vs Output Length")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2.plot(gen_lens, throughputs, marker="s", color="#d62728", linewidth=2)
            ax2.set_xlabel("Generated Tokens")
            ax2.set_ylabel("Tokens / Second")
            ax2.set_title("Ollama: Client Generation Throughput")
            ax2.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            out_file = os.path.join(results_dir, "output_scaling.png")
            plt.savefig(out_file, dpi=200)
            plt.close()
            generated.append(out_file)

    # 3. Model Comparison Plot
    model_path = os.path.join(results_dir, "model_comparison.json")
    if os.path.exists(model_path):
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
            models = [d["model_name"] for d in data]
            throughputs = [d["mean_eval_tokens_per_sec"] for d in data]
            ttfts = [d["mean_ttft_ms"] for d in data]

            x = np.arange(len(models))
            width = 0.4

            ax1.bar(x, throughputs, width, color=["#1f77b4", "#e377c2"])
            ax1.set_xticks(x)
            ax1.set_xticklabels(models)
            ax1.set_ylabel("Server Tokens / Sec")
            ax1.set_title("Decode Throughput by Model Size")
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2.bar(x, ttfts, width, color=["#ff7f0e", "#bcbd22"])
            ax2.set_xticks(x)
            ax2.set_xticklabels(models)
            ax2.set_ylabel("TTFT (ms)")
            ax2.set_title("Time to First Token (TTFT) by Model Size")
            ax2.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            out_file = os.path.join(results_dir, "model_comparison.png")
            plt.savefig(out_file, dpi=200)
            plt.close()
            generated.append(out_file)

    # 4. Concurrency Scaling Plot
    conc_path = os.path.join(results_dir, "concurrency.json")
    if os.path.exists(conc_path):
        with open(conc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            concurrencies = [d["concurrency"] for d in data]
            agg_thrus = [d["aggregate_tokens_per_sec"] for d in data]
            mean_lat = [d["mean_latency_ms"] for d in data]
            p95_lat = [d["p95_latency_ms"] for d in data]

            ax1.bar([str(c) for c in concurrencies], agg_thrus, color="#9467bd", width=0.4)
            ax1.set_xlabel("Concurrency Level (Simultaneous Requests)")
            ax1.set_ylabel("Aggregate Tokens / Sec")
            ax1.set_title("Ollama: Aggregate Serving Throughput under Concurrency")
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2.plot([str(c) for c in concurrencies], mean_lat, marker="o", label="Mean Latency", color="#1f77b4")
            ax2.plot([str(c) for c in concurrencies], p95_lat, marker="s", label="P95 Latency", color="#d62728")
            ax2.set_xlabel("Concurrency Level")
            ax2.set_ylabel("Request Latency (ms)")
            ax2.set_title("Latency Distribution Under Concurrency")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            out_file = os.path.join(results_dir, "concurrency.png")
            plt.savefig(out_file, dpi=200)
            plt.close()
            generated.append(out_file)

    return generated

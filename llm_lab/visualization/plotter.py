from __future__ import annotations

import json
import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np


def generate_benchmark_plots(
    results_json_path: str,
    output_dir: str = "benchmarks/results",
) -> List[str]:
    """Generates visual benchmark charts from structured benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    with open(results_json_path, "r", encoding="utf-8") as f:
        data: List[dict[str, Any]] = json.load(f)

    generated_charts = []

    # 1. KV-Cache Scaling Plot
    kv_runs = [d for d in data if "kv_cache_" in d["scenario"]]
    if kv_runs:
        chart_path = os.path.join(output_dir, "kv_cache_scaling.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        gen_lens = sorted(list(set(d["gen_len"] for d in kv_runs)))
        cached_tpots = [
            next(d["avg_tpot_ms"] for d in kv_runs if d["scenario"] == "kv_cache_enabled" and d["gen_len"] == gl)
            for gl in gen_lens
        ]
        non_cached_tpots = [
            next(d["avg_tpot_ms"] for d in kv_runs if d["scenario"] == "kv_cache_disabled" and d["gen_len"] == gl)
            for gl in gen_lens
        ]

        cached_thru = [
            next(d["tokens_per_second"] for d in kv_runs if d["scenario"] == "kv_cache_enabled" and d["gen_len"] == gl)
            for gl in gen_lens
        ]
        non_cached_thru = [
            next(d["tokens_per_second"] for d in kv_runs if d["scenario"] == "kv_cache_disabled" and d["gen_len"] == gl)
            for gl in gen_lens
        ]

        x = np.arange(len(gen_lens))
        width = 0.35

        # Subplot 1: TPOT Comparison
        ax1.bar(x - width / 2, cached_tpots, width, label="With KV Cache", color="#2ca02c")
        ax1.bar(x + width / 2, non_cached_tpots, width, label="No KV Cache (Recompute)", color="#d62728")
        ax1.set_xlabel("Generated Token Length")
        ax1.set_ylabel("Avg Time Per Output Token (ms)")
        ax1.set_title("Decode Latency: KV Cache vs Naive Recompute")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(gl) for gl in gen_lens])
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Subplot 2: Throughput Comparison
        ax2.plot(gen_lens, cached_thru, marker="o", linewidth=2, label="With KV Cache", color="#2ca02c")
        ax2.plot(gen_lens, non_cached_thru, marker="s", linewidth=2, label="No KV Cache", color="#d62728")
        ax2.set_xlabel("Generated Token Length")
        ax2.set_ylabel("Throughput (tokens/sec)")
        ax2.set_title("Token Generation Throughput")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=200)
        plt.close()
        generated_charts.append(chart_path)

    # 2. Batch Scaling Plot
    batch_runs = [d for d in data if d["scenario"] == "batch_scaling"]
    if batch_runs:
        chart_path = os.path.join(output_dir, "batch_scaling.png")
        fig, ax1 = plt.subplots(figsize=(7, 5))

        batch_sizes = [d["batch_size"] for d in batch_runs]
        per_stream = [d["tokens_per_second"] for d in batch_runs]
        agg_throughput = [d["aggregate_throughput"] for d in batch_runs]

        x = np.arange(len(batch_sizes))
        width = 0.35

        ax1.bar(x - width / 2, per_stream, width, label="Per-Stream Throughput", color="#1f77b4")
        ax1.bar(x + width / 2, agg_throughput, width, label="Aggregate Batch Throughput", color="#ff7f0e")
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Tokens / Second")
        ax1.set_title("Batch Size Scaling and Throughput Efficiency")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(b) for b in batch_sizes])
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=200)
        plt.close()
        generated_charts.append(chart_path)

    # 3. Quantization Footprint Plot
    q_runs = [d for d in data if d["scenario"].startswith("precision_")]
    if q_runs:
        chart_path = os.path.join(output_dir, "quantization_comparison.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        labels = ["FP32", "INT8"]
        mem_mb = [d["process_memory_mb"] for d in q_runs]
        lat_ms = [d["total_latency_ms"] for d in q_runs]

        ax1.bar(labels, mem_mb, color=["#1f77b4", "#2ca02c"], width=0.5)
        ax1.set_ylabel("Weight Size (MB)")
        ax1.set_title("Model Memory Footprint (4x INT8 Compression)")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.bar(labels, lat_ms, color=["#1f77b4", "#ff7f0e"], width=0.5)
        ax2.set_ylabel("Total Latency (ms)")
        ax2.set_title("Generation Latency (FP32 vs INT8)")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=200)
        plt.close()
        generated_charts.append(chart_path)

    return generated_charts

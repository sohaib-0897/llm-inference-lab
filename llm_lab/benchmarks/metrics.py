from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List

import numpy as np
import psutil


@dataclass
class RunMetrics:
    """Comprehensive performance measurements for an inference execution run."""

    scenario: str
    batch_size: int
    prompt_len: int
    gen_len: int
    use_cache: bool
    ttft_ms: float
    avg_tpot_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float
    total_latency_ms: float
    tokens_per_second: float
    aggregate_throughput: float  # batch_size * tokens_per_second
    kv_cache_mb: float
    process_memory_mb: float
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_current_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in megabytes."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def compute_metrics(
    scenario: str,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    use_cache: bool,
    ttft_s: float,
    tpot_latencies_s: List[float],
    total_time_s: float,
    kv_cache_mb: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> RunMetrics:
    """Compute and structure statistical latency and throughput metrics."""
    ttft_ms = ttft_s * 1000.0
    tpot_ms = [t * 1000.0 for t in tpot_latencies_s] if tpot_latencies_s else [0.0]

    avg_tpot = float(np.mean(tpot_ms))
    p50_tpot = float(np.percentile(tpot_ms, 50))
    p95_tpot = float(np.percentile(tpot_ms, 95))
    total_latency_ms = total_time_s * 1000.0

    num_tokens = max(len(tpot_latencies_s) + 1, 1)  # prefill token + decode tokens
    tokens_per_sec = num_tokens / total_time_s if total_time_s > 0 else 0.0
    agg_throughput = tokens_per_sec * batch_size

    return RunMetrics(
        scenario=scenario,
        batch_size=batch_size,
        prompt_len=prompt_len,
        gen_len=gen_len,
        use_cache=use_cache,
        ttft_ms=round(ttft_ms, 3),
        avg_tpot_ms=round(avg_tpot, 3),
        p50_tpot_ms=round(p50_tpot, 3),
        p95_tpot_ms=round(p95_tpot, 3),
        total_latency_ms=round(total_latency_ms, 3),
        tokens_per_second=round(tokens_per_sec, 2),
        aggregate_throughput=round(agg_throughput, 2),
        kv_cache_mb=round(kv_cache_mb, 4),
        process_memory_mb=round(get_current_process_memory_mb(), 2),
        extra=extra or {},
    )

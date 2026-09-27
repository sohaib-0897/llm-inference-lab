from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS: dict[str, tuple[str, str]] = {
    "custom": ("benchmarks/results/benchmark_results.json", "Custom PyTorch and ONNX"),
    "ollama_baseline": ("benchmarks/results/ollama/baseline.json", "Ollama baseline"),
    "ollama_context": ("benchmarks/results/ollama/context_scaling.json", "Ollama context scaling"),
    "ollama_output": ("benchmarks/results/ollama/output_scaling.json", "Ollama output scaling"),
    "ollama_models": ("benchmarks/results/ollama/model_comparison.json", "Ollama model comparison"),
    "ollama_concurrency": ("benchmarks/results/ollama/concurrency.json", "Ollama concurrency"),
}


class ArtifactMissingError(FileNotFoundError):
    """An authoritative benchmark artifact is missing."""


@lru_cache(maxsize=len(ARTIFACTS))
def read_artifact(family: str) -> list[dict[str, Any]]:
    if family not in ARTIFACTS:
        raise KeyError(f"Unknown benchmark family: {family}")
    relative, _ = ARTIFACTS[family]
    path = ROOT / relative
    if not path.is_file():
        raise ArtifactMissingError(relative)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Invalid benchmark artifact format: {relative}")
    return payload


def benchmark_index() -> list[dict[str, Any]]:
    return [
        {"name": key, "description": description, "artifact": path,
         "available": (ROOT / path).is_file()}
        for key, (path, description) in ARTIFACTS.items()
    ]

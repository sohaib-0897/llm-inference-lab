from __future__ import annotations

import importlib.util
import platform
from typing import TypedDict

import psutil
import torch

from llm_lab import __version__
from llm_lab.backends.ollama_backend import OllamaBackend


class RuntimeInfo(TypedDict):
    python_version: str
    project_version: str
    cpu: str
    ram_total_mb: int
    gpu: str | None
    ollama_available: bool
    ollama_version: str | None
    installed_backends: list[str]


def runtime_info() -> RuntimeInfo:
    ollama = OllamaBackend(timeout=3.0)
    available = ollama.is_available()
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "python_version": platform.python_version(),
        "project_version": __version__,
        "cpu": platform.processor() or "Unknown CPU",
        "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024)),
        "gpu": gpu,
        "ollama_available": available,
        "ollama_version": ollama.get_runtime_version() if available else None,
        "installed_backends": [
            "custom_pytorch",
            *( ["onnxruntime"] if importlib.util.find_spec("onnxruntime") else [] ),
            *( ["ollama"] if available else [] ),
        ],
    }

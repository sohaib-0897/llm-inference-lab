"""LLM Inference Lab - A lightweight, modular laboratory for LLM inference profiling,
KV-cache optimization, speculative decoding, and quantization benchmarking.
"""

__version__ = "0.1.0"

from llm_lab.config import BenchmarkConfig, GenerationConfig, ModelConfig
from llm_lab.engine.generation import generate
from llm_lab.engine.kv_cache import KVCache
from llm_lab.models.transformer import CausalLM

__all__ = [
    "ModelConfig",
    "GenerationConfig",
    "BenchmarkConfig",
    "CausalLM",
    "KVCache",
    "generate",
]

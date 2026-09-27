from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Architectural configuration for modern Causal Decoder-Only LLMs."""

    vocab_size: int = 4096
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: Optional[int] = 4  # Supports Grouped-Query Attention (GQA) if < n_heads
    n_layers: int = 4
    d_ff: Optional[int] = None  # If None, defaults to 8/3 * d_model (SwiGLU standard)
    max_seq_len: int = 1024
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_ff is None:
            # SwiGLU standard dimension: (2/3) * 4 * d_model rounded to multiple of 64
            hidden = int(2 * 4 * self.d_model / 3)
            self.d_ff = ((hidden + 63) // 64) * 64

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def nano(cls) -> ModelConfig:
        """Ultra-lightweight ~1.2M parameter preset for rapid unit testing."""
        return cls(
            vocab_size=1024,
            d_model=128,
            n_heads=4,
            n_kv_heads=2,
            n_layers=3,
            max_seq_len=512,
        )

    @classmethod
    def micro(cls) -> ModelConfig:
        """Lightweight ~8.5M parameter preset balanced for CPU profiling."""
        return cls(
            vocab_size=4096,
            d_model=256,
            n_heads=8,
            n_kv_heads=4,
            n_layers=6,
            max_seq_len=1024,
        )

    @classmethod
    def small(cls) -> ModelConfig:
        """~32M parameter preset for detailed multi-head and GQA analysis."""
        return cls(
            vocab_size=8192,
            d_model=512,
            n_heads=16,
            n_kv_heads=4,
            n_layers=8,
            max_seq_len=2048,
        )


@dataclass
class GenerationConfig:
    """Generation parameters for autoregressive sampling."""

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    do_sample: bool = False  # False = Greedy decoding
    eos_token_id: Optional[int] = None
    pad_token_id: int = 0
    use_cache: bool = True


@dataclass
class BenchmarkConfig:
    """Experiment setup for latency, throughput, and memory profiling."""

    suite_name: str = "default_benchmark"
    device: str = "cpu"
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4])
    prompt_lengths: list[int] = field(default_factory=lambda: [16, 64, 128])
    generate_lengths: list[int] = field(default_factory=lambda: [16, 32, 64])
    warmup_steps: int = 2
    num_trials: int = 3
    output_dir: str = "benchmarks/results"

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Generator, Optional


@dataclass
class GenerationRequest:
    """Unified generation options across backends."""

    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0  # 0.0 for deterministic evaluation
    top_k: int = 50
    top_p: float = 0.95
    seed: Optional[int] = 42
    num_ctx: Optional[int] = 4096
    stream: bool = True


@dataclass
class GenerationResult:
    """Standardized output and metrics across inference backends.

    Clearly separates client-observed wall-clock measurements from backend-reported metrics.
    """

    backend: str
    model_name: str
    text: str
    prompt_tokens: int
    output_tokens: int

    # Client wall-clock metrics (observed by the caller)
    client_ttft_ms: float
    client_total_latency_ms: float
    client_avg_tpot_ms: float
    client_tokens_per_sec: float

    # Server / Runtime reported metrics (extracted directly from engine/daemon)
    server_load_duration_ms: Optional[float] = None
    server_prompt_eval_duration_ms: Optional[float] = None
    server_prompt_eval_count: Optional[int] = None
    server_eval_duration_ms: Optional[float] = None
    server_eval_count: Optional[int] = None
    server_eval_tokens_per_sec: Optional[float] = None

    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseInferenceBackend(ABC):
    """Abstract interface for LLM inference engines (custom PyTorch, ONNX, Ollama)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if backend runtime is reachable and operational."""
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest, model_name: Optional[str] = None) -> GenerationResult:
        """Run full autoregressive generation and return standardized metrics."""
        pass

    @abstractmethod
    def stream(
        self, request: GenerationRequest, model_name: Optional[str] = None
    ) -> Generator[tuple[str, float, bool], None, None]:
        """Stream generated text chunks.

        Yields:
            (chunk_text, step_latency_ms, is_first_chunk)
        """
        pass

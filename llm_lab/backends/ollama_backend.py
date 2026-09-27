from __future__ import annotations

import json
import time
from typing import Any, Generator, Optional

import requests

from llm_lab.backends.base import BaseInferenceBackend, GenerationRequest, GenerationResult


class OllamaError(Exception):
    """Base exception for Ollama backend errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when Ollama daemon is unreachable or offline."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not found in local Ollama installation."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when a generation request times out."""
    pass


class OllamaBackend(BaseInferenceBackend):
    """Ollama HTTP REST backend for real pretrained model inference."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama local daemon is running and responding."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List all installed local models and metadata."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if resp.status_code != 200:
                raise OllamaError(f"Failed to fetch models: HTTP {resp.status_code}")
            data = resp.json()
            return data.get("models", [])
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.base_url}") from e
        except requests.exceptions.Timeout as e:
            raise OllamaTimeoutError("Timeout while contacting Ollama") from e

    def get_running_models(self) -> list[dict[str, Any]]:
        """Query currently active models loaded in VRAM/RAM via /api/ps."""
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=5.0)
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("models", [])
        except Exception:
            return []

    def get_runtime_version(self) -> Optional[str]:
        """Fetch version of running Ollama service."""
        try:
            resp = requests.get(f"{self.base_url}/api/version", timeout=3.0)
            if resp.status_code == 200:
                return resp.json().get("version")
        except Exception:
            pass
        return None

    def stream(
        self, request: GenerationRequest, model_name: Optional[str] = None
    ) -> Generator[tuple[str, float, bool], None, None]:
        """Stream generated text chunks and step timings from Ollama."""
        target_model = model_name or "qwen2.5:0.5b"
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "num_predict": request.max_tokens,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        if request.num_ctx is not None:
            options["num_ctx"] = request.num_ctx

        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": request.prompt,
            "stream": True,
            "options": options,
        }

        try:
            t0 = time.perf_counter()
            prev_t = t0
            is_first = True

            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as response:
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(f"Model '{target_model}' not found in Ollama.")
                if response.status_code != 200:
                    raise OllamaError(f"Ollama request failed with HTTP {response.status_code}: {response.text}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    now = time.perf_counter()
                    chunk_latency = (now - prev_t) * 1000.0
                    prev_t = now

                    data = json.loads(line)
                    chunk_text = data.get("response", "")
                    yield chunk_text, chunk_latency, is_first
                    is_first = False

        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.base_url}") from e
        except requests.exceptions.Timeout as e:
            raise OllamaTimeoutError("Ollama request timed out.") from e

    def generate(self, request: GenerationRequest, model_name: Optional[str] = None) -> GenerationResult:
        """Run full generation request and compute both client and server metrics."""
        target_model = model_name or "qwen2.5:0.5b"
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "num_predict": request.max_tokens,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        if request.num_ctx is not None:
            options["num_ctx"] = request.num_ctx

        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": request.prompt,
            "stream": True,  # Use streaming to accurately capture client TTFT
            "options": options,
        }

        try:
            t_req_start = time.perf_counter()
            first_token_time: Optional[float] = None
            tpot_list: list[float] = []
            prev_token_time = t_req_start
            collected_text: list[str] = []
            final_metadata: dict[str, Any] = {}

            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as response:
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(f"Model '{target_model}' not found in Ollama.")
                if response.status_code != 200:
                    raise OllamaError(f"Ollama request failed with HTTP {response.status_code}: {response.text}")

                for line in response.iter_lines():
                    if not line:
                        continue
                    now = time.perf_counter()
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        collected_text.append(chunk)
                        if first_token_time is None:
                            first_token_time = now
                        else:
                            tpot_list.append((now - prev_token_time) * 1000.0)
                        prev_token_time = now

                    if data.get("done", False):
                        final_metadata = data

            t_req_end = time.perf_counter()
            client_total_ms = (t_req_end - t_req_start) * 1000.0
            client_ttft_ms = ((first_token_time - t_req_start) * 1000.0) if first_token_time else client_total_ms
            client_avg_tpot_ms = (sum(tpot_list) / len(tpot_list)) if tpot_list else 0.0

            # Server reported metrics (durations given in nanoseconds)
            load_ns = final_metadata.get("load_duration", 0)
            prompt_eval_ns = final_metadata.get("prompt_eval_duration", 0)
            prompt_eval_count = final_metadata.get("prompt_eval_count", 0)
            eval_ns = final_metadata.get("eval_duration", 0)
            eval_count = final_metadata.get("eval_count", len(collected_text))

            server_load_ms = (load_ns / 1_000_000.0) if load_ns else None
            server_prompt_eval_ms = (prompt_eval_ns / 1_000_000.0) if prompt_eval_ns else None
            server_eval_ms = (eval_ns / 1_000_000.0) if eval_ns else None

            # Server-calculated throughput (tokens/sec strictly during decode phase)
            server_tokens_per_sec = (
                (eval_count / (eval_ns / 1_000_000_000.0))
                if (eval_ns and eval_ns > 0)
                else None
            )

            # Client wall-clock throughput (generated tokens / total request time)
            client_tokens_per_sec = (
                (eval_count / ((t_req_end - t_req_start)))
                if (t_req_end > t_req_start)
                else 0.0
            )

            full_text = "".join(collected_text)

            return GenerationResult(
                backend="ollama",
                model_name=target_model,
                text=full_text,
                prompt_tokens=prompt_eval_count,
                output_tokens=eval_count,
                client_ttft_ms=round(client_ttft_ms, 3),
                client_total_latency_ms=round(client_total_ms, 3),
                client_avg_tpot_ms=round(client_avg_tpot_ms, 3),
                client_tokens_per_sec=round(client_tokens_per_sec, 2),
                server_load_duration_ms=round(server_load_ms, 3) if server_load_ms is not None else None,
                server_prompt_eval_duration_ms=round(server_prompt_eval_ms, 3) if server_prompt_eval_ms is not None else None,
                server_prompt_eval_count=prompt_eval_count,
                server_eval_duration_ms=round(server_eval_ms, 3) if server_eval_ms is not None else None,
                server_eval_count=eval_count,
                server_eval_tokens_per_sec=round(server_tokens_per_sec, 2) if server_tokens_per_sec is not None else None,
                raw_metadata=final_metadata,
            )

        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama daemon at {self.base_url}") from e
        except requests.exceptions.Timeout as e:
            raise OllamaTimeoutError("Ollama request timed out.") from e

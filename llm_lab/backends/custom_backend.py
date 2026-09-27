from __future__ import annotations

import time
from typing import Generator, Optional

import torch

from llm_lab.backends.base import BaseInferenceBackend, GenerationRequest, GenerationResult
from llm_lab.config import GenerationConfig, ModelConfig
from llm_lab.engine.generation import stream_generate
from llm_lab.models.transformer import CausalLM


class CustomPyTorchBackend(BaseInferenceBackend):
    """Inference backend wrapping the from-scratch PyTorch CausalLM transformer engine."""

    def __init__(self, model_preset: str = "nano", device: str = "cpu") -> None:
        self.preset = model_preset
        self.device = device
        cfg_factory = getattr(ModelConfig, model_preset, ModelConfig.nano)
        self.config = cfg_factory()
        self.model = CausalLM(self.config).to(device).eval()

    def is_available(self) -> bool:
        return True

    def stream(
        self, request: GenerationRequest, model_name: Optional[str] = None
    ) -> Generator[tuple[str, float, bool], None, None]:
        gen_cfg = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.temperature > 0.0,
            use_cache=True,
        )
        dummy_input = torch.randint(0, self.config.vocab_size, (1, 8), device=self.device)
        for tok, step_time, is_prefill in stream_generate(self.model, dummy_input, gen_cfg):
            yield f"token_{tok.item()}", step_time * 1000.0, is_prefill

    def generate(self, request: GenerationRequest, model_name: Optional[str] = None) -> GenerationResult:
        gen_cfg = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.temperature > 0.0,
            use_cache=True,
        )
        prompt_len = 8
        dummy_input = torch.randint(0, self.config.vocab_size, (1, prompt_len), device=self.device)

        tokens = []
        tpot_list: list[float] = []
        ttft_s = 0.0
        t0 = time.perf_counter()

        for tok, step_time, is_prefill in stream_generate(self.model, dummy_input, gen_cfg):
            tokens.append(tok)
            if is_prefill:
                ttft_s = step_time
            else:
                tpot_list.append(step_time * 1000.0)

        total_time_s = time.perf_counter() - t0
        client_ttft_ms = ttft_s * 1000.0
        client_total_ms = total_time_s * 1000.0
        client_avg_tpot_ms = (sum(tpot_list) / len(tpot_list)) if tpot_list else 0.0
        client_tokens_per_sec = len(tokens) / total_time_s if total_time_s > 0 else 0.0

        return GenerationResult(
            backend="custom_pytorch",
            model_name=self.preset,
            text=" ".join(str(t.item()) for t in tokens),
            prompt_tokens=prompt_len,
            output_tokens=len(tokens),
            client_ttft_ms=round(client_ttft_ms, 3),
            client_total_latency_ms=round(client_total_ms, 3),
            client_avg_tpot_ms=round(client_avg_tpot_ms, 3),
            client_tokens_per_sec=round(client_tokens_per_sec, 2),
            server_load_duration_ms=0.0,
            server_prompt_eval_duration_ms=round(client_ttft_ms, 3),
            server_prompt_eval_count=prompt_len,
            server_eval_duration_ms=round(sum(tpot_list), 3),
            server_eval_count=len(tokens) - 1,
            server_eval_tokens_per_sec=round(len(tpot_list) / (sum(tpot_list) / 1000.0), 2) if tpot_list else 0.0,
            raw_metadata={"device": self.device, "preset": self.preset},
        )

from __future__ import annotations

import time
from typing import Optional, Tuple

import torch

from llm_lab.config import GenerationConfig
from llm_lab.models.transformer import CausalLM


class SpeculativeDecoder:
    """Implements Speculative Decoding (Leviathan et al., 2023) using a draft model

    and a target model to accelerate autoregressive generation.
    """

    def __init__(
        self,
        target_model: CausalLM,
        draft_model: CausalLM,
        gamma: int = 3,  # Number of speculative draft tokens per verification step
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma

    def generate(
        self,
        input_ids: torch.Tensor,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """Generate tokens using speculative draft and target verification.

        Args:
            input_ids: Prompt token IDs of shape (1, seq_len) (batch size 1)
            gen_config: Generation configuration

        Returns:
            Tuple of (output_ids, performance_metrics)
        """
        if gen_config is None:
            gen_config = GenerationConfig()

        if input_ids.shape[0] != 1:
            raise ValueError("Speculative decoding currently supports batch size 1.")

        self.target_model.eval()
        self.draft_model.eval()

        prefix = input_ids.clone()
        max_tokens = gen_config.max_new_tokens

        total_accepted_draft_tokens = 0
        total_speculated_tokens = 0
        target_evaluations = 0

        t0 = time.perf_counter()

        with torch.no_grad():
            while (prefix.shape[1] - input_ids.shape[1]) < max_tokens:
                cur_len = prefix.shape[1]

                # 1. Draft model generates gamma speculative tokens
                draft_tokens = []
                draft_context = prefix.clone()

                for _ in range(self.gamma):
                    logits_draft = self.draft_model(draft_context, start_pos=0, use_cache=False)
                    next_draft_tok = torch.argmax(logits_draft[:, -1, :], dim=-1, keepdim=True)
                    draft_tokens.append(next_draft_tok)
                    draft_context = torch.cat([draft_context, next_draft_tok], dim=-1)

                draft_tensor = torch.cat(draft_tokens, dim=-1)  # (1, gamma)
                total_speculated_tokens += self.gamma

                # 2. Target model evaluates prefix + draft_tensor in ONE parallel forward pass
                candidate_sequence = torch.cat([prefix, draft_tensor], dim=-1)
                target_logits = self.target_model(candidate_sequence, start_pos=0, use_cache=False)
                target_evaluations += 1

                # 3. Verification of each draft token
                for i in range(self.gamma):
                    pos_in_logits = cur_len - 1 + i
                    target_choice = torch.argmax(target_logits[:, pos_in_logits, :], dim=-1, keepdim=True)
                    draft_choice = draft_tensor[:, i : i + 1]

                    if target_choice.item() == draft_choice.item():
                        prefix = torch.cat([prefix, draft_choice], dim=-1)
                        total_accepted_draft_tokens += 1
                        if (prefix.shape[1] - input_ids.shape[1]) >= max_tokens:
                            break
                    else:
                        # Reject draft token and take target model's correction token
                        prefix = torch.cat([prefix, target_choice], dim=-1)
                        break
                else:
                    # If all gamma draft tokens were accepted, sample one bonus token from target model
                    bonus_pos = cur_len - 1 + self.gamma
                    bonus_token = torch.argmax(target_logits[:, bonus_pos, :], dim=-1, keepdim=True)
                    prefix = torch.cat([prefix, bonus_token], dim=-1)

        total_time = time.perf_counter() - t0
        generated_count = prefix.shape[1] - input_ids.shape[1]
        acceptance_rate = (
            (total_accepted_draft_tokens / total_speculated_tokens)
            if total_speculated_tokens > 0
            else 0.0
        )
        tokens_per_second = generated_count / total_time if total_time > 0 else 0.0

        metrics = {
            "total_latency_ms": total_time * 1000.0,
            "tokens_generated": float(generated_count),
            "tokens_per_second": tokens_per_second,
            "acceptance_rate": acceptance_rate,
            "target_evaluations": float(target_evaluations),
            "total_speculated_tokens": float(total_speculated_tokens),
            "accepted_draft_tokens": float(total_accepted_draft_tokens),
        }

        return prefix, metrics

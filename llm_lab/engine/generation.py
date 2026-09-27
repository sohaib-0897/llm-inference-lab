from __future__ import annotations

import time
from typing import Generator, Optional, Tuple

import torch
import torch.nn.functional as F

from llm_lab.config import GenerationConfig
from llm_lab.engine.kv_cache import KVCache
from llm_lab.models.transformer import CausalLM


def apply_repetition_penalty(
    logits: torch.Tensor,
    context_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Applies repetition penalty by penalizing logits of previously generated tokens."""
    if penalty == 1.0 or context_tokens.numel() == 0:
        return logits

    logits = logits.clone()
    batch_size = logits.shape[0]

    for b in range(batch_size):
        tokens = context_tokens[b].unique()
        for token_id in tokens:
            token_val = logits[b, token_id]
            if token_val > 0:
                logits[b, token_id] = token_val / penalty
            else:
                logits[b, token_id] = token_val * penalty

    return logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    context_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample or select the next token from the output logits.

    Args:
        logits: Shape (batch_size, vocab_size)
        config: GenerationConfig options
        context_tokens: Optional past tokens for repetition penalty

    Returns:
        Tensor of shape (batch_size, 1) containing selected token ids
    """
    if context_tokens is not None and config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, context_tokens, config.repetition_penalty)

    if not config.do_sample or config.temperature <= 0.0:
        # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Sampling with temperature
    scaled_logits = logits / max(config.temperature, 1e-5)
    filtered_logits = top_k_top_p_filtering(
        scaled_logits, top_k=config.top_k, top_p=config.top_p
    )
    probs = F.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens


def stream_generate(
    model: CausalLM,
    input_ids: torch.Tensor,
    gen_config: Optional[GenerationConfig] = None,
) -> Generator[Tuple[torch.Tensor, float, bool], None, None]:
    """Yields generated tokens one-by-one with per-token step latency.

    Yields:
        (token_tensor, step_latency_seconds, is_first_token)
    """
    if gen_config is None:
        gen_config = GenerationConfig()

    model.eval()
    prompt_len = input_ids.shape[1]
    current_tokens = input_ids.clone()

    kv_cache: Optional[KVCache] = model.allocate_kv_cache() if gen_config.use_cache else None

    # --- PREFILL PHASE ---
    t0 = time.perf_counter()
    with torch.no_grad():
        if gen_config.use_cache:
            logits = model(input_ids, start_pos=0, kv_cache=kv_cache, use_cache=True)
            last_logits = logits[:, -1, :]
        else:
            logits = model(input_ids, start_pos=0, use_cache=False)
            last_logits = logits[:, -1, :]

    next_token = sample_next_token(last_logits, gen_config, current_tokens)
    prefill_time = time.perf_counter() - t0
    current_tokens = torch.cat([current_tokens, next_token], dim=-1)

    yield next_token, prefill_time, True

    # --- DECODE PHASE ---
    for step in range(1, gen_config.max_new_tokens):
        t_step = time.perf_counter()
        with torch.no_grad():
            if gen_config.use_cache:
                step_logits = model(
                    next_token,
                    start_pos=prompt_len + step - 1,
                    kv_cache=kv_cache,
                    use_cache=True,
                )
                decode_logits = step_logits[:, -1, :]
            else:
                step_logits = model(current_tokens, start_pos=0, use_cache=False)
                decode_logits = step_logits[:, -1, :]

        next_token = sample_next_token(decode_logits, gen_config, current_tokens)
        step_time = time.perf_counter() - t_step
        current_tokens = torch.cat([current_tokens, next_token], dim=-1)

        yield next_token, step_time, False

        # Stop if all batches generated eos_token
        if gen_config.eos_token_id is not None:
            if (next_token == gen_config.eos_token_id).all():
                break


def generate(
    model: CausalLM,
    input_ids: torch.Tensor,
    gen_config: Optional[GenerationConfig] = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Generate complete output sequence and return profiling performance metrics."""
    if gen_config is None:
        gen_config = GenerationConfig()

    tokens = []
    decode_times = []
    ttft = 0.0

    t_start = time.perf_counter()
    for token, step_time, is_prefill in stream_generate(model, input_ids, gen_config):
        tokens.append(token)
        if is_prefill:
            ttft = step_time
        else:
            decode_times.append(step_time)

    total_time = time.perf_counter() - t_start
    generated_ids = torch.cat(tokens, dim=-1)
    num_generated = generated_ids.shape[-1]

    avg_tpot = (sum(decode_times) / len(decode_times)) if decode_times else 0.0
    tokens_per_second = num_generated / total_time if total_time > 0 else 0.0

    full_output = torch.cat([input_ids, generated_ids], dim=-1)
    metrics = {
        "ttft_ms": ttft * 1000.0,
        "avg_tpot_ms": avg_tpot * 1000.0,
        "total_latency_ms": total_time * 1000.0,
        "tokens_per_second": tokens_per_second,
        "tokens_generated": float(num_generated),
    }

    return full_output, metrics

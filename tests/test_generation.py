import torch

from llm_lab.config import GenerationConfig, ModelConfig
from llm_lab.engine.generation import (
    apply_repetition_penalty,
    generate,
    stream_generate,
    top_k_top_p_filtering,
)
from llm_lab.models.transformer import CausalLM


def test_top_k_top_p_filtering():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    # Top-K = 2 should keep only the top 2 elements (4.0 and 5.0)
    filtered = top_k_top_p_filtering(logits.clone(), top_k=2)
    assert filtered[0, 4] == 5.0
    assert filtered[0, 3] == 4.0
    assert filtered[0, 2] == -float("Inf")
    assert filtered[0, 1] == -float("Inf")
    assert filtered[0, 0] == -float("Inf")


def test_repetition_penalty():
    logits = torch.tensor([[2.0, 4.0, 1.0]])
    context = torch.tensor([[1]])  # token 1 was generated previously

    penalized = apply_repetition_penalty(logits, context, penalty=2.0)
    # Token 1 logit should be halved from 4.0 to 2.0
    assert penalized[0, 1].item() == 2.0
    # Other tokens should remain unchanged
    assert penalized[0, 0].item() == 2.0
    assert penalized[0, 2].item() == 1.0


def test_greedy_generation_determinism():
    cfg = ModelConfig.nano()
    model = CausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    gen_cfg = GenerationConfig(max_new_tokens=10, do_sample=False)

    out1, m1 = generate(model, input_ids, gen_cfg)
    out2, m2 = generate(model, input_ids, gen_cfg)

    assert torch.equal(out1, out2)
    assert out1.shape == (1, 8 + 10)
    assert m1["tokens_generated"] == 10.0
    assert m1["ttft_ms"] > 0


def test_stream_generation():
    cfg = ModelConfig.nano()
    model = CausalLM(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))

    gen_cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
    tokens = []
    is_prefill_list = []

    for tok, step_time, is_prefill in stream_generate(model, input_ids, gen_cfg):
        tokens.append(tok)
        is_prefill_list.append(is_prefill)
        assert step_time > 0

    assert len(tokens) == 5
    assert is_prefill_list == [True, False, False, False, False]

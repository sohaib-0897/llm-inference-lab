import torch

from llm_lab.config import GenerationConfig, ModelConfig
from llm_lab.engine.speculative import SpeculativeDecoder
from llm_lab.models.transformer import CausalLM


def test_speculative_decoding_execution():
    target_cfg = ModelConfig.nano()
    draft_cfg = ModelConfig.nano()
    draft_cfg.d_model = 64
    draft_cfg.n_heads = 2
    draft_cfg.n_kv_heads = 1
    draft_cfg.n_layers = 2
    draft_cfg.vocab_size = target_cfg.vocab_size

    target_model = CausalLM(target_cfg).eval()
    draft_model = CausalLM(draft_cfg).eval()

    decoder = SpeculativeDecoder(target_model, draft_model, gamma=2)

    prompt = torch.randint(0, target_cfg.vocab_size, (1, 4))
    gen_cfg = GenerationConfig(max_new_tokens=6, do_sample=False)

    output, metrics = decoder.generate(prompt, gen_cfg)

    assert output.shape[0] == 1
    assert output.shape[1] >= 4 + 6
    assert metrics["tokens_generated"] >= 6
    assert 0.0 <= metrics["acceptance_rate"] <= 1.0
    assert metrics["target_evaluations"] > 0

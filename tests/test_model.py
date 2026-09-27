import pytest
import torch

from llm_lab.config import ModelConfig
from llm_lab.models.attention import (
    apply_rotary_emb,
    precompute_rope_freqs_cis,
)
from llm_lab.models.transformer import CausalLM, RMSNorm


def test_model_config_validation():
    # Valid config
    cfg = ModelConfig.nano()
    assert cfg.head_dim == 32
    assert cfg.n_heads == 4
    assert cfg.n_kv_heads == 2

    # Invalid heads
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=128, n_heads=5)

    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=128, n_heads=8, n_kv_heads=3)


def test_rmsnorm():
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim)
    out = norm(x)
    assert out.shape == x.shape
    # Check that variance along last dim is close to 1
    var = out.pow(2).mean(-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_rope_embeddings():
    dim = 32
    seq_len = 16
    cos, sin = precompute_rope_freqs_cis(dim, seq_len)
    assert cos.shape == (seq_len, dim // 2)
    assert sin.shape == (seq_len, dim // 2)

    x = torch.randn(2, 4, seq_len, dim)
    rx = apply_rotary_emb(x, cos, sin, start_pos=0)
    assert rx.shape == x.shape

    # Offset test
    rx_offset = apply_rotary_emb(x[:, :, :4, :], cos, sin, start_pos=2)
    assert rx_offset.shape == (2, 4, 4, dim)


def test_causal_lm_forward():
    cfg = ModelConfig.nano()
    model = CausalLM(cfg).eval()

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids, start_pos=0, use_cache=False)

    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert not torch.isnan(logits).any()


def test_parameter_count():
    cfg = ModelConfig.nano()
    model = CausalLM(cfg)
    counts = model.count_parameters()
    assert "total" in counts
    assert "trainable" in counts
    assert counts["total"] > 0
    assert counts["trainable"] == counts["total"]

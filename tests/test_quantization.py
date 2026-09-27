import torch
import torch.nn as nn

from llm_lab.config import ModelConfig
from llm_lab.models.transformer import CausalLM
from llm_lab.quantization.quantized_linear import (
    QuantizedLinear,
    calculate_model_weight_bytes,
    quantize_model,
)


def test_quantized_linear_layer():
    linear = nn.Linear(32, 64, bias=True)
    q_layer = QuantizedLinear.from_float(linear)

    assert q_layer.weight_int8.dtype == torch.int8
    assert q_layer.weight_int8.shape == (64, 32)
    assert q_layer.scales.shape == (64, 1)

    x = torch.randn(2, 5, 32)
    out_orig = linear(x)
    out_q = q_layer(x)

    assert out_q.shape == out_orig.shape
    # Check that quantization error is small (cosine similarity > 0.98)
    cos_sim = torch.nn.functional.cosine_similarity(out_orig.flatten(), out_q.flatten(), dim=0)
    assert cos_sim.item() > 0.98


def test_quantize_full_model():
    cfg = ModelConfig.nano()
    model = CausalLM(cfg).eval()

    orig_bytes, _ = calculate_model_weight_bytes(model)
    quantize_model(model)
    q_bytes, _ = calculate_model_weight_bytes(model)

    # Weights should be significantly reduced
    assert q_bytes < orig_bytes

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, 8, cfg.vocab_size)
    assert not torch.isnan(logits).any()

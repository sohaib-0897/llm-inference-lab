import pytest
import torch

from llm_lab.config import ModelConfig
from llm_lab.engine.kv_cache import KVCache, LayerKVCache
from llm_lab.models.transformer import CausalLM


def test_layer_kv_cache_update():
    layer = LayerKVCache()
    k1 = torch.randn(1, 2, 4, 32)
    v1 = torch.randn(1, 2, 4, 32)

    k_out, v_out = layer.update(k1, v1)
    assert k_out.shape == (1, 2, 4, 32)
    assert v_out.shape == (1, 2, 4, 32)
    assert layer.seq_len == 4

    k2 = torch.randn(1, 2, 1, 32)
    v2 = torch.randn(1, 2, 1, 32)
    k_out2, v_out2 = layer.update(k2, v2)
    assert k_out2.shape == (1, 2, 5, 32)
    assert v_out2.shape == (1, 2, 5, 32)
    assert layer.seq_len == 5

    layer.reset()
    assert layer.key is None
    assert layer.seq_len == 0


def test_kv_cache_memory():
    cache = KVCache(num_layers=4)
    for layer in cache.layers:
        layer.update(torch.randn(1, 2, 16, 32), torch.randn(1, 2, 16, 32))

    # 4 layers * 2 tensors * (1 * 2 * 16 * 32) elements * 4 bytes
    expected_bytes = 4 * 2 * (1 * 2 * 16 * 32) * 4
    assert cache.total_memory_bytes == expected_bytes
    assert cache.total_memory_mb == expected_bytes / (1024 * 1024)

    theo_mb = KVCache.theoretical_memory_mb(
        batch_size=1, n_layers=4, n_kv_heads=2, seq_len=16, head_dim=32, dtype_bytes=4
    )
    assert pytest.approx(cache.total_memory_mb) == theo_mb


def test_kv_cache_numerical_equivalence():
    """Verify that cached autoregressive decoding produces mathematically identical

    logits to full sequence recomputation without cache.
    """
    torch.manual_seed(42)
    cfg = ModelConfig.nano()
    model = CausalLM(cfg).eval()

    prompt_len = 5
    decode_steps = 3
    total_tokens = prompt_len + decode_steps

    input_ids = torch.randint(0, cfg.vocab_size, (1, total_tokens))

    # --- Baseline: Full recomputation without cache ---
    with torch.no_grad():
        full_logits = model(input_ids, start_pos=0, use_cache=False)
        target_last_logits = full_logits[:, -1, :]

    # --- Cached Execution: Prefill + Step-by-Step Decoding ---
    cache = model.allocate_kv_cache()
    with torch.no_grad():
        # Prefill phase
        prefill_ids = input_ids[:, :prompt_len]
        _ = model(prefill_ids, start_pos=0, kv_cache=cache, use_cache=True)

        # Step-by-step decode
        step_logits = None
        for i in range(decode_steps):
            next_single_token = input_ids[:, prompt_len + i : prompt_len + i + 1]
            pos = prompt_len + i
            step_logits = model(
                next_single_token,
                start_pos=pos,
                kv_cache=cache,
                use_cache=True,
            )

    cached_last_logits = step_logits[:, -1, :]

    # Check maximum absolute difference
    max_diff = torch.max(torch.abs(target_last_logits - cached_last_logits)).item()
    assert max_diff < 1e-4, f"KV cache discrepancy too high: {max_diff}"

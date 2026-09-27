from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_lab.config import ModelConfig
from llm_lab.engine.kv_cache import KVCache
from llm_lab.models.attention import (
    GroupedQueryAttention,
    precompute_rope_freqs_cis,
)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm formula: x / sqrt(mean(x^2) + eps) * weight
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward layer as used in LLaMA / Mistral architectures."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d_ff = config.d_ff or int(8 * config.d_model / 3)
        self.w1 = nn.Linear(config.d_model, d_ff, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_ff, config.d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(config.d_model, d_ff, bias=False)  # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single decoder block with pre-norm RMSNorm, GQA, and SwiGLU FFN."""

    def __init__(self, layer_id: int, config: ModelConfig) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.feed_forward = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # Attention with residual
        norm_x = self.attn_norm(x)
        layer_cache = kv_cache[self.layer_id] if (use_cache and kv_cache is not None) else None
        h = x + self.attention(
            norm_x,
            cos=cos,
            sin=sin,
            start_pos=start_pos,
            kv_cache=layer_cache,
            use_cache=use_cache,
        )

        # Feed-forward with residual
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class CausalLM(nn.Module):
    """Modern Autoregressive Causal Language Model."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # Precompute RoPE tables
        cos, sin = precompute_rope_freqs_cis(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(i, config) for i in range(config.n_layers)]
        )

        # Final norm and LM head
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass through causal language model.

        Args:
            input_ids: (batch_size, seq_len)
            start_pos: Position index for rotary embeddings
            kv_cache: Optional multi-layer KV cache instance
            use_cache: Whether to use and update KV cache

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        _, seq_len = input_ids.shape
        h = self.tok_embeddings(input_ids)

        cos: torch.Tensor = self.cos
        sin: torch.Tensor = self.sin

        for layer in self.layers:
            h = layer(
                h,
                cos=cos,
                sin=sin,
                start_pos=start_pos,
                kv_cache=kv_cache,
                use_cache=use_cache,
            )

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    def count_parameters(self) -> dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def allocate_kv_cache(self) -> KVCache:
        """Convenience method to construct an empty KVCache for this model."""
        return KVCache(self.config.n_layers)

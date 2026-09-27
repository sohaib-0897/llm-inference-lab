from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_lab.config import ModelConfig
from llm_lab.engine.kv_cache import LayerKVCache


def precompute_rope_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine frequencies for Rotary Positional Embeddings (RoPE)."""
    half_dim = dim // 2
    freq_indices = torch.arange(0, half_dim, dtype=torch.float32, device=device)
    freqs = 1.0 / (base ** (2.0 * freq_indices / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs_matrix = torch.outer(t, freqs)  # (max_seq_len, half_dim)
    cos = torch.cos(freqs_matrix)
    sin = torch.sin(freqs_matrix)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Apply Rotary Position Embedding to input tensor x.

    Args:
        x: Shape (batch, n_heads, seq_len, head_dim)
        cos: Shape (max_seq_len, head_dim // 2)
        sin: Shape (max_seq_len, head_dim // 2)
        start_pos: Offset for autoregressive generation steps with KV cache

    Returns:
        Tensor of shape (batch, n_heads, seq_len, head_dim) with RoPE applied.
    """
    seq_len = x.shape[2]
    # Slice cos and sin for the target position range: [start_pos, start_pos + seq_len]
    cur_cos = cos[start_pos : start_pos + seq_len, :].to(dtype=x.dtype, device=x.device)
    cur_sin = sin[start_pos : start_pos + seq_len, :].to(dtype=x.dtype, device=x.device)

    # Broadcast shapes to (1, 1, seq_len, head_dim // 2)
    cur_cos = cur_cos.unsqueeze(0).unsqueeze(0)
    cur_sin = cur_sin.unsqueeze(0).unsqueeze(0)

    # Split x into first half and second half
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # Apply 2D rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    rx1 = x1 * cur_cos - x2 * cur_sin
    rx2 = x1 * cur_sin + x2 * cur_cos
    return torch.cat([rx1, rx2], dim=-1)


class GroupedQueryAttention(nn.Module):
    """Multi-Head and Grouped-Query Attention (GQA) layer with KV cache support."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.head_dim = config.head_dim
        self.num_rep = self.n_heads // self.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[LayerKVCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass for GQA.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            cos: Precomputed cosine frequencies
            sin: Precomputed sine frequencies
            start_pos: Sequence position offset for RoPE and attention masks
            kv_cache: Optional LayerKVCache instance
            use_cache: Whether to store/retrieve key/values from kv_cache

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary Positional Embeddings
        q = apply_rotary_emb(q, cos, sin, start_pos=start_pos)
        k = apply_rotary_emb(k, cos, sin, start_pos=start_pos)

        # Manage KV Cache
        if use_cache and kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Total key sequence length after cache update
        total_k_len = k.shape[2]

        # Expand KV heads if Grouped-Query Attention (num_rep > 1)
        if self.num_rep > 1:
            k = torch.repeat_interleave(k, repeats=self.num_rep, dim=1)
            v = torch.repeat_interleave(v, repeats=self.num_rep, dim=1)

        # Compute Attention Scores
        # q: (batch, n_heads, q_len, head_dim)
        # k: (batch, n_heads, k_len, head_dim)
        # scores: (batch, n_heads, q_len, k_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask when computing multi-token queries
        if seq_len > 1:
            # Create lower triangular causal mask
            mask = torch.full(
                (seq_len, total_k_len),
                float("-inf"),
                device=x.device,
                dtype=scores.dtype,
            )
            # Query i at position start_pos + i can attend to keys up to start_pos + i
            mask = torch.triu(mask, diagonal=start_pos + 1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        # Softmax and context aggregation
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)

        # Transpose and flatten back to (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)

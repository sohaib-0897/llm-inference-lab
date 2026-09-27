from __future__ import annotations

from typing import Optional, Tuple

import torch


class LayerKVCache:
    """Manages Key and Value cache tensors for a single transformer layer."""

    def __init__(self) -> None:
        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None

    def update(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value states along sequence dimension (dim=2).

        Args:
            new_key: Shape (batch_size, n_kv_heads, seq_len, head_dim)
            new_value: Shape (batch_size, n_kv_heads, seq_len, head_dim)

        Returns:
            Tuple of updated (full_key, full_value)
        """
        if self.key is None or self.value is None:
            self.key = new_key
            self.value = new_value
        else:
            self.key = torch.cat([self.key, new_key], dim=2)
            self.value = torch.cat([self.value, new_value], dim=2)

        return self.key, self.value

    def reset(self) -> None:
        self.key = None
        self.value = None

    @property
    def seq_len(self) -> int:
        if self.key is None:
            return 0
        return self.key.shape[2]

    @property
    def memory_bytes(self) -> int:
        mem = 0
        if self.key is not None:
            mem += self.key.nelement() * self.key.element_size()
        if self.value is not None:
            mem += self.value.nelement() * self.value.element_size()
        return mem


class KVCache:
    """Multi-layer Key-Value cache manager for transformer decoders."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.layers: list[LayerKVCache] = [LayerKVCache() for _ in range(num_layers)]

    def __getitem__(self, layer_idx: int) -> LayerKVCache:
        return self.layers[layer_idx]

    def reset(self) -> None:
        """Clear all stored key/value states."""
        for layer in self.layers:
            layer.reset()

    @property
    def current_seq_len(self) -> int:
        """Current sequence length stored in the cache (based on layer 0)."""
        if not self.layers or self.layers[0].key is None:
            return 0
        return self.layers[0].seq_len

    @property
    def total_memory_bytes(self) -> int:
        """Total memory consumed by cache across all layers in bytes."""
        return sum(layer.memory_bytes for layer in self.layers)

    @property
    def total_memory_mb(self) -> float:
        """Total memory consumed by cache across all layers in megabytes."""
        return self.total_memory_bytes / (1024 * 1024)

    @staticmethod
    def theoretical_memory_mb(
        batch_size: int,
        n_layers: int,
        n_kv_heads: int,
        seq_len: int,
        head_dim: int,
        dtype_bytes: int = 4,  # 4 for float32, 2 for float16
    ) -> float:
        """Calculate theoretical memory required for KV cache in MB."""
        # 2 tensors (K and V) * layers * batch * n_kv_heads * seq_len * head_dim * bytes_per_elem
        total_elements = 2 * n_layers * batch_size * n_kv_heads * seq_len * head_dim
        return (total_elements * dtype_bytes) / (1024 * 1024)

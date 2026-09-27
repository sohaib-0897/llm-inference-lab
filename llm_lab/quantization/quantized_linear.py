from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from llm_lab.models.transformer import CausalLM


class QuantizedLinear(nn.Module):
    """Weight-only symmetric INT8 quantized linear layer.

    Stores weights as int8 (1 byte per parameter vs 4 bytes in float32),
    achieving 4x weight compression. Scales are stored per output channel.
    """

    weight_int8: torch.Tensor
    scales: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones((out_features, 1), dtype=torch.float32))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, float_linear: nn.Linear) -> QuantizedLinear:
        """Quantize standard nn.Linear float32 layer into QuantizedLinear INT8 layer."""
        q_layer = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=float_linear.bias is not None,
        )

        with torch.no_grad():
            w = float_linear.weight.float()
            # Per-channel symmetric quantization:
            # scale = max(|w|, dim=-1) / 127
            max_val = torch.max(torch.abs(w), dim=-1, keepdim=True).values.clamp(min=1e-8)
            scales = max_val / 127.0
            w_q = torch.clamp(torch.round(w / scales), -128, 127).to(torch.int8)

            q_layer.weight_int8.copy_(w_q)
            q_layer.scales.copy_(scales)

            if float_linear.bias is not None and q_layer.bias is not None:
                q_layer.bias.copy_(float_linear.bias.float())

        return q_layer

    def dequantize(self) -> torch.Tensor:
        """Dequantize int8 weights to float32."""
        return self.weight_int8.float() * self.scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using dequantized weights on-the-fly for CPU execution."""
        w_float = self.dequantize()
        return torch.nn.functional.linear(x, w_float, self.bias)

    @property
    def weight_memory_bytes(self) -> int:
        """Memory consumed by stored weights and scales."""
        int8_bytes = self.weight_int8.nelement() * self.weight_int8.element_size()
        scale_bytes = self.scales.nelement() * self.scales.element_size()
        bias_bytes = (self.bias.nelement() * self.bias.element_size()) if self.bias is not None else 0
        return int8_bytes + scale_bytes + bias_bytes


def quantize_model(model: CausalLM) -> CausalLM:
    """Recursively replaces all nn.Linear projection layers in CausalLM with INT8 QuantizedLinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name != "lm_head":
            # Keep lm_head in fp32 for classification stability, quantize interior linear layers
            setattr(model, name, QuantizedLinear.from_float(module))
        elif isinstance(module, nn.Module):
            _quantize_children(module)
    return model


def _quantize_children(parent: nn.Module) -> None:
    for name, child in parent.named_children():
        if isinstance(child, nn.Linear):
            setattr(parent, name, QuantizedLinear.from_float(child))
        else:
            _quantize_children(child)


def calculate_model_weight_bytes(model: nn.Module) -> Tuple[int, int]:
    """Calculate total parameter memory bytes and parameter count."""
    total_bytes = 0
    param_count = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
        param_count += p.nelement()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes, param_count

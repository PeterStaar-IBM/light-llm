"""
Feed-forward network (FFN) sub-layers.

  SwiGLU – gated linear unit with SiLU activation (default, used in LLaMA/Mistral).
  GeGLU  – gated linear unit with GELU activation.
  MLP    – standard 2-layer MLP with configurable activation.

All variants use two projections for gated variants (gate + up) or one for MLP,
followed by a down projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from light_llm.models.transformer.config import TransformerConfig


class SwiGLU(nn.Module):
    """
    SwiGLU: gate(x) = SiLU(gate_proj(x)) * up_proj(x)
    Output  = down_proj(gate(x))

    Parameter count is ~equivalent to a 4× MLP when expansion_factor ≈ 8/3.
    """

    def __init__(self, d_model: int, ffn_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.up_proj   = nn.Linear(d_model, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GeGLU(nn.Module):
    """
    GeGLU: gate(x) = GELU(gate_proj(x)) * up_proj(x)
    """

    def __init__(self, d_model: int, ffn_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.up_proj   = nn.Linear(d_model, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):
    """Standard 2-layer MLP: Linear → Activation → Dropout → Linear."""

    _ACTIVATIONS = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
    }

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation!r}")
        self.fc1  = nn.Linear(d_model, ffn_dim, bias=bias)
        self.fc2  = nn.Linear(ffn_dim, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.act  = self._ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


def build_ffn(cfg: TransformerConfig) -> nn.Module:
    f = cfg.ffn
    ffn_dim = cfg.ffn_dim
    match f.variant:
        case "swiglu":
            return SwiGLU(cfg.d_model, ffn_dim, bias=f.bias)
        case "geglu":
            return GeGLU(cfg.d_model, ffn_dim, bias=f.bias)
        case "mlp":
            return MLP(cfg.d_model, ffn_dim, f.activation, f.dropout, f.bias)
        case _:
            raise ValueError(f"Unknown FFN variant: {f.variant!r}")

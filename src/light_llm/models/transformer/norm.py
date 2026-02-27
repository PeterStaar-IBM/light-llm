"""
Normalisation layers.

  RMSNorm  – modern default (no mean subtraction, lighter than LayerNorm)
  LayerNorm – classic; optionally without bias
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from light_llm.models.transformer.config import NormConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # Cast to float32 for numerical stability, then restore dtype
        orig_dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(orig_dtype)


class LayerNorm(nn.Module):
    """Standard Layer Normalisation with optional bias."""

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=True)
        if not bias:
            # Zero out the bias parameter so it has no effect
            self.norm.bias = None  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


def build_norm(d_model: int, cfg: NormConfig) -> nn.Module:
    match cfg.type:
        case "rmsnorm":
            return RMSNorm(d_model, eps=cfg.eps)
        case "layernorm":
            return LayerNorm(d_model, eps=cfg.eps)
        case _:
            raise ValueError(f"Unknown norm type: {cfg.type!r}")

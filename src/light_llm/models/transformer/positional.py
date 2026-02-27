"""
Positional encoding strategies.

  RoPE        – Rotary Position Embedding (Su et al., 2021).  Modern default.
  ALiBi       – Attention with Linear Biases (Press et al., 2021).
  Sinusoidal  – Classic fixed sinusoidal (Vaswani et al., 2017).
  Learned     – Learnable absolute position embeddings.

Usage
-----
RoPE and ALiBi are applied *inside* the attention module (see attention.py).
Sinusoidal and Learned embeddings are added to the token embeddings before
the first transformer layer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from light_llm.models.transformer.config import PositionalConfig


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    RoPE: rotates Q and K by position-dependent angles.

    Supports:
      - Standard RoPE (base=10_000)
      - Extended context via linear scaling (rope_scaling > 1)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 4096,
        base: float = 10_000.0,
        scaling: Optional[float] = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.base = base
        self.scaling = scaling

        # Precompute frequency table; register as buffer so it moves with .to(device)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()  # type: ignore[attr-defined]
        if self.scaling is not None:
            t = t / self.scaling
        freqs = torch.outer(t, self.inv_freq)  # [seq, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
        self.register_buffer("cos_cache", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cache", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)
        self._cached_seq_len = seq_len

    def _ensure_cache(self, seq_len: int) -> None:
        if seq_len > self._cached_seq_len:
            self._build_cache(seq_len)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply(self, q: Tensor, k: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: [batch, num_heads, seq, head_dim]
            k: [batch, num_kv_heads, seq, head_dim]
            offset: starting position (for KV-cache incremental decoding)
        """
        seq = q.shape[2]
        self._ensure_cache(offset + seq)
        cos = self.cos_cache[:, :, offset : offset + seq, :]  # type: ignore[index]
        sin = self.sin_cache[:, :, offset : offset + seq, :]  # type: ignore[index]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# ALiBi (Attention with Linear Biases)
# ---------------------------------------------------------------------------

class ALiBiEmbedding(nn.Module):
    """
    ALiBi adds a per-head linear penalty to attention logits based on distance.
    The bias matrix is NOT learned; only the slopes differ per head.
    """

    def __init__(self, num_heads: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        slopes = self._get_slopes(num_heads)  # [num_heads]
        # Precompute bias: slopes * relative positions
        pos = torch.arange(max_seq_len)
        # bias[h, i, j] = -slope_h * (i - j) for j <= i
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)  # [seq, seq]
        bias = -slopes.unsqueeze(-1).unsqueeze(-1) * rel.unsqueeze(0).abs()
        # [num_heads, max_seq, max_seq]
        self.register_buffer("bias", bias, persistent=False)
        self._max_seq = max_seq_len

    @staticmethod
    def _get_slopes(n: int) -> Tensor:
        def _pow2_slopes(m: int) -> list[float]:
            start = 2 ** (-(2 ** -(math.log2(m) - 3)))
            return [start * (start**i) for i in range(m)]

        if math.log2(n) == int(math.log2(n)):
            return torch.tensor(_pow2_slopes(n))
        # Non-power-of-2: interpolate
        floor = 2 ** math.floor(math.log2(n))
        slopes = _pow2_slopes(floor)
        extra = _pow2_slopes(2 * floor)[0::2][: n - floor]
        return torch.tensor(slopes + extra)

    def get_bias(self, seq_len: int, offset: int = 0) -> Tensor:
        """Return [1, num_heads, seq_len, seq_len + offset] bias tensor."""
        end = offset + seq_len
        if end > self._max_seq:
            # Extend cache on the fly
            self._rebuild(end)
        return self.bias[:, offset:end, offset:end].unsqueeze(0)  # type: ignore[index]

    def _rebuild(self, new_max: int) -> None:
        n = self.bias.shape[0]  # type: ignore[union-attr]
        slopes = self.bias[:, 0, 1].neg() / 1  # recover slopes
        pos = torch.arange(new_max, device=self.bias.device)  # type: ignore[union-attr]
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)
        bias = -slopes.view(-1, 1, 1) * rel.unsqueeze(0).abs()
        self.register_buffer("bias", bias, persistent=False)
        self._max_seq = new_max


# ---------------------------------------------------------------------------
# Sinusoidal (fixed, added to embeddings)
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Non-learned sinusoidal position embeddings (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq = x.shape[1]
        return x + self.pe[:, offset : offset + seq]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Learned absolute position embeddings
# ---------------------------------------------------------------------------

class LearnedEmbedding(nn.Module):
    """Learnable absolute position embeddings."""

    def __init__(self, d_model: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq = x.shape[1]
        pos = torch.arange(offset, offset + seq, device=x.device)
        return x + self.emb(pos)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_positional(
    d_model: int,
    head_dim: int,
    num_heads: int,
    max_seq_len: int,
    cfg: PositionalConfig,
) -> nn.Module | None:
    """
    Returns the appropriate positional module (or None for RoPE/ALiBi which
    are handled inside the attention layer).

    The caller should check the encoding type and wire accordingly:
      - "rope"       → pass RotaryEmbedding to attention
      - "alibi"      → pass ALiBiEmbedding to attention
      - "sinusoidal" → call module(x) on embeddings
      - "learned"    → call module(x) on embeddings
      - "none"       → no-op
    """
    match cfg.encoding:
        case "rope":
            return RotaryEmbedding(head_dim, max_seq_len, cfg.rope_base, cfg.rope_scaling)
        case "alibi":
            return ALiBiEmbedding(num_heads, max_seq_len)
        case "sinusoidal":
            return SinusoidalEmbedding(d_model, max_seq_len)
        case "learned":
            return LearnedEmbedding(d_model, max_seq_len)
        case "none":
            return None
        case _:
            raise ValueError(f"Unknown positional encoding: {cfg.encoding!r}")

"""
Attention implementations.

  Attention        – Standard real-valued MHA / GQA / MQA with RoPE or ALiBi.
  ComplexAttention – Complex-valued multi-head attention (experimental).

KV-cache is supported in both via an optional `past_kv` argument.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from light_llm.models.transformer.config import TransformerConfig
from light_llm.models.transformer.positional import ALiBiEmbedding, RotaryEmbedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _causal_mask(seq_q: int, seq_k: int, device: torch.device) -> Tensor:
    """Upper-triangular mask (True = ignore)."""
    return torch.ones(seq_q, seq_k, device=device, dtype=torch.bool).triu(seq_k - seq_q + 1)


def _sliding_window_mask(
    seq_q: int, seq_k: int, window: int, device: torch.device
) -> Tensor:
    """Causal mask that also blocks positions > window tokens in the past."""
    row = torch.arange(seq_q, device=device).unsqueeze(1)
    col = torch.arange(seq_k, device=device).unsqueeze(0)
    future = col > row + (seq_k - seq_q)          # future positions
    too_far = (row + (seq_k - seq_q)) - col > window  # beyond window
    return future | too_far


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads to match Q heads for GQA."""
    if n_rep == 1:
        return x
    # x: [batch, kv_heads, seq, head_dim]
    return x.repeat_interleave(n_rep, dim=1)


# ---------------------------------------------------------------------------
# Real-valued Attention (MHA / GQA / MQA)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Configurable multi-head attention.

    Supports:
      - MHA  (num_kv_heads == num_heads)
      - GQA  (1 < num_kv_heads < num_heads)
      - MQA  (num_kv_heads == 1)
      - Causal, bidirectional, and sliding-window attention masks
      - RoPE or ALiBi positional encoding
      - Optional Flash Attention (requires flash-attn)
      - KV cache for autoregressive inference
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        rope: Optional[RotaryEmbedding] = None,
        alibi: Optional[ALiBiEmbedding] = None,
    ) -> None:
        super().__init__()
        a = cfg.attention
        self.num_heads = a.num_heads
        self.num_kv_heads = a.num_kv_heads  # type: ignore[assignment]
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.mask_type = a.mask_type
        self.window_size = a.window_size
        self.use_flash = a.use_flash_attn
        self.n_rep = self.num_heads // self.num_kv_heads

        self.rope = rope
        self.alibi = alibi

        d = cfg.d_model
        bias = a.bias
        self.q_proj = nn.Linear(d, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d, bias=bias)

        self.attn_drop = nn.Dropout(a.dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[tuple[Tensor, Tensor]] = None,
        return_kv: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            x:        [batch, seq, d_model]
            past_kv:  optional tuple of (k, v) each [batch, kv_heads, past_seq, head_dim]
            return_kv: whether to return updated (k, v) cache
        Returns:
            output [batch, seq, d_model], and optionally (k, v)
        """
        B, T, _ = x.shape

        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.num_kv_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.num_kv_heads)

        offset = past_kv[0].shape[2] if past_kv is not None else 0

        # Apply RoPE
        if self.rope is not None:
            q, k = self.rope.apply(q, k, offset=offset)

        # Append past KV
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv: Optional[tuple[Tensor, Tensor]] = (k, v) if return_kv else None

        # Expand KV heads for GQA/MQA
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        seq_q, seq_k = T, k.shape[2]

        if self.use_flash:
            out = self._flash_forward(q, k, v, seq_q, seq_k)
        else:
            out = self._sdpa_forward(q, k, v, seq_q, seq_k, offset)

        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.out_proj(out)

        if return_kv:
            return out, new_kv  # type: ignore[return-value]
        return out

    # ------------------------------------------------------------------

    def _sdpa_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        seq_q: int,
        seq_k: int,
        offset: int,
    ) -> Tensor:
        """Scaled dot-product attention (PyTorch built-in, fused when possible)."""
        attn_bias: Optional[Tensor] = None

        if self.alibi is not None:
            attn_bias = self.alibi.get_bias(seq_q, offset).to(q.dtype)

        if self.mask_type == "causal":
            mask = _causal_mask(seq_q, seq_k, q.device)
        elif self.mask_type == "sliding_window":
            window = self.window_size or seq_k
            mask = _sliding_window_mask(seq_q, seq_k, window, q.device)
        else:
            mask = None

        # F.scaled_dot_product_attention handles fused attention on modern GPUs
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            is_causal=(self.mask_type == "causal" and attn_bias is None),
            dropout_p=self.attn_drop.p if self.training else 0.0,
        ) if mask is None and attn_bias is None else self._manual_sdpa(
            q, k, v, mask, attn_bias
        )

    def _manual_sdpa(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        attn_bias: Optional[Tensor],
    ) -> Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)
        return torch.matmul(weights, v)

    def _flash_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        seq_q: int,
        seq_k: int,
    ) -> Tensor:
        try:
            from flash_attn import flash_attn_func  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "flash-attn is not installed. Run: pip install flash-attn"
            ) from e
        # flash_attn expects [batch, seq, heads, head_dim]
        q = rearrange(q, "b h t d -> b t h d")
        k = rearrange(k, "b h t d -> b t h d")
        v = rearrange(v, "b h t d -> b t h d")
        causal = self.mask_type == "causal"
        out = flash_attn_func(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, causal=causal)
        return rearrange(out, "b t h d -> b h t d")


# ---------------------------------------------------------------------------
# Complex-valued Attention (experimental)
# ---------------------------------------------------------------------------

class ComplexAttention(nn.Module):
    """
    Complex-valued multi-head attention.

    Each query and key lives in complex space: q = q_re + i·q_im.
    The attention score uses the Hermitian inner product:
        score_ij = Re(<q_i, k_j>) = q_re · k_re + q_im · k_im

    Values can optionally be complex too (controlled by `complex_values`).
    The final output is the real part (or magnitude) of the weighted sum.

    This is useful for learning phase/frequency-based representations and
    can be combined with complex rotary embeddings.
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        complex_values: bool = False,
        rope: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        a = cfg.attention
        self.num_heads = a.num_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.mask_type = a.mask_type
        self.complex_values = complex_values
        self.rope = rope

        d = cfg.d_model
        bias = a.bias

        # Project real input → complex Q, K (re + im parts)
        self.q_re = nn.Linear(d, self.num_heads * self.head_dim, bias=bias)
        self.q_im = nn.Linear(d, self.num_heads * self.head_dim, bias=bias)
        self.k_re = nn.Linear(d, self.num_heads * self.head_dim, bias=bias)
        self.k_im = nn.Linear(d, self.num_heads * self.head_dim, bias=bias)

        if complex_values:
            self.v_re = nn.Linear(d, self.num_heads * self.head_dim, bias=bias)
            self.v_im = nn.Linear(d, self.num_heads * self.head_dim, bias=bias)
            self.out_proj = nn.Linear(self.num_heads * self.head_dim, d, bias=bias)
        else:
            self.v_proj = nn.Linear(d, self.num_heads * self.head_dim, bias=bias)
            self.out_proj = nn.Linear(self.num_heads * self.head_dim, d, bias=bias)

        self.attn_drop = nn.Dropout(a.dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[tuple[Tensor, Tensor]] = None,
        return_kv: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        def _split(proj: nn.Linear) -> Tensor:
            return rearrange(proj(x), "b t (h d) -> b h t d", h=H)

        q_re, q_im = _split(self.q_re), _split(self.q_im)
        k_re, k_im = _split(self.k_re), _split(self.k_im)

        # Optional: apply complex RoPE (phase rotation)
        if self.rope is not None:
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            # Treat (re, im) as the real/imaginary part of complex RoPE rotation
            q_re, q_im, k_re, k_im = self._apply_complex_rope(
                q_re, q_im, k_re, k_im, offset
            )

        # KV cache
        if past_kv is not None:
            k_re = torch.cat([past_kv[0], k_re], dim=2)
            k_im = torch.cat([past_kv[1], k_im], dim=2)

        new_kv: Optional[tuple[Tensor, Tensor]] = (k_re, k_im) if return_kv else None

        # Hermitian attention score: Re(<q, k>) = q_re·k_re + q_im·k_im
        scores = (
            torch.matmul(q_re, k_re.transpose(-2, -1))
            + torch.matmul(q_im, k_im.transpose(-2, -1))
        ) * self.scale

        seq_q, seq_k = T, k_re.shape[2]
        if self.mask_type == "causal":
            mask = _causal_mask(seq_q, seq_k, x.device)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        if self.complex_values:
            v_re = rearrange(self.v_re(x), "b t (h d) -> b h t d", h=H)
            v_im = rearrange(self.v_im(x), "b t (h d) -> b h t d", h=H)
            out_re = torch.matmul(weights, v_re)
            # Output is the real part; imaginary part is discarded
            out = rearrange(out_re, "b h t d -> b t (h d)")
        else:
            v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=H)
            out = rearrange(torch.matmul(weights, v), "b h t d -> b t (h d)")

        out = self.out_proj(out)

        if return_kv:
            return out, new_kv  # type: ignore[return-value]
        return out

    def _apply_complex_rope(
        self,
        q_re: Tensor,
        q_im: Tensor,
        k_re: Tensor,
        k_im: Tensor,
        offset: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Complex RoPE: treat (q_re, q_im) as a complex vector and rotate it
        by position-dependent complex exponentials e^{i·θ}.

        Multiplication by e^{iθ} = cosθ + i·sinθ:
            (a + ib)(cosθ + i·sinθ) = a·cosθ - b·sinθ + i(a·sinθ + b·cosθ)
        """
        assert self.rope is not None
        seq = q_re.shape[2]
        self.rope._ensure_cache(offset + seq)
        cos = self.rope.cos_cache[:, :, offset : offset + seq, :]  # type: ignore[index]
        sin = self.rope.sin_cache[:, :, offset : offset + seq, :]  # type: ignore[index]

        new_q_re = q_re * cos - q_im * sin
        new_q_im = q_re * sin + q_im * cos
        new_k_re = k_re * cos - k_im * sin
        new_k_im = k_re * sin + k_im * cos
        return new_q_re, new_q_im, new_k_re, new_k_im


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_attention(
    cfg: TransformerConfig,
    rope: Optional[RotaryEmbedding] = None,
    alibi: Optional[ALiBiEmbedding] = None,
) -> nn.Module:
    if cfg.attention.variant == "complex":
        return ComplexAttention(cfg, complex_values=False, rope=rope)
    return Attention(cfg, rope=rope, alibi=alibi)

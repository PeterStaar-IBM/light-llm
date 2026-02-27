"""
Transformer block (a single layer).

Supports:
  - Pre-norm  (default, modern)    : norm → sub-layer → residual
  - Post-norm (classic, BERT-style): sub-layer → residual → norm
  - Sandwich  (pre + post norm)    : norm → sub-layer → norm → residual
  - Parallel  (PaLM-style)         : attn and FFN computed in parallel,
                                     their outputs summed before residual
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from light_llm.models.transformer.attention import build_attention
from light_llm.models.transformer.config import TransformerConfig
from light_llm.models.transformer.ffn import build_ffn
from light_llm.models.transformer.norm import build_norm
from light_llm.models.transformer.positional import ALiBiEmbedding, RotaryEmbedding


class TransformerBlock(nn.Module):
    """
    One transformer layer with configurable norm placement and optional
    parallel (attention + FFN) computation.
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        rope: Optional[RotaryEmbedding] = None,
        alibi: Optional[ALiBiEmbedding] = None,
    ) -> None:
        super().__init__()
        self.parallel = cfg.parallel_layers
        self.placement = cfg.norm.placement

        self.attn = build_attention(cfg, rope=rope, alibi=alibi)
        self.ffn  = build_ffn(cfg)

        self.norm1 = build_norm(cfg.d_model, cfg.norm)
        self.norm2 = build_norm(cfg.d_model, cfg.norm)

        # Sandwich norm needs extra post-sub-layer norms
        if self.placement == "sandwich":
            self.post_norm1 = build_norm(cfg.d_model, cfg.norm)
            self.post_norm2 = build_norm(cfg.d_model, cfg.norm)
        else:
            self.post_norm1 = None  # type: ignore[assignment]
            self.post_norm2 = None  # type: ignore[assignment]

        self.drop = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[tuple[Tensor, Tensor]] = None,
        return_kv: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        if self.parallel:
            return self._parallel_forward(x, past_kv, return_kv)
        return self._sequential_forward(x, past_kv, return_kv)

    # ------------------------------------------------------------------

    def _sequential_forward(
        self,
        x: Tensor,
        past_kv: Optional[tuple[Tensor, Tensor]],
        return_kv: bool,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        placement = self.placement

        # --- Attention sub-layer ---
        if placement in ("pre", "sandwich"):
            residual = x
            x_norm = self.norm1(x)
        else:  # post
            residual = x
            x_norm = x

        attn_out = self.attn(x_norm, past_kv=past_kv, return_kv=return_kv)
        if return_kv:
            attn_out, new_kv = attn_out  # type: ignore[misc]
        else:
            new_kv = None

        if placement == "sandwich":
            attn_out = self.post_norm1(attn_out)  # type: ignore[arg-type]
        x = residual + self.drop(attn_out)
        if placement == "post":
            x = self.norm1(x)

        # --- FFN sub-layer ---
        if placement in ("pre", "sandwich"):
            residual = x
            x_in = self.norm2(x)
        else:
            residual = x
            x_in = x

        ffn_out = self.ffn(x_in)
        if placement == "sandwich":
            ffn_out = self.post_norm2(ffn_out)  # type: ignore[arg-type]
        x = residual + self.drop(ffn_out)
        if placement == "post":
            x = self.norm2(x)

        if return_kv:
            return x, new_kv  # type: ignore[return-value]
        return x

    def _parallel_forward(
        self,
        x: Tensor,
        past_kv: Optional[tuple[Tensor, Tensor]],
        return_kv: bool,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """PaLM-style: attn and FFN share the same pre-norm input."""
        x_norm = self.norm1(x)

        attn_out = self.attn(x_norm, past_kv=past_kv, return_kv=return_kv)
        if return_kv:
            attn_out, new_kv = attn_out  # type: ignore[misc]
        else:
            new_kv = None

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop(attn_out) + self.drop(ffn_out)

        if return_kv:
            return x, new_kv  # type: ignore[return-value]
        return x

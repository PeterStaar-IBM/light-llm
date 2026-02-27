"""
Full Transformer language model (decoder-only, GPT-style).

Features
--------
- Configurable via TransformerConfig (Pydantic)
- Supports all attention variants (MHA / GQA / MQA / Complex)
- Supports all positional encodings (RoPE / ALiBi / Sinusoidal / Learned / None)
- KV-cache for efficient autoregressive inference
- Weight tying between input and output embeddings (optional)
- Mixed-precision friendly (uses bfloat16/float16 via torch.autocast externally)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from light_llm.models.transformer.blocks import TransformerBlock
from light_llm.models.transformer.config import TransformerConfig
from light_llm.models.transformer.norm import build_norm
from light_llm.models.transformer.positional import (
    ALiBiEmbedding,
    LearnedEmbedding,
    RotaryEmbedding,
    SinusoidalEmbedding,
    build_positional,
)


class Transformer(nn.Module):
    """
    Decoder-only Transformer language model.

    Example
    -------
    >>> cfg = TransformerConfig(vocab_size=32_000, d_model=512, num_layers=6)
    >>> model = Transformer(cfg)
    >>> tokens = torch.randint(0, 32_000, (2, 128))
    >>> logits = model(tokens)  # [2, 128, 32_000]
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ------------------------------------------------------------------
        # Token embedding
        # ------------------------------------------------------------------
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        # ------------------------------------------------------------------
        # Positional encoding
        # ------------------------------------------------------------------
        pos_module = build_positional(
            cfg.d_model,
            cfg.head_dim,
            cfg.attention.num_heads,
            cfg.max_seq_len,
            cfg.positional,
        )
        enc = cfg.positional.encoding

        self.rope: Optional[RotaryEmbedding]   = pos_module if enc == "rope"   else None
        self.alibi: Optional[ALiBiEmbedding]   = pos_module if enc == "alibi"  else None
        self.pos_emb: Optional[nn.Module]      = pos_module if enc in ("sinusoidal", "learned") else None

        # ------------------------------------------------------------------
        # Transformer layers
        # ------------------------------------------------------------------
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, rope=self.rope, alibi=self.alibi) for _ in range(cfg.num_layers)]
        )

        # ------------------------------------------------------------------
        # Final norm + LM head
        # ------------------------------------------------------------------
        self.final_norm = build_norm(cfg.d_model, cfg.norm)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # ------------------------------------------------------------------
        # Initialisation
        # ------------------------------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        cfg = self.cfg
        std = cfg.init_std

        def _normal(module: nn.Linear | nn.Embedding, s: float) -> None:
            nn.init.normal_(module.weight, mean=0.0, std=s)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        _normal(self.tok_emb, std)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Residual projection weights use scaled init (GPT-2 style)
                is_residual = any(k in name for k in ("out_proj", "down_proj", "fc2"))
                s = cfg.residual_init_std(std) if is_residual else std
                _normal(module, s)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: Tensor,
        past_kvs: Optional[list[tuple[Tensor, Tensor]]] = None,
        return_kvs: bool = False,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """
        Args:
            tokens:    [batch, seq_len]  (long)
            past_kvs:  list of (k, v) tensors per layer (for KV-cache inference)
            return_kvs: whether to return updated KV cache
        Returns:
            logits [batch, seq_len, vocab_size]
            optionally, new_kvs: list of (k, v) per layer
        """
        B, T = tokens.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.tok_emb(tokens)

        if self.pos_emb is not None:
            offset = past_kvs[0][0].shape[2] if past_kvs else 0
            if isinstance(self.pos_emb, (SinusoidalEmbedding, LearnedEmbedding)):
                x = self.pos_emb(x, offset=offset)

        x = self.emb_drop(x)

        new_kvs: list[tuple[Tensor, Tensor]] = []
        for i, layer in enumerate(self.layers):
            pkv = past_kvs[i] if past_kvs else None
            result = layer(x, past_kv=pkv, return_kv=return_kvs)
            if return_kvs:
                x, kv = result  # type: ignore[misc]
                new_kvs.append(kv)
            else:
                x = result  # type: ignore[assignment]

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if return_kvs:
            return logits, new_kvs
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressive token generation with KV-cache.

        Args:
            prompt_tokens: [1, prompt_len]  (batch size 1 only)
            max_new_tokens: number of tokens to generate
            temperature:    sampling temperature (1.0 = unscaled)
            top_k:          keep only top-k logits before sampling
            top_p:          nucleus sampling threshold
            repetition_penalty: >1 penalises repeated tokens
            eos_token_id:   stop when this token is generated
        Returns:
            [1, prompt_len + generated] token tensor
        """
        self.eval()
        device = prompt_tokens.device
        tokens = prompt_tokens.clone()
        past_kvs: Optional[list[tuple[Tensor, Tensor]]] = None

        for _ in range(max_new_tokens):
            # On first pass use the full prompt; afterwards use only last token
            input_ids = tokens if past_kvs is None else tokens[:, -1:]

            logits, past_kvs = self(input_ids, past_kvs=past_kvs, return_kvs=True)  # type: ignore[misc]
            logits = logits[:, -1, :]  # [1, vocab]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tok in set(tokens[0].tolist()):
                    logits[0, tok] /= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-p (nucleus)
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_tok], dim=1)

            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break

        return tokens

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        params = self.num_parameters()
        return (
            f"Transformer(\n"
            f"  vocab={self.cfg.vocab_size}, d_model={self.cfg.d_model}, "
            f"layers={self.cfg.num_layers},\n"
            f"  attn={self.cfg.attention.variant}, "
            f"pos={self.cfg.positional.encoding}, "
            f"ffn={self.cfg.ffn.variant},\n"
            f"  params={params / 1e6:.1f}M\n"
            f")"
        )

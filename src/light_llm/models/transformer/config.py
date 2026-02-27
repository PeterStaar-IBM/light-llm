"""
Pydantic configuration classes for the Transformer model.

Design philosophy:
  - All tuneable knobs live here; the model modules read from these.
  - Sane defaults match a ~modern~ small-LLM recipe:
      RMSNorm + RoPE + SwiGLU + GQA (no biases).
  - Experimental features (complex attention, ALiBi, sliding window …)
    are exposed as first-class options so they can be toggled per run.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class NormConfig(BaseModel):
    """Layer-normalisation settings."""

    type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    eps: float = 1e-5
    # Where to place the norm relative to the sub-layer
    placement: Literal["pre", "post", "sandwich"] = "pre"


class PositionalConfig(BaseModel):
    """Positional encoding settings."""

    encoding: Literal["rope", "alibi", "sinusoidal", "learned", "none"] = "rope"
    # RoPE options
    rope_base: float = 10_000.0
    # Optional linear RoPE scaling factor (>1 extends context)
    rope_scaling: Optional[float] = None


class AttentionConfig(BaseModel):
    """Multi-head attention settings."""

    # Structural variant
    variant: Literal["mha", "gqa", "mqa", "complex"] = "gqa"
    num_heads: int = 8
    # num_kv_heads is used for GQA / MQA.
    #   None  → resolved automatically (= num_heads for mha, 1 for mqa)
    #   int   → used for gqa; must divide num_heads evenly
    num_kv_heads: Optional[int] = None
    # If None, inferred as d_model // num_heads
    head_dim: Optional[int] = None

    dropout: float = 0.0
    bias: bool = False  # QKV + output projection biases

    # Attention pattern
    mask_type: Literal["causal", "bidirectional", "sliding_window"] = "causal"
    # Used when mask_type == "sliding_window"
    window_size: Optional[int] = None

    # Flash Attention (requires flash-attn package)
    use_flash_attn: bool = False

    @model_validator(mode="after")
    def _resolve_kv_heads(self) -> "AttentionConfig":
        match self.variant:
            case "mha":
                self.num_kv_heads = self.num_heads
            case "mqa":
                self.num_kv_heads = 1
            case "gqa":
                if self.num_kv_heads is None:
                    # Default: 1/4 of query heads
                    self.num_kv_heads = max(1, self.num_heads // 4)
                if self.num_heads % self.num_kv_heads != 0:
                    raise ValueError(
                        f"num_heads ({self.num_heads}) must be divisible by "
                        f"num_kv_heads ({self.num_kv_heads})"
                    )
            case "complex":
                # Complex attention uses a single head-group (effectively MHA)
                self.num_kv_heads = self.num_heads
        return self


class FFNConfig(BaseModel):
    """Feed-forward network settings."""

    variant: Literal["swiglu", "geglu", "mlp"] = "swiglu"
    # Intermediate dimension multiplier.
    # For SwiGLU, 8/3 keeps parity with a 4× standard MLP parameter count.
    expansion_factor: float = 8 / 3
    dropout: float = 0.0
    # Activation used only for the plain "mlp" variant
    activation: Literal["silu", "gelu", "relu"] = "silu"
    bias: bool = False


class TransformerConfig(BaseModel):
    """Top-level model configuration."""

    # Vocabulary & sequence
    vocab_size: int = 32_000
    max_seq_len: int = 2048

    # Core dimensions
    d_model: int = 512
    num_layers: int = 6

    # Embedding
    dropout: float = 0.0
    tie_embeddings: bool = True  # Share input/output embedding weights

    # Optional: compute attention + FFN in parallel (PaLM-style)
    parallel_layers: bool = False

    # Sub-module configs
    norm: NormConfig = Field(default_factory=NormConfig)
    positional: PositionalConfig = Field(default_factory=PositionalConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    ffn: FFNConfig = Field(default_factory=FFNConfig)

    # Initialisation
    init_std: float = 0.02
    # Scaled init for residual projections (GPT-2 style): std / sqrt(2 * n_layers)
    residual_scaled_init: bool = True

    @model_validator(mode="after")
    def _derived_fields(self) -> "TransformerConfig":
        # Infer head_dim when not given
        if self.attention.head_dim is None:
            self.attention.head_dim = self.d_model // self.attention.num_heads
        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def head_dim(self) -> int:
        assert self.attention.head_dim is not None
        return self.attention.head_dim

    @property
    def ffn_dim(self) -> int:
        """Intermediate FFN dimension (rounded to nearest multiple of 64)."""
        raw = int(self.d_model * self.ffn.expansion_factor)
        return math.ceil(raw / 64) * 64

    def residual_init_std(self, base_std: Optional[float] = None) -> float:
        std = base_std or self.init_std
        if self.residual_scaled_init:
            return std / math.sqrt(2 * self.num_layers)
        return std

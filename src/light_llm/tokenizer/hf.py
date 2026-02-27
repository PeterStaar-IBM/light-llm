"""
HuggingFace tokenizer wrapper.

Wraps any ``transformers.PreTrainedTokenizer`` / ``PreTrainedTokenizerFast``
so it satisfies the ``BaseTokenizer`` interface.

This lets you reuse the exact same vocabulary and merge rules as any
publicly available model (GPT-2, LLaMA-3, Mistral, Falcon, …) while
keeping the rest of the training pipeline tokenizer-agnostic.

Usage
-----
    # From a pretrained model name
    tok = HFTokenizer.from_pretrained("gpt2")
    tok = HFTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    # From a local directory previously saved with .save()
    tok = HFTokenizer.load("/path/to/saved/tokenizer")

    ids = tok.encode("Hello world")
    text = tok.decode(ids)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from light_llm.tokenizer.base import BaseTokenizer


class HFTokenizer(BaseTokenizer):
    """
    Thin wrapper around a HuggingFace ``PreTrainedTokenizer``.

    All heavy lifting is delegated to the HF tokenizer; this class only
    adapts the interface to ``BaseTokenizer``.
    """

    def __init__(self, hf_tokenizer: "transformers.PreTrainedTokenizerBase") -> None:  # type: ignore[name-defined]  # noqa: F821
        self._tok = hf_tokenizer

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "HFTokenizer":
        """
        Load any HuggingFace tokenizer by model name or local path.

        Parameters
        ----------
        name_or_path:
            HF model id (e.g. ``"gpt2"``, ``"meta-llama/Meta-Llama-3-8B"``)
            or a local directory path.
        use_fast:
            Prefer the Rust-backed fast tokenizer when available.
        trust_remote_code:
            Needed for some community tokenizers.
        **kwargs:
            Forwarded to ``AutoTokenizer.from_pretrained``.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for HFTokenizer. "
                "Install it with: pip install transformers"
            ) from e

        hf_tok = AutoTokenizer.from_pretrained(
            name_or_path,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(hf_tok)

    # ------------------------------------------------------------------
    # BaseTokenizer interface
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tok.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._tok.pad_token_id

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._tok.unk_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = True
    ) -> list[list[int]]:
        result = self._tok(
            texts,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
        )
        return result["input_ids"]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save tokenizer to a directory (HF format — loadable with .load())."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._tok.save_pretrained(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "HFTokenizer":
        """Load a tokenizer previously saved with .save()."""
        return cls.from_pretrained(str(path), use_fast=True)

    # ------------------------------------------------------------------
    # Passthrough access to the underlying HF tokenizer
    # ------------------------------------------------------------------

    @property
    def backend(self) -> "transformers.PreTrainedTokenizerBase":  # type: ignore[name-defined]  # noqa: F821
        """Access the raw HuggingFace tokenizer for advanced use."""
        return self._tok

    def __repr__(self) -> str:
        return f"HFTokenizer({self._tok.__class__.__name__}, vocab_size={self.vocab_size})"

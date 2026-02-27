"""
Abstract base class for all tokenizers.

Every tokenizer must be serialisable (save/load) and expose a consistent
encode/decode interface so the rest of the codebase is tokenizer-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator


class BaseTokenizer(ABC):
    """
    Minimal interface that every tokenizer must implement.

    Subclasses implement the actual encoding/decoding logic and can carry
    whatever internal state they need (vocabulary, merges, etc.).
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""

    @property
    @abstractmethod
    def bos_token_id(self) -> int | None:
        """Beginning-of-sequence token id (or None if not defined)."""

    @property
    @abstractmethod
    def eos_token_id(self) -> int | None:
        """End-of-sequence token id (or None if not defined)."""

    @property
    @abstractmethod
    def pad_token_id(self) -> int | None:
        """Padding token id (or None if not defined)."""

    @property
    @abstractmethod
    def unk_token_id(self) -> int | None:
        """Unknown token id (or None if not defined)."""

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert a string to a list of token ids."""

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert a list of token ids back to a string."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = True
    ) -> list[list[int]]:
        return [self.encode(t, add_special_tokens) for t in texts]

    def decode_batch(
        self, batch: list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        return [self.decode(ids, skip_special_tokens) for ids in batch]

    def iter_chunks(
        self, text: str, chunk_size: int, add_special_tokens: bool = True
    ) -> Iterator[list[int]]:
        """
        Encode `text` and yield non-overlapping windows of `chunk_size` tokens.
        Useful for chunking a large document during preprocessing.
        """
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            if chunk:
                yield chunk

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the tokenizer to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseTokenizer":
        """Load a tokenizer from disk."""

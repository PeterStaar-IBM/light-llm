"""
BPE tokenizer backed by HuggingFace ``tokenizers`` (Rust, fast).

Trains a brand-new BPE vocabulary from scratch on your own text corpus,
which is useful when you want a custom vocabulary size or domain-specific
token distribution rather than borrowing one from an existing model.

For reusing an *existing* model's tokenizer (GPT-2, LLaMA, …) see
``HFTokenizer.from_pretrained()``.

Usage
-----
    # Train on a list of files
    tok = BPETokenizer.train(
        files=["data/train.txt"],
        vocab_size=8_000,
        save_path="tokenizer/bpe-8k",
    )

    # Load later
    tok = BPETokenizer.load("tokenizer/bpe-8k")

    ids = tok.encode("Hello world")
    print(tok.decode(ids))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from light_llm.tokenizer.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    Byte-level BPE tokenizer.

    Internally uses ``tokenizers.ByteLevelBPETokenizer`` (the same backend
    that powers GPT-2) so the vocabulary is byte-level and handles any
    Unicode text without an explicit <unk>.
    """

    _SPECIAL = {
        "bos": "<|bos|>",
        "eos": "<|eos|>",
        "pad": "<|pad|>",
    }
    _META = "meta.json"

    def __init__(
        self,
        hf_tokenizer: "tokenizers.Tokenizer",  # type: ignore[name-defined]  # noqa: F821
        bos_id: Optional[int],
        eos_id: Optional[int],
        pad_id: Optional[int],
    ) -> None:
        self._tok = hf_tokenizer
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._pad_id = pad_id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        files: list[str],
        vocab_size: int = 8_000,
        min_frequency: int = 2,
        save_path: Optional[str | Path] = None,
        show_progress: bool = True,
    ) -> "BPETokenizer":
        """
        Train a byte-level BPE tokenizer on plain-text files.

        Parameters
        ----------
        files:
            List of plain-text file paths used as training corpus.
        vocab_size:
            Target vocabulary size (including special tokens).
        min_frequency:
            Minimum merge frequency; pairs seen fewer times are skipped.
        save_path:
            If provided, automatically saves after training.
        show_progress:
            Show a tqdm progress bar during training.
        """
        try:
            from tokenizers import ByteLevelBPETokenizer  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "tokenizers is required. Install with: pip install tokenizers"
            ) from e

        special_tokens = list(cls._SPECIAL.values())
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=files,
            vocab_size=vocab_size - len(special_tokens),
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=show_progress,
        )

        bos_id = tokenizer.token_to_id(cls._SPECIAL["bos"])
        eos_id = tokenizer.token_to_id(cls._SPECIAL["eos"])
        pad_id = tokenizer.token_to_id(cls._SPECIAL["pad"])

        instance = cls(tokenizer, bos_id, eos_id, pad_id)
        if save_path is not None:
            instance.save(save_path)
        return instance

    # ------------------------------------------------------------------
    # BaseTokenizer interface
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_id

    @property
    def unk_token_id(self) -> Optional[int]:
        return None  # byte-level BPE has no <unk>

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_special_tokens:
            if self._bos_id is not None:
                ids = [self._bos_id] + ids
            if self._eos_id is not None:
                ids = ids + [self._eos_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special = {self._bos_id, self._eos_id, self._pad_id} - {None}
            ids = [i for i in ids if i not in special]
        return self._tok.decode(ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._tok.save_model(str(path))
        meta = {
            "type": "bpe",
            "bos_id": self._bos_id,
            "eos_id": self._eos_id,
            "pad_id": self._pad_id,
        }
        (path / self._META).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        try:
            from tokenizers import ByteLevelBPETokenizer  # type: ignore[import]
        except ImportError as e:
            raise ImportError("pip install tokenizers") from e

        tokenizer = ByteLevelBPETokenizer(
            vocab=str(path / "vocab.json"),
            merges=str(path / "merges.txt"),
        )
        meta = json.loads((path / cls._META).read_text())
        return cls(tokenizer, meta["bos_id"], meta["eos_id"], meta["pad_id"])

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size})"

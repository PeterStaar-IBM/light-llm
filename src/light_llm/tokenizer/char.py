"""
Character-level tokenizer.

Maps each unique Unicode character (or byte, in byte-level mode) to an
integer id.  Dead simple and great for quick experiments on small corpora.

Usage
-----
    # Build from text
    tok = CharTokenizer.from_text(open("data/input.txt").read())
    tok.save("tokenizer/char")

    # Load
    tok = CharTokenizer.load("tokenizer/char")
    ids = tok.encode("Hello!")
    print(tok.decode(ids))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


from light_llm.tokenizer.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    Simple character-level tokenizer.

    Vocabulary is the set of unique characters seen during construction,
    plus a small set of special tokens.
    """

    _SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(
        self,
        char_to_id: dict[str, int],
        id_to_char: dict[int, str],
    ) -> None:
        self._c2i = char_to_id
        self._i2c = id_to_char

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_text(cls, text: str, save_path: Optional[str | Path] = None) -> "CharTokenizer":
        """Build vocabulary from a string (or concatenated corpus)."""
        chars = sorted(set(text))
        # Reserve ids 0-3 for special tokens
        specials = cls._SPECIAL_TOKENS
        vocab = specials + [c for c in chars if c not in specials]
        c2i = {c: i for i, c in enumerate(vocab)}
        i2c = {i: c for c, i in c2i.items()}
        instance = cls(c2i, i2c)
        if save_path is not None:
            instance.save(save_path)
        return instance

    @classmethod
    def from_files(cls, *paths: str | Path, save_path: Optional[str | Path] = None) -> "CharTokenizer":
        """Build vocabulary by scanning one or more plain-text files."""
        text = ""
        for p in paths:
            text += Path(p).read_text(encoding="utf-8")
        return cls.from_text(text, save_path=save_path)

    # ------------------------------------------------------------------
    # BaseTokenizer interface
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._c2i)

    @property
    def bos_token_id(self) -> int:
        return self._c2i["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self._c2i["<eos>"]

    @property
    def pad_token_id(self) -> int:
        return self._c2i["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self._c2i["<unk>"]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        unk = self.unk_token_id
        ids = [self._c2i.get(c, unk) for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special = {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id}
        chars = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            chars.append(self._i2c.get(i, ""))
        return "".join(chars)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {"type": "char", "vocab": self._c2i}
        (path / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads((Path(path) / "tokenizer.json").read_text())
        c2i: dict[str, int] = data["vocab"]
        i2c = {int(i): c for c, i in c2i.items()}
        return cls(c2i, i2c)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"

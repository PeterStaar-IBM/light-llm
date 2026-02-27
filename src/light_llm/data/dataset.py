"""
Dataset classes for token-sequence training.

Workflow
--------
1. ``preprocess`` tool tokenises raw text → saves parquet files where each
   row is one document's token sequence (variable length list[int]).
2. ``TokenDataset`` loads those parquet files and serves fixed-length
   windows using one of two strategies:
     - "pack"   : concatenate all tokens (with optional EOS separator)
                  and slice into non-overlapping windows.  Zero padding waste.
     - "truncate": each row is independently truncated / padded to seq_len.

The dataset returns (input_ids, labels) where labels = input_ids shifted
left by one (standard next-token prediction).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """
    PyTorch dataset over pre-tokenised parquet files.

    Parameters
    ----------
    paths:
        One or more parquet files / glob patterns.
    seq_len:
        Context window length fed to the model.
    strategy:
        "pack"     – efficient packing (recommended for training).
        "truncate" – per-document truncation (useful for evaluation).
    eos_token_id:
        Token appended between documents when strategy="pack".
    column:
        Column name in the parquet file that contains the token lists.
    """

    def __init__(
        self,
        paths: list[str | Path],
        seq_len: int,
        strategy: Literal["pack", "truncate"] = "pack",
        eos_token_id: Optional[int] = None,
        column: str = "tokens",
    ) -> None:
        self.seq_len = seq_len
        self.strategy = strategy

        # ------------------------------------------------------------------
        # Load all parquet shards
        # ------------------------------------------------------------------
        frames = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                frames.extend(pd.read_parquet(f) for f in sorted(p.glob("*.parquet")))
            else:
                frames.append(pd.read_parquet(p))
        if not frames:
            raise FileNotFoundError(f"No parquet files found in {paths}")
        df = pd.concat(frames, ignore_index=True)

        # ------------------------------------------------------------------
        # Build the token buffer
        # ------------------------------------------------------------------
        if strategy == "pack":
            self._data = self._build_packed(df[column], eos_token_id)
            n = len(self._data)
            # Number of complete windows
            self._length = (n - 1) // seq_len
        else:
            self._rows: list[np.ndarray] = [
                np.asarray(row, dtype=np.int32) for row in df[column]
            ]
            self._length = len(self._rows)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_packed(series: pd.Series, eos: Optional[int]) -> np.ndarray:
        """Concatenate all token sequences into one flat array."""
        parts = []
        sep = np.array([eos], dtype=np.int32) if eos is not None else None
        for row in series:
            parts.append(np.asarray(row, dtype=np.int32))
            if sep is not None:
                parts.append(sep)
        return np.concatenate(parts)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if self.strategy == "pack":
            start = idx * self.seq_len
            chunk = self._data[start : start + self.seq_len + 1]
            x = torch.from_numpy(chunk[:-1].astype(np.int64))
            y = torch.from_numpy(chunk[1:].astype(np.int64))
        else:
            row = self._rows[idx]
            if len(row) >= self.seq_len + 1:
                row = row[: self.seq_len + 1]
            # Pad if shorter (shouldn't happen often)
            elif len(row) < self.seq_len + 1:
                pad = np.zeros(self.seq_len + 1 - len(row), dtype=np.int32)
                row = np.concatenate([row, pad])
            x = torch.from_numpy(row[:-1].astype(np.int64))
            y = torch.from_numpy(row[1:].astype(np.int64))

        return x, y

    # ------------------------------------------------------------------

    @classmethod
    def from_glob(
        cls,
        pattern: str,
        seq_len: int,
        **kwargs,
    ) -> "TokenDataset":
        """Convenience constructor accepting a glob pattern."""
        paths = sorted(Path(".").glob(pattern))
        return cls(paths, seq_len, **kwargs)

"""
ShardedTokenDataset – streaming IterableDataset over pre-tokenised parquet shards.

Epoch ordering
--------------
- Each epoch the shard list is shuffled with a deterministic seed (base_seed + epoch).
- Within each shard, rows are read linearly (no in-shard shuffling).
- Tokens from consecutive rows are concatenated in a carry buffer;
  non-overlapping windows of ``seq_len`` are yielded as (x, y) pairs.

Multi-worker DataLoader
-----------------------
Each worker receives a disjoint slice of the shuffled shard list:
  worker k  →  shards[k :: num_workers]
This guarantees full coverage with no duplicates.

Epoch control
-------------
Call ``dataset.set_epoch(n)`` before each new ``iter(DataLoader)`` so that
the shard order changes between epochs::

    for epoch in range(num_epochs):
        train_ds.set_epoch(epoch)
        for x, y in DataLoader(train_ds, ...):
            ...
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import IterableDataset


class ShardedTokenDataset(IterableDataset):
    """
    Iterable dataset that reads pre-tokenised parquet shards in a
    per-epoch random order while iterating within each shard linearly.

    Parameters
    ----------
    shard_paths:
        List of parquet file paths.  Directories are expanded automatically.
    seq_len:
        Token window length for each training example.
    eos_token_id:
        Appended after every document's tokens before packing.
        Helps the model learn to predict end-of-document.
    token_column:
        Column name that holds the token lists inside the parquet files.
    base_seed:
        Base random seed; epoch seed = base_seed + epoch.
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        seq_len: int,
        eos_token_id: Optional[int] = None,
        token_column: str = "tokens",
        base_seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.token_column = token_column
        self.base_seed = base_seed
        self._epoch = 0

        # Resolve directories → individual parquet files
        resolved: list[Path] = []
        for p in shard_paths:
            p = Path(p)
            if p.is_dir():
                resolved.extend(sorted(p.glob("*.parquet")))
            elif p.is_file():
                resolved.append(p)
            else:
                # Glob pattern relative to cwd
                resolved.extend(sorted(Path(".").glob(str(p))))

        if not resolved:
            raise FileNotFoundError(f"No parquet shards found in: {shard_paths}")

        self.shards: list[Path] = resolved

    # ------------------------------------------------------------------
    # Epoch control
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Call before each new ``iter(DataLoader)`` to rotate shard order."""
        self._epoch = epoch

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        epoch = self._epoch

        # Shuffle shard list with per-epoch seed
        rng = random.Random(self.base_seed + epoch)
        shards = self.shards.copy()
        rng.shuffle(shards)

        # Split shards across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        return self._iter_shards(shards)

    def _iter_shards(self, shards: list[Path]) -> Iterator[tuple[Tensor, Tensor]]:
        carry = np.empty(0, dtype=np.int32)
        sep = np.array([self.eos_token_id], dtype=np.int32) if self.eos_token_id is not None else None

        for shard in shards:
            # Read only the tokens column to keep memory low
            df = pd.read_parquet(shard, columns=[self.token_column])

            for raw_tokens in df[self.token_column]:
                tokens = np.asarray(raw_tokens, dtype=np.int32)
                if sep is not None:
                    tokens = np.concatenate([tokens, sep])

                all_tokens = np.concatenate([carry, tokens])

                # Yield non-overlapping windows of seq_len+1
                n_windows = (len(all_tokens) - 1) // self.seq_len
                end_idx = n_windows * self.seq_len

                for i in range(n_windows):
                    start = i * self.seq_len
                    chunk = all_tokens[start : start + self.seq_len + 1]
                    yield (
                        torch.from_numpy(chunk[:-1].astype(np.int64)),
                        torch.from_numpy(chunk[1:].astype(np.int64)),
                    )

                # Keep leftover tokens as carry for the next document
                carry = all_tokens[end_idx:].copy()

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ShardedTokenDataset("
            f"shards={len(self.shards)}, seq_len={self.seq_len})"
        )

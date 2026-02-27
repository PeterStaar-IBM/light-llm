"""
Data collation utilities.

``LMCollator`` is used with ``DataLoader(collate_fn=...)`` to batch
(input_ids, labels) pairs.  It optionally masks padding tokens in the loss
by setting label values to -100 (PyTorch's ``CrossEntropyLoss`` default
ignore index).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class LMCollator:
    """
    Collate (x, y) pairs into padded batches.

    For packed datasets every sequence has the same length, so this is a
    trivial stack.  For truncate-mode datasets with variable lengths it
    pads and optionally masks the labels.

    Parameters
    ----------
    pad_token_id:
        Token used for padding x (input).
    label_pad_id:
        Value used to mask padding positions in y.
        -100 causes ``F.cross_entropy`` to ignore those positions.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_id: int = -100,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(
        self, batch: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        xs, ys = zip(*batch)

        # Fast path: all same length (packed dataset)
        if all(x.shape == xs[0].shape for x in xs):
            return torch.stack(list(xs)), torch.stack(list(ys))

        # Variable-length: pad
        x_pad = pad_sequence(xs, batch_first=True, padding_value=self.pad_token_id)
        y_pad = pad_sequence(ys, batch_first=True, padding_value=self.label_pad_id)
        return x_pad, y_pad

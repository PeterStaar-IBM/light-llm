"""
Learning-rate schedulers.

All return a ``torch.optim.lr_scheduler.LambdaLR``-compatible lambda so
they compose naturally with PyTorch's optimiser framework.

Available schedules
-------------------
  cosine_with_warmup  – linear warmup → cosine decay to min_lr  (default)
  linear_with_warmup  – linear warmup → linear decay to min_lr
  constant_with_warmup – linear warmup → flat
  wsd                 – Warmup-Stable-Decay (used in MiniCPM/Mistral research)
"""

from __future__ import annotations

import math
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Linear warmup followed by cosine decay."""

    def _lr(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, _lr)


def linear_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Linear warmup followed by linear decay."""

    def _lr(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

    return LambdaLR(optimizer, _lr)


def constant_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
) -> LambdaLR:
    """Linear warmup then constant LR."""

    def _lr(step: int) -> float:
        return min(1.0, step / max(1, warmup_steps))

    return LambdaLR(optimizer, _lr)


def wsd_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Warmup-Stable-Decay schedule.

    Three phases:
      1. Warmup  : 0 → 1  over ``warmup_steps``
      2. Stable  : 1       for ``stable_steps``
      3. Decay   : 1 → min over ``decay_steps`` (cosine)
    """

    def _lr(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        step -= warmup_steps
        if step < stable_steps:
            return 1.0
        step -= stable_steps
        progress = step / max(1, decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, _lr)


def build_scheduler(
    optimizer: Optimizer,
    schedule: Literal["cosine", "linear", "constant", "wsd"],
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    match schedule:
        case "cosine":
            return cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio)
        case "linear":
            return linear_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio)
        case "constant":
            return constant_with_warmup(optimizer, warmup_steps)
        case "wsd":
            # Sensible defaults: 10% warmup, 80% stable, 10% decay
            stable = int(total_steps * 0.8)
            decay  = total_steps - warmup_steps - stable
            return wsd_schedule(optimizer, warmup_steps, stable, max(1, decay), min_lr_ratio)
        case _:
            raise ValueError(f"Unknown schedule: {schedule!r}")

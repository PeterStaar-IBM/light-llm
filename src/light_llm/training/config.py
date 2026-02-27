"""
Pydantic training configuration.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from light_llm.models.transformer.config import TransformerConfig


class OptimizerConfig(BaseModel):
    type: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0


class SchedulerConfig(BaseModel):
    type: Literal["cosine", "linear", "constant", "wsd"] = "cosine"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1


class TrainingConfig(BaseModel):
    # ---- data ----
    # Single path/glob string or a list of paths (one per preprocessed dataset).
    train_data: Union[str, list[str]]
    val_data: Optional[Union[str, list[str]]] = None

    # ---- model ----
    model: TransformerConfig = Field(default_factory=TransformerConfig)

    # ---- tokenizer ----
    # Path to a saved tokenizer directory (HF / BPE / Char format)
    tokenizer_path: Optional[str] = None

    # ---- sequences ----
    seq_len: int = 1024
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # ---- training loop ----
    num_steps: int = 10_000
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    # ---- precision ----
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

    # ---- checkpointing ----
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1_000
    keep_last_n: int = 3         # how many checkpoint files to keep

    # ---- logging ----
    log_every: int = 10
    eval_every: int = 500
    eval_steps: int = 50         # batches to evaluate on

    # ---- wandb ----
    use_wandb: bool = False
    wandb_project: str = "light-llm"
    wandb_run_name: Optional[str] = None

    # ---- misc ----
    seed: int = 42
    num_workers: int = 4
    compile_model: bool = False  # torch.compile (requires PyTorch 2.x)

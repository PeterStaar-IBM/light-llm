"""
Training loop.

Supports:
  - Mixed-precision via ``torch.amp.autocast`` (bfloat16 / float16 / float32)
  - Gradient scaling for fp16
  - Gradient accumulation
  - Gradient clipping
  - AdamW / Adam / SGD optimisers
  - Cosine / linear / constant / WSD LR schedulers with warmup
  - ShardedTokenDataset: per-epoch randomised shard order, linear row iteration
  - Periodic checkpointing (safetensors weights + optimiser state)
  - Config file copied to checkpoint dir for reproducibility
  - Optional torch.compile
  - Optional Weights & Biases logging
  - Validation loss evaluation
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from light_llm.data.collator import LMCollator
from light_llm.data.shard_dataset import ShardedTokenDataset
from light_llm.models.transformer.model import Transformer
from light_llm.training.config import TrainingConfig
from light_llm.training.scheduler import build_scheduler


class Trainer:
    """
    Manages the full training lifecycle for a Transformer LM.

    Parameters
    ----------
    cfg:
        Fully validated ``TrainingConfig``.
    model:
        Initialised ``Transformer`` (on CPU; Trainer moves it to device).
    config_path:
        Optional path to the config file used to launch this run.
        When provided, the file is copied into ``cfg.checkpoint_dir`` so
        every checkpoint folder is self-contained and reproducible.
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        model: Transformer,
        config_path: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = self._resolve_dtype()
        self.use_amp = self.dtype in (torch.float16, torch.bfloat16)

        torch.manual_seed(cfg.seed)

        # ------------------------------------------------------------------
        # Move / compile model
        # ------------------------------------------------------------------
        self.model = self.model.to(self.device)
        if cfg.compile_model:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # Optimizer & scaler
        # ------------------------------------------------------------------
        self.optimizer = self._build_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == torch.float16))

        # ------------------------------------------------------------------
        # Datasets & loaders
        # ------------------------------------------------------------------
        self.train_ds, self.val_ds = self._build_datasets()
        self.train_loader = self._make_loader(self.train_ds, shuffle=False)
        self.val_loader = self._make_loader(self.val_ds, shuffle=False) if self.val_ds else None

        # ------------------------------------------------------------------
        # LR scheduler
        # ------------------------------------------------------------------
        sc = cfg.scheduler
        self.scheduler = build_scheduler(
            self.optimizer,
            schedule=sc.type,
            warmup_steps=sc.warmup_steps,
            total_steps=cfg.num_steps,
            min_lr_ratio=sc.min_lr_ratio,
        )

        # ------------------------------------------------------------------
        # Training state
        # ------------------------------------------------------------------
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self._step_checkpoints: list[Path] = []

        # ------------------------------------------------------------------
        # WandB (optional)
        # ------------------------------------------------------------------
        self._wandb = None
        if cfg.use_wandb:
            self._init_wandb()

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        model = self.model
        model.train()

        # Copy config to checkpoint dir for reproducibility
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path and self.config_path.exists():
            dest = ckpt_dir / self.config_path.name
            if not dest.exists():
                shutil.copy2(self.config_path, dest)

        # Per-epoch shard shuffle
        self.train_ds.set_epoch(self.epoch)
        train_iter = iter(self.train_loader)

        accum_loss = 0.0
        t0 = time.perf_counter()
        pbar = tqdm(total=cfg.num_steps, initial=self.step, desc="Training", dynamic_ncols=True)

        while self.step < cfg.num_steps:

            self.optimizer.zero_grad(set_to_none=True)
            micro_loss = 0.0

            for _ in range(cfg.gradient_accumulation_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    # Epoch boundary: rotate shard order
                    self.epoch += 1
                    self.train_ds.set_epoch(self.epoch)
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.use_amp):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=-100,
                    ) / cfg.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                micro_loss += loss.item()

            # Clip → step → schedule
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.step += 1

            accum_loss += micro_loss

            # ---- Logging ----
            if self.step % cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = accum_loss / cfg.log_every
                lr = self.scheduler.get_last_lr()[0]
                tokens_per_sec = (
                    cfg.log_every * cfg.batch_size * cfg.gradient_accumulation_steps
                    * cfg.seq_len / max(elapsed, 1e-6)
                )
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{lr:.2e}",
                    epoch=self.epoch,
                    tok_s=f"{tokens_per_sec/1e3:.1f}k",
                )
                if self._wandb:
                    self._wandb.log(
                        {"train/loss": avg_loss, "train/lr": lr, "train/epoch": self.epoch},
                        step=self.step,
                    )
                accum_loss = 0.0
                t0 = time.perf_counter()

            # ---- Evaluation ----
            if self.step % cfg.eval_every == 0 and self.val_loader is not None:
                val_loss = self.evaluate()
                if self._wandb:
                    self._wandb.log({"val/loss": val_loss}, step=self.step)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")
                model.train()

            # ---- Checkpoint ----
            if self.step % cfg.save_every == 0:
                self._save_checkpoint(f"step{self.step:07d}")

            pbar.update(1)

        pbar.close()
        self._save_checkpoint("final")
        if self._wandb:
            self._wandb.finish()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0

        assert self.val_loader is not None
        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.cfg.eval_steps:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                )
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        from safetensors.torch import save_file

        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        weights_path = ckpt_dir / f"{tag}.safetensors"
        save_file(
            {k: v.contiguous() for k, v in self.model.state_dict().items()},
            str(weights_path),
        )

        state_path = ckpt_dir / f"{tag}.state.pt"
        torch.save(
            {
                "step": self.step,
                "epoch": self.epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            state_path,
        )

        # Full config JSON snapshot alongside each checkpoint
        cfg_path = ckpt_dir / f"{tag}.config.json"
        cfg_path.write_text(self.cfg.model_dump_json(indent=2))

        if "step" in tag:
            self._step_checkpoints.append(weights_path)
            self._prune_old_checkpoints()

    def _prune_old_checkpoints(self) -> None:
        keep = self.cfg.keep_last_n
        for old in self._step_checkpoints[:-keep]:
            stem = old.stem
            for ext in (".safetensors", ".state.pt", ".config.json"):
                p = old.parent / f"{stem}{ext}"
                if p.exists():
                    p.unlink()

    def load_checkpoint(self, path: str | Path) -> None:
        from safetensors.torch import load_file

        path = Path(path)
        weights = path if path.suffix == ".safetensors" else path.with_suffix(".safetensors")
        self.model.load_state_dict(load_file(str(weights), device=str(self.device)))

        state_path = path.parent / f"{path.stem}.state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.step = state["step"]
            self.epoch = state.get("epoch", 0)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            self.best_val_loss = state.get("best_val_loss", float("inf"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            self.cfg.dtype, torch.float32
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        oc = self.cfg.optimizer
        decay = [p for p in self.model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay,    "weight_decay": oc.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        match oc.type:
            case "adamw":
                return torch.optim.AdamW(groups, lr=oc.lr, betas=(oc.beta1, oc.beta2), eps=oc.eps)
            case "adam":
                return torch.optim.Adam(groups, lr=oc.lr, betas=(oc.beta1, oc.beta2), eps=oc.eps)
            case "sgd":
                return torch.optim.SGD(groups, lr=oc.lr, momentum=oc.beta1)
            case _:
                raise ValueError(f"Unknown optimizer: {oc.type!r}")

    def _build_datasets(self) -> tuple[ShardedTokenDataset, Optional[ShardedTokenDataset]]:
        cfg = self.cfg

        train_ds = ShardedTokenDataset(
            shard_paths=self._resolve_paths(cfg.train_data),
            seq_len=cfg.seq_len,
            base_seed=cfg.seed,
        )

        val_ds = None
        if cfg.val_data:
            val_ds = ShardedTokenDataset(
                shard_paths=self._resolve_paths(cfg.val_data),
                seq_len=cfg.seq_len,
                base_seed=cfg.seed,
            )

        return train_ds, val_ds

    def _make_loader(self, ds: ShardedTokenDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=LMCollator(),
            # IterableDataset: DataLoader must NOT shuffle; the dataset handles it
            shuffle=False,
        )

    @staticmethod
    def _resolve_paths(pattern: str) -> list[Path]:
        p = Path(pattern)
        if p.is_file():
            return [p]
        if p.is_dir():
            return sorted(p.glob("*.parquet"))
        return sorted(Path(".").glob(pattern))

    def _init_wandb(self) -> None:
        try:
            import wandb
            self._wandb = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config=self.cfg.model_dump(),
            )
        except ImportError:
            pass

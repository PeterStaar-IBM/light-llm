"""
Train a Transformer LM from a config file (JSON or YAML).

The config file is the single source of truth for all hyperparameters.
It is automatically copied into the checkpoint directory at the start of
training so each run is fully reproducible.

Usage
-----
    llm-train configs/small.yaml
    llm-train configs/small.yaml --resume checkpoints/step0005000

Config format
-------------
Both JSON and YAML are supported (detected by file extension).
See ``configs/`` for annotated examples.

All fields correspond 1-to-1 with ``TrainingConfig`` (Pydantic model).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from rich.console import Console

from light_llm.models.transformer.model import Transformer
from light_llm.training.config import TrainingConfig
from light_llm.training.trainer import Trainer

console = Console()
app = typer.Typer(
    help="Train a Transformer LM from a JSON/YAML config file.",
    add_completion=False,
)


@app.command()
def main(
    config: Annotated[Path, typer.Argument(help="Path to training config file (.json or .yaml).")],
    resume: Annotated[Optional[str], typer.Option("--resume", "-r", help="Resume from checkpoint path/tag.")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory for checkpoints. Defaults to <config-name>-<yyyymmdd-hhmm>.")] = None,
) -> None:

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Load and validate config
    # ------------------------------------------------------------------
    cfg = _load_config(config)

    # ------------------------------------------------------------------
    # Resolve output / checkpoint directory
    # ------------------------------------------------------------------
    if output is not None:
        cfg.checkpoint_dir = str(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        cfg.checkpoint_dir = f"{config.stem}-{timestamp}"

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = Transformer(cfg.model)
    _print_summary(model, cfg, config)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(cfg, model, config_path=config)

    if resume:
        console.print(f"[yellow]Resuming from {resume}[/yellow]")
        trainer.load_checkpoint(resume)

    trainer.train()
    console.print("[bold green]Training complete.[/bold green]")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> TrainingConfig:
    """Parse a JSON or YAML config file into a TrainingConfig."""
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("pip install pyyaml") from e
        raw = yaml.safe_load(path.read_text())
    elif suffix == ".json":
        raw = json.loads(path.read_text())
    else:
        # Try JSON first, fall back to YAML
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError:
            try:
                import yaml
                raw = yaml.safe_load(path.read_text())
            except Exception as e:
                console.print(f"[red]Could not parse config as JSON or YAML: {e}[/red]")
                raise typer.Exit(1) from e

    try:
        return TrainingConfig.model_validate(raw)
    except Exception as e:
        console.print(f"[red]Invalid config: {e}[/red]")
        raise typer.Exit(1) from e


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def _print_summary(model: Transformer, cfg: TrainingConfig, config_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = ""
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_info = f"  ({name}, {mem_gb:.0f} GB)"

    console.print(f"\n[bold]{model}[/bold]")
    console.print(f"  config    : {config_path}")
    console.print(f"  output    : {cfg.checkpoint_dir}")
    console.print(f"  device    : {device}{gpu_info}")
    console.print(f"  dtype     : {cfg.dtype}")
    console.print(f"  steps     : {cfg.num_steps:,}")
    console.print(f"  batch     : {cfg.batch_size} × {cfg.gradient_accumulation_steps} accum  "
                  f"(effective {cfg.batch_size * cfg.gradient_accumulation_steps})")
    console.print(f"  seq_len   : {cfg.seq_len}")
    console.print(f"  train_data: {cfg.train_data}")
    if cfg.val_data:
        console.print(f"  val_data  : {cfg.val_data}")
    console.print()

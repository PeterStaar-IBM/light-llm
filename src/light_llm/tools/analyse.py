"""
Analyse pre-tokenised parquet shards.

Computes three families of statistics over the ``tokens`` column produced by
``llm-preprocess``:

  1. Length histogram  — distribution of document lengths in tokens
  2. Token frequency   — occurrence counts per token id + Zipfian shape
  3. Token bigram graph — directed weighted graph of consecutive token pairs,
                          with optional graph analytics (PageRank, centralities)

All analytics are controlled by a YAML config file; results are written as
JSON, one file per shard (optional) and one global aggregate (optional).

Config file
-----------
  llm-analyse --config configs/analyse/example.yaml

CLI overrides
-------------
  llm-analyse --config configs/analyse/example.yaml \\
              --input data/wikipedia/train \\
              --output analysis/wikipedia

Inline (no config file)
-----------------------
  llm-analyse --input data/wikipedia/train --output analysis/wikipedia
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console

from light_llm.analysis.analytics import run_analysis
from light_llm.analysis.config import AnalysisConfig

console = Console()
app = typer.Typer(help="Analyse pre-tokenised parquet shards.")


@app.command()
def main(
    # ---- Config file -------------------------------------------------------
    config: Annotated[
        Optional[str],
        typer.Option(
            "--config", "-c",
            help="YAML/JSON analysis config file. CLI flags override file values.",
        ),
    ] = None,

    # ---- I/O ---------------------------------------------------------------
    input: Annotated[
        Optional[str],
        typer.Option("--input", "-i", help="Directory of parquet shards to analyse."),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Output directory for JSON results. [default: analysis]"),
    ] = None,

    # ---- Scope -------------------------------------------------------------
    per_shard: Annotated[
        Optional[bool],
        typer.Option(
            "--per-shard/--no-per-shard",
            help="Write {shard}_analysis.json for every shard. [default: true]",
        ),
    ] = None,
    aggregate: Annotated[
        Optional[bool],
        typer.Option(
            "--aggregate/--no-aggregate",
            help="Write global_analysis.json aggregated over all shards. [default: true]",
        ),
    ] = None,

    # ---- Caching -----------------------------------------------------------
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force", "-f",
            help="Recompute even if results JSON already exists. [default: false]",
        ),
    ] = False,
) -> None:
    # ------------------------------------------------------------------
    # Build effective config: defaults → YAML/JSON → CLI overrides
    # ------------------------------------------------------------------
    cfg = _load_config(config)
    _apply_overrides(cfg, {
        "input":     input,
        "output":    output,
        "per_shard": per_shard,
        "aggregate": aggregate,
    })

    if not cfg.input:
        console.print("[red]Provide --input or set 'input' in a --config file.[/red]")
        raise typer.Exit(1)

    shard_dir = Path(cfg.input)
    if not shard_dir.exists():
        console.print(f"[red]Input path does not exist: {shard_dir}[/red]")
        raise typer.Exit(1)

    shard_paths = (
        sorted(shard_dir.glob("*.parquet")) if shard_dir.is_dir() else [shard_dir]
    )
    if not shard_paths:
        console.print(f"[red]No parquet files found in {shard_dir}[/red]")
        raise typer.Exit(1)

    out_dir = Path(cfg.output)
    console.print(
        f"[bold]Analysing {len(shard_paths)} shard(s) → {out_dir}/[/bold]\n"
        f"  per_shard={cfg.per_shard}  aggregate={cfg.aggregate}\n"
        f"  analytics: "
        f"length_histogram={cfg.analytics.length_histogram}  "
        f"token_frequency={cfg.analytics.token_frequency}  "
        f"token_graph={cfg.analytics.token_graph}"
    )

    run_analysis(shard_paths, cfg, out_dir, force=force)

    console.print(f"\n[bold green]Done.[/bold green]  Results in {out_dir}/")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(path: Optional[str]) -> AnalysisConfig:
    if path is None:
        return AnalysisConfig()

    p = Path(path)
    if not p.exists():
        console.print(f"[red]Config file not found: {path}[/red]")
        raise typer.Exit(1)

    suffix = p.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(p.read_text())
        else:
            raw = json.loads(p.read_text())
    except Exception as e:
        console.print(f"[red]Could not parse config {path}: {e}[/red]")
        raise typer.Exit(1) from e

    try:
        return AnalysisConfig.model_validate(raw)
    except Exception as e:
        console.print(f"[red]Invalid analysis config: {e}[/red]")
        raise typer.Exit(1) from e


def _apply_overrides(cfg: AnalysisConfig, overrides: dict) -> None:
    for field, value in overrides.items():
        if value is not None:
            setattr(cfg, field, value)

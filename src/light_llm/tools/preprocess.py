"""
Preprocess text into tokenised parquet shards.

Output schema
-------------
Each output parquet file has four columns:
  source  – str       : "{repo_id}/{split}:{source_id}" or "{filename}:{row_idx}"
  length  – int       : number of tokens
  text    – str       : raw document text
  tokens  – list[int] : token ids

Input modes
-----------
  --hf-dataset  : stream a HuggingFace dataset (recommended for large corpora)
  --input       : tokenise local text / jsonl / parquet files

Config files
------------
Any flag can be stored in a YAML/JSON config and passed via --config/-c.
CLI flags always override config-file values.

  llm-preprocess --config configs/preprocess/wikipedia_en.yaml
  llm-preprocess --config configs/preprocess/wikipedia_en.yaml --output data/wiki/train

HF dataset subsets
------------------
Many HF datasets have named subsets (configs). Pass them with --hf-config:

  wikimedia/wikipedia   →  --hf-config 20231101.en
  allenai/c4            →  --hf-config en
  HuggingFaceFW/fineweb-edu  →  --hf-config CC-MAIN-2024-10

Examples
--------
# Stream Wikipedia using a preprocess config file
llm-preprocess --config configs/preprocess/wikipedia_en.yaml

# Stream wikimedia/wikipedia (English), overriding the output dir
llm-preprocess \\
    --config configs/preprocess/wikipedia_en.yaml \\
    --output data/wikipedia/val \\
    --hf-split validation

# Fully inline (no config file)
llm-preprocess \\
    --hf-dataset wikimedia/wikipedia \\
    --hf-config  20231101.en \\
    --hf-split   train \\
    --output     data/wikipedia/train \\
    --tokenizer  hf \\
    --tokenizer-path gpt2

# Local text files
llm-preprocess \\
    --input data/raw/ \\
    --output data/tokens \\
    --tokenizer bpe --vocab-size 16000 --save-tokenizer tokenizer/bpe-16k
"""

from __future__ import annotations

import glob as _glob
import importlib.util
import json
from pathlib import Path
from typing import Annotated, Iterator, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from light_llm.data.hf_adapters import DatasetAdapter, get_adapter
from light_llm.data.preprocess_config import PreprocessConfig
from light_llm.tokenizer import BPETokenizer, CharTokenizer, HFTokenizer
from light_llm.tokenizer.base import BaseTokenizer

console = Console()
app = typer.Typer(help="Tokenise datasets into parquet shards.")


@app.command()
def main(
    # ---- Config file -------------------------------------------------------
    config: Annotated[
        Optional[str],
        typer.Option("--config", "-c", help="YAML/JSON preprocess config file. CLI flags override values from the file."),
    ] = None,

    # ---- Source: HF Hub ----------------------------------------------------
    hf_dataset: Annotated[
        Optional[str],
        typer.Option("--hf-dataset", help="HuggingFace dataset repo ID."),
    ] = None,
    hf_config: Annotated[
        Optional[str],
        typer.Option("--hf-config", help="Dataset subset / config name (e.g. '20231101.en' for Wikipedia, 'en' for C4)."),
    ] = None,
    hf_split: Annotated[
        Optional[str],
        typer.Option("--hf-split", help="Dataset split to use. [default: train]"),
    ] = None,
    trust_remote_code: Annotated[
        Optional[bool],
        typer.Option("--trust-remote-code/--no-trust-remote-code", help="Trust remote code when loading HF datasets."),
    ] = None,
    adapter_module: Annotated[
        Optional[str],
        typer.Option("--adapter-module", help="Python file with custom DatasetAdapter registrations."),
    ] = None,

    # ---- Source: local files -----------------------------------------------
    input: Annotated[
        Optional[str],
        typer.Option("--input", "-i", help="Local file, directory, or glob pattern."),
    ] = None,
    text_column: Annotated[
        Optional[str],
        typer.Option("--text-column", help="Column name for text in parquet/jsonl input."),
    ] = None,

    # ---- Output ------------------------------------------------------------
    output: Annotated[
        Optional[str],
        typer.Option("--output", "-o", help="Output directory for parquet shards."),
    ] = None,
    shard_size: Annotated[
        Optional[int],
        typer.Option("--shard-size", help="Documents per parquet shard. [default: 50000]"),
    ] = None,

    # ---- Tokenizer ---------------------------------------------------------
    tokenizer: Annotated[
        Optional[str],
        typer.Option("--tokenizer", help="Tokenizer type: hf | bpe | char. [default: hf]"),
    ] = None,
    tokenizer_path: Annotated[
        Optional[str],
        typer.Option("--tokenizer-path", help="HF model name or saved tokenizer directory. [default: gpt2]"),
    ] = None,
    vocab_size: Annotated[
        Optional[int],
        typer.Option("--vocab-size", help="[bpe] Target vocabulary size. [default: 8000]"),
    ] = None,
    min_frequency: Annotated[
        Optional[int],
        typer.Option("--min-frequency", help="[bpe] Minimum BPE merge frequency. [default: 2]"),
    ] = None,
    save_tokenizer: Annotated[
        Optional[str],
        typer.Option("--save-tokenizer", help="[bpe/char] Save trained tokenizer to this path."),
    ] = None,
    add_special_tokens: Annotated[
        Optional[bool],
        typer.Option("--add-special-tokens/--no-add-special-tokens", help="Prepend BOS / append EOS to each document. [default: true]"),
    ] = None,

    # ---- Filtering ---------------------------------------------------------
    min_length: Annotated[
        Optional[int],
        typer.Option("--min-length", help="Minimum token count; shorter documents are dropped. [default: 32]"),
    ] = None,
) -> None:

    # ------------------------------------------------------------------
    # Build effective config: defaults → YAML/JSON → CLI overrides
    # ------------------------------------------------------------------
    cfg = _load_config(config)
    _apply_cli_overrides(cfg, {
        "hf_dataset":        hf_dataset,
        "hf_config":         hf_config,
        "hf_split":          hf_split,
        "trust_remote_code": trust_remote_code,
        "adapter_module":    adapter_module,
        "input":             input,
        "text_column":       text_column,
        "output":            output,
        "shard_size":        shard_size,
        "tokenizer":         tokenizer,
        "tokenizer_path":    tokenizer_path,
        "vocab_size":        vocab_size,
        "min_frequency":     min_frequency,
        "save_tokenizer":    save_tokenizer,
        "add_special_tokens": add_special_tokens,
        "min_length":        min_length,
    })

    # Validate required fields
    if cfg.hf_dataset is None and cfg.input is None:
        console.print("[red]Provide either --hf-dataset or --input (or set them in a --config file).[/red]")
        raise typer.Exit(1)
    if cfg.output is None:
        console.print("[red]--output is required (or set 'output' in a --config file).[/red]")
        raise typer.Exit(1)
    if cfg.hf_config and cfg.hf_configs:
        console.print("[red]Use either hf_config (single) or hf_configs (list), not both.[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Load optional custom adapter module
    # ------------------------------------------------------------------
    if cfg.adapter_module:
        _load_adapter_module(cfg.adapter_module)

    # ------------------------------------------------------------------
    # Build tokenizer
    # ------------------------------------------------------------------
    local_files: list[Path] = [] if cfg.hf_dataset else _collect_files(cfg.input or "")
    tok = _build_tokenizer(cfg.tokenizer, cfg.tokenizer_path, cfg.vocab_size, cfg.min_frequency, cfg.save_tokenizer, local_files)
    console.print(f"[green]Tokenizer: {tok}[/green]")

    # ------------------------------------------------------------------
    # Stream source documents
    # ------------------------------------------------------------------
    base_out = Path(cfg.output)

    if cfg.hf_configs:
        # Multiple subsets: process each in sequence, each to its own subdirectory
        console.print(f"[bold]Processing {len(cfg.hf_configs)} subsets of {cfg.hf_dataset!r}[/bold]")
        for subset in cfg.hf_configs:
            console.rule(f"[cyan]{subset}[/cyan]")
            out_dir = base_out / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            source_iter = _stream_hf(cfg.hf_dataset, subset, cfg.hf_split, cfg.trust_remote_code)
            _process_and_write(
                source_iter=source_iter,
                source_name=f"{cfg.hf_dataset}/{cfg.hf_split}/{subset}",
                tok=tok,
                out_dir=out_dir,
                shard_size=cfg.shard_size,
                add_special_tokens=cfg.add_special_tokens,
                min_length=cfg.min_length,
            )
    else:
        base_out.mkdir(parents=True, exist_ok=True)
        if cfg.hf_dataset:
            source_iter = _stream_hf(cfg.hf_dataset, cfg.hf_config, cfg.hf_split, cfg.trust_remote_code)
            source_name = f"{cfg.hf_dataset}/{cfg.hf_split}"
        else:
            source_iter = _stream_local(local_files, cfg.text_column)
            source_name = cfg.input or "local"

        _process_and_write(
            source_iter=source_iter,
            source_name=source_name,
            tok=tok,
            out_dir=base_out,
            shard_size=cfg.shard_size,
            add_special_tokens=cfg.add_special_tokens,
            min_length=cfg.min_length,
        )


# ---------------------------------------------------------------------------
# Config loading / merging
# ---------------------------------------------------------------------------

def _load_config(path: Optional[str]) -> PreprocessConfig:
    """Load a YAML or JSON preprocess config, or return defaults if no path given."""
    if path is None:
        return PreprocessConfig()

    p = Path(path)
    if not p.exists():
        console.print(f"[red]Config file not found: {path}[/red]")
        raise typer.Exit(1)

    suffix = p.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError("pip install pyyaml") from e
            raw = yaml.safe_load(p.read_text())
        else:
            raw = json.loads(p.read_text())
    except Exception as e:
        console.print(f"[red]Could not parse config {path}: {e}[/red]")
        raise typer.Exit(1) from e

    try:
        return PreprocessConfig.model_validate(raw)
    except Exception as e:
        console.print(f"[red]Invalid preprocess config: {e}[/red]")
        raise typer.Exit(1) from e


def _apply_cli_overrides(cfg: PreprocessConfig, overrides: dict) -> None:
    """Set any non-None CLI values onto cfg, overriding YAML/defaults."""
    for field, value in overrides.items():
        if value is not None:
            setattr(cfg, field, value)


# ---------------------------------------------------------------------------
# Core processing pipeline
# ---------------------------------------------------------------------------

def _process_and_write(
    source_iter: Iterator[tuple[str, str]],  # (source_id, text)
    source_name: str,
    tok: BaseTokenizer,
    out_dir: Path,
    shard_size: int,
    add_special_tokens: bool,
    min_length: int,
) -> None:
    """Tokenise documents and flush to parquet shards."""

    rows: list[dict] = []
    shard_idx = 0
    total_docs = 0
    total_tokens = 0
    skipped = 0

    def _flush() -> None:
        nonlocal shard_idx
        path = out_dir / f"shard_{shard_idx:05d}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        console.print(
            f"  [dim]shard {shard_idx:05d}: {len(rows):,} docs, "
            f"{sum(r['length'] for r in rows):,} tokens → {path.name}[/dim]"
        )
        shard_idx += 1
        rows.clear()

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Processing {source_name}…", total=None)

        for source_id, text in source_iter:
            text = text.strip()
            if not text:
                continue

            ids = tok.encode(text, add_special_tokens=add_special_tokens)
            if len(ids) < min_length:
                skipped += 1
                continue

            rows.append({
                "source": f"{source_name}:{source_id}",
                "length": len(ids),
                "text": text,
                "tokens": ids,
            })
            total_docs += 1
            total_tokens += len(ids)

            if len(rows) >= shard_size:
                _flush()

            progress.update(
                task,
                description=f"Processing {source_name}… "
                            f"docs={total_docs:,}  tokens={total_tokens / 1e6:.1f}M  "
                            f"shards={shard_idx}",
            )

    if rows:
        _flush()

    console.print(
        f"\n[bold green]Done.[/bold green]  "
        f"{total_docs:,} documents, "
        f"{total_tokens / 1e6:.1f}M tokens, "
        f"{shard_idx} shards.  "
        f"({skipped} docs skipped as too short)"
    )


# ---------------------------------------------------------------------------
# Source iterators
# ---------------------------------------------------------------------------

def _stream_hf(
    repo_id: str,
    config_name: Optional[str],
    split: str,
    trust_remote_code: bool,
) -> Iterator[tuple[str, str]]:
    """Yield (source_id, text) from a HuggingFace dataset in streaming mode."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    adapter: DatasetAdapter = get_adapter(repo_id)
    console.print(f"[cyan]Streaming {repo_id!r} (config={config_name!r}, split={split!r}) …[/cyan]")
    console.print(f"  [dim]Using adapter: {adapter.__class__.__name__}[/dim]")

    ds = load_dataset(
        repo_id,
        name=config_name,
        split=split,
        streaming=True,
        trust_remote_code=trust_remote_code,
    )

    for i, row in enumerate(ds):
        text = adapter.get_text(row, i)
        if text:
            source_id = adapter.get_source_id(row, i)
            yield source_id, text


def _stream_local(
    files: list[Path],
    text_column: Optional[str],
) -> Iterator[tuple[str, str]]:
    """Yield (source_id, text) from local files."""
    for file in files:
        for idx, text in enumerate(_read_texts(file, text_column)):
            yield f"{file.name}:{idx}", text


def _read_texts(file: Path, column: Optional[str]) -> list[str]:
    suffix = file.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(file)
        col = column or "text"
        return df[col].dropna().tolist()
    if suffix in (".json", ".jsonl"):
        texts = []
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj.get(column or "text", ""))
        return texts
    return [file.read_text(encoding="utf-8", errors="replace")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_files(pattern: str) -> list[Path]:
    p = Path(pattern)
    if p.is_file():
        return [p]
    if p.is_dir():
        return (
            sorted(p.glob("*.txt"))
            + sorted(p.glob("*.jsonl"))
            + sorted(p.glob("*.json"))
            + sorted(p.glob("*.parquet"))
        )
    return [Path(f) for f in sorted(_glob.glob(pattern))]


def _build_tokenizer(
    kind: str,
    path: Optional[str],
    vocab_size: int,
    min_freq: int,
    save_path: Optional[str],
    files: list[Path],
) -> BaseTokenizer:
    match kind:
        case "hf":
            if not path:
                console.print("[red]--tokenizer-path required for hf tokenizer[/red]")
                raise typer.Exit(1)
            return HFTokenizer.from_pretrained(path)
        case "bpe":
            if path and Path(path).exists():
                return BPETokenizer.load(path)
            console.print(f"Training BPE tokenizer (vocab_size={vocab_size})…")
            return BPETokenizer.train(
                files=[str(f) for f in files],
                vocab_size=vocab_size,
                min_frequency=min_freq,
                save_path=save_path,
            )
        case "char":
            if path and Path(path).exists():
                return CharTokenizer.load(path)
            console.print("Building char tokenizer from corpus…")
            text = "".join(f.read_text(encoding="utf-8", errors="replace") for f in files)
            return CharTokenizer.from_text(text, save_path=save_path)
        case _:
            console.print(f"[red]Unknown tokenizer: {kind!r}. Use hf | bpe | char[/red]")
            raise typer.Exit(1)


def _load_adapter_module(path: str) -> None:
    """Import a Python file so its @register_adapter decorators run."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Adapter module not found: {path}[/red]")
        raise typer.Exit(1)
    spec = importlib.util.spec_from_file_location("_user_adapters", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    console.print(f"[dim]Loaded adapter module: {path}[/dim]")

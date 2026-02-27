"""
Preprocess text into tokenised parquet shards.

Output schema
-------------
Each output parquet file has four columns:
  source  – str   : "{repo_id}/{split}:{source_id}" or "{filename}:{row_idx}"
  length  – int   : number of tokens
  text    – str   : raw document text
  tokens  – list[int] : token ids

Input modes
-----------
  --hf-dataset  : stream a HuggingFace dataset (recommended for large corpora)
  --input       : tokenise local text / jsonl / parquet files

Examples
--------
# Stream wikimedia/wikipedia (English, 2023-11-01 snapshot)
llm-preprocess \\
    --hf-dataset wikimedia/wikipedia \\
    --hf-config  20231101.en \\
    --hf-split   train \\
    --output     data/wikipedia \\
    --tokenizer  hf \\
    --tokenizer-path gpt2

# Custom dataset with a user-defined adapter module
llm-preprocess \\
    --hf-dataset my-org/my-corpus \\
    --adapter-module adapters/my_adapter.py \\
    --output data/my-corpus \\
    --tokenizer hf --tokenizer-path meta-llama/Meta-Llama-3-8B

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
from light_llm.tokenizer import BPETokenizer, CharTokenizer, HFTokenizer
from light_llm.tokenizer.base import BaseTokenizer

console = Console()
app = typer.Typer(help="Tokenise datasets into parquet shards.")


@app.command()
def main(
    # ---- Source: HF Hub ----
    hf_dataset: Annotated[Optional[str], typer.Option("--hf-dataset", help="HuggingFace dataset repo ID.")] = None,
    hf_config: Annotated[Optional[str], typer.Option("--hf-config", help="Dataset config name (e.g. '20231101.en' for Wikipedia).")] = None,
    hf_split: Annotated[str, typer.Option("--hf-split", help="Dataset split to use.")] = "train",
    trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code when loading HF datasets.")] = False,
    adapter_module: Annotated[Optional[str], typer.Option("--adapter-module", help="Python file with custom DatasetAdapter registrations.")] = None,

    # ---- Source: local files ----
    input: Annotated[Optional[str], typer.Option("--input", "-i", help="Local file, directory, or glob pattern.")] = None,
    text_column: Annotated[Optional[str], typer.Option(help="Column name for text in parquet/jsonl input.")] = None,

    # ---- Output ----
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory for parquet shards.")] = ...,
    shard_size: Annotated[int, typer.Option(help="Documents per parquet shard.")] = 50_000,

    # ---- Tokenizer ----
    tokenizer: Annotated[str, typer.Option(help="Tokenizer type: hf | bpe | char.")] = "hf",
    tokenizer_path: Annotated[Optional[str], typer.Option(help="HF model name or saved tokenizer directory.")] = "gpt2",
    vocab_size: Annotated[int, typer.Option(help="[bpe] Target vocabulary size.")] = 8_000,
    min_frequency: Annotated[int, typer.Option(help="[bpe] Minimum BPE merge frequency.")] = 2,
    save_tokenizer: Annotated[Optional[str], typer.Option(help="[bpe/char] Save trained tokenizer to this path.")] = None,
    add_special_tokens: Annotated[bool, typer.Option(help="Prepend BOS / append EOS to each document.")] = True,

    # ---- Filtering ----
    min_length: Annotated[int, typer.Option(help="Minimum token count; shorter documents are dropped.")] = 32,
) -> None:

    if hf_dataset is None and input is None:
        console.print("[red]Provide either --hf-dataset or --input.[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Load optional custom adapter module
    # ------------------------------------------------------------------
    if adapter_module:
        _load_adapter_module(adapter_module)

    # ------------------------------------------------------------------
    # Build tokenizer
    # ------------------------------------------------------------------
    local_files: list[Path] = [] if hf_dataset else _collect_files(input or "")
    tok = _build_tokenizer(tokenizer, tokenizer_path, vocab_size, min_frequency, save_tokenizer, local_files)
    console.print(f"[green]Tokenizer: {tok}[/green]")

    # ------------------------------------------------------------------
    # Stream source documents
    # ------------------------------------------------------------------
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hf_dataset:
        source_iter = _stream_hf(hf_dataset, hf_config, hf_split, trust_remote_code)
        source_name = f"{hf_dataset}/{hf_split}"
    else:
        source_iter = _stream_local(local_files, text_column)
        source_name = input or "local"

    _process_and_write(
        source_iter=source_iter,
        source_name=source_name,
        tok=tok,
        out_dir=out_dir,
        shard_size=shard_size,
        add_special_tokens=add_special_tokens,
        min_length=min_length,
    )


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

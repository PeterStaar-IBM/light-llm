"""
Core analytics runner for pre-tokenised parquet shards.

Each shard is a parquet file with at least the columns:
  length  – int       : token count for the document
  tokens  – list[int] : token ids

The runner streams over shard files once, accumulating per-shard and/or
global results depending on AnalysisConfig.per_shard / .aggregate.

Output (JSON)
-------------
  length_histogram:
    bins, counts, min, max, mean, median, std, total_docs

  token_frequency:
    top_tokens        list of {token_id, count}
    total_tokens      int
    vocab_size_observed  int
    frequency_distribution  {bins, counts}   # shape of the Zipf distribution

  graph_summary:
    num_nodes, num_edges, total_transitions

  graph_analytics:
    pagerank               list of {token_id, score}
    in_degree_centrality   list of {token_id, score}
    out_degree_centrality  list of {token_id, score}
    betweenness_centrality list of {token_id, score}   (if enabled)
    closeness_centrality   list of {token_id, score}   (if enabled)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console

from .config import AnalysisConfig, GraphAnalyticsConfig
from .graph import TokenGraph

console = Console()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_analysis(
    shard_paths: list[Path],
    cfg: AnalysisConfig,
    out_dir: Path,
    force: bool = False,
) -> None:
    """Stream over *shard_paths*, compute analytics, write JSON to *out_dir*.

    When ``cfg.per_shard`` is True a ``{shard_stem}_analysis.json`` is
    written for every shard.  When ``cfg.aggregate`` is True a single
    ``global_analysis.json`` is written after all shards are processed.

    Lazy mode (default): if the output JSON for a shard already exists it is
    loaded and only plots are regenerated.  Pass ``force=True`` to recompute
    everything from scratch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    global_out = out_dir / "global_analysis.json"
    global_exists = global_out.exists() and not force

    # Accumulators for global/aggregate pass (only needed when recomputing global)
    global_lengths: list[int] = []
    global_counter: Counter[int] = Counter()
    global_graph: Optional[TokenGraph] = (
        TokenGraph()
        if (cfg.aggregate and cfg.analytics.token_graph and not global_exists)
        else None
    )

    for shard_path in shard_paths:
        shard_out = out_dir / f"{shard_path.stem}_analysis.json"
        shard_exists = shard_out.exists() and not force

        # When both per-shard result and global result are already on disk we
        # can skip computation entirely and just regenerate plots.
        if shard_exists and (not cfg.aggregate or global_exists):
            console.print(f"[dim]{shard_path.name} — cached, plots only[/dim]")
            if cfg.per_shard:
                _save_plots(json.loads(shard_out.read_text()), out_dir / shard_path.stem)
            continue

        console.print(f"[cyan]Analysing {shard_path.name}…[/cyan]")
        df = _read_shard(shard_path, cfg)
        shard_result: dict = {}

        # ---- Length histogram ----------------------------------------
        if cfg.analytics.length_histogram:
            lengths = df["length"].tolist()
            if cfg.aggregate and not global_exists:
                global_lengths.extend(lengths)
            if cfg.per_shard:
                shard_result["length_histogram"] = _length_histogram(
                    lengths, cfg.histogram_bins
                )

        # ---- Token frequency -----------------------------------------
        if cfg.analytics.token_frequency:
            counter = _count_tokens(df)
            if cfg.aggregate and not global_exists:
                global_counter += counter
            if cfg.per_shard:
                shard_result["token_frequency"] = _format_token_frequency(
                    counter, cfg.top_k_tokens
                )

        # ---- Token bigram graph --------------------------------------
        if cfg.analytics.token_graph:
            graph = _build_graph(df)
            if cfg.aggregate and global_graph is not None:
                global_graph.merge(graph)
            if cfg.per_shard:
                shard_result["graph_summary"] = graph.summary()
                if _any_graph_analytics(cfg.analytics.graph):
                    shard_result["graph_analytics"] = _graph_analytics(
                        graph, cfg.analytics.graph
                    )

        if cfg.per_shard:
            if not shard_exists:
                _save_json(shard_out, shard_result)
                console.print(f"  [dim]→ {shard_out.name}[/dim]")
            _save_plots(shard_result, out_dir / shard_path.stem)

    # ---- Global / aggregate result -----------------------------------
    if cfg.aggregate:
        if global_exists:
            console.print(f"[dim]global_analysis.json — cached, plots only[/dim]")
            _save_plots(json.loads(global_out.read_text()), out_dir / "global")
        else:
            global_result: dict = {}

            if cfg.analytics.length_histogram and global_lengths:
                global_result["length_histogram"] = _length_histogram(
                    global_lengths, cfg.histogram_bins
                )

            if cfg.analytics.token_frequency:
                global_result["token_frequency"] = _format_token_frequency(
                    global_counter, cfg.top_k_tokens
                )

            if cfg.analytics.token_graph and global_graph is not None:
                global_result["graph_summary"] = global_graph.summary()
                if _any_graph_analytics(cfg.analytics.graph):
                    global_result["graph_analytics"] = _graph_analytics(
                        global_graph, cfg.analytics.graph
                    )

            _save_json(global_out, global_result)
            console.print(f"[green]Global  → {global_out.name}[/green]")
            _save_plots(global_result, out_dir / "global")


# ---------------------------------------------------------------------------
# Shard reading
# ---------------------------------------------------------------------------


def _read_shard(path: Path, cfg: AnalysisConfig) -> pd.DataFrame:
    cols: list[str] = []
    if cfg.analytics.length_histogram:
        cols.append("length")
    if cfg.analytics.token_frequency or cfg.analytics.token_graph:
        cols.append("tokens")
    return pd.read_parquet(path, columns=cols if cols else None)


# ---------------------------------------------------------------------------
# Length histogram
# ---------------------------------------------------------------------------


def _length_histogram(lengths: list[int], bins: int) -> dict:
    arr = np.array(lengths, dtype=np.int64)
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(arr.std()), 4),
        "total_docs": len(lengths),
    }


# ---------------------------------------------------------------------------
# Token frequency
# ---------------------------------------------------------------------------


def _count_tokens(df: pd.DataFrame) -> Counter[int]:
    counter: Counter[int] = Counter()
    for tokens in df["tokens"]:
        counter.update(tokens)
    return counter


def _format_token_frequency(
    counter: Counter[int], top_k: Optional[int]
) -> dict:
    total = sum(counter.values())
    vocab_size = len(counter)
    most_common = counter.most_common(top_k)

    # Distribution of per-token counts — reveals the Zipfian shape.
    freq_arr = np.array(list(counter.values()), dtype=np.int64)
    n_bins = min(50, max(1, vocab_size))
    freq_counts, freq_edges = np.histogram(freq_arr, bins=n_bins)

    return {
        "top_tokens": [
            {"token_id": int(tid), "count": int(cnt)} for tid, cnt in most_common
        ],
        "total_tokens": int(total),
        "vocab_size_observed": int(vocab_size),
        "frequency_distribution": {
            "bins": freq_edges.tolist(),
            "counts": freq_counts.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Token bigram graph
# ---------------------------------------------------------------------------


def _build_graph(df: pd.DataFrame) -> TokenGraph:
    graph = TokenGraph()
    for tokens in df["tokens"]:
        graph.add_sequence(tokens)
    return graph


def _any_graph_analytics(cfg: GraphAnalyticsConfig) -> bool:
    return any(
        [
            cfg.pagerank,
            cfg.in_degree_centrality,
            cfg.out_degree_centrality,
            cfg.betweenness_centrality,
            cfg.closeness_centrality,
        ]
    )


def _graph_analytics(graph: TokenGraph, cfg: GraphAnalyticsConfig) -> dict:
    try:
        import networkx as nx
    except ImportError:
        console.print(
            "[yellow]networkx not installed — skipping graph analytics. "
            "Install with: uv pip install 'light-llm[analysis]'[/yellow]"
        )
        return {}
    try:
        import scipy  # noqa: F401 — networkx PageRank requires scipy
    except ImportError:
        console.print(
            "[yellow]scipy not installed — skipping graph analytics. "
            "Install with: uv pip install 'light-llm[analysis]'[/yellow]"
        )
        return {}

    G = graph.to_networkx()
    result: dict = {}
    top_k = cfg.top_k

    if cfg.pagerank:
        console.print("  [dim]PageRank…[/dim]")
        result["pagerank"] = _top_k_scores(nx.pagerank(G, weight="weight"), top_k)

    if cfg.in_degree_centrality:
        console.print("  [dim]In-degree centrality…[/dim]")
        result["in_degree_centrality"] = _top_k_scores(
            nx.in_degree_centrality(G), top_k
        )

    if cfg.out_degree_centrality:
        console.print("  [dim]Out-degree centrality…[/dim]")
        result["out_degree_centrality"] = _top_k_scores(
            nx.out_degree_centrality(G), top_k
        )

    if cfg.betweenness_centrality:
        console.print("  [dim]Betweenness centrality (slow for large graphs)…[/dim]")
        result["betweenness_centrality"] = _top_k_scores(
            nx.betweenness_centrality(G, weight="weight"), top_k
        )

    if cfg.closeness_centrality:
        console.print("  [dim]Closeness centrality (slow for large graphs)…[/dim]")
        result["closeness_centrality"] = _top_k_scores(
            nx.closeness_centrality(G), top_k
        )

    return result


def _top_k_scores(scores: dict, top_k: int) -> list[dict]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"token_id": int(tid), "score": round(float(score), 8)} for tid, score in items]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save_plots(result: dict, stem: Path) -> None:
    """Save PNG plots for any analytics present in *result*."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print(
            "[yellow]matplotlib not installed — skipping plots. "
            "Install with: uv pip install 'light-llm[analysis]'[/yellow]"
        )
        return

    if "length_histogram" in result:
        _plot_length_histogram(
            result["length_histogram"],
            stem.parent / f"{stem.name}_length_histogram.png",
            plt,
        )
    if "token_frequency" in result:
        _plot_token_counts(
            result["token_frequency"],
            stem.parent / f"{stem.name}_token_counts.png",
            plt,
        )


def _plot_length_histogram(data: dict, out_path: Path, plt) -> None:
    bins = data["bins"]    # bin edges, len = n_bins + 1
    counts = data["counts"]
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
    widths = [bins[i + 1] - bins[i] for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(centers, counts, width=widths, align="center", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Document length (tokens)")
    ax.set_ylabel("Number of documents")
    ax.set_title(
        f"Document length distribution  "
        f"(n={data['total_docs']:,}  mean={data['mean']:.1f}  median={data['median']:.1f})"
    )
    # Use log x-scale when bins span more than 2 orders of magnitude.
    positive_edges = [b for b in bins if b > 0]
    if positive_edges and max(positive_edges) / min(positive_edges) > 100:
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"  [dim]→ {out_path.name}[/dim]")


def _plot_token_counts(data: dict, out_path: Path, plt) -> None:
    # Sort by token_id so x-axis represents position in the vocabulary.
    top_tokens = sorted(data["top_tokens"], key=lambda t: t["token_id"])
    token_ids = [t["token_id"] for t in top_tokens]
    counts = [t["count"] for t in top_tokens]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(token_ids, counts, s=4, alpha=0.6)
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Token frequency by ID  "
        f"(vocab_size_observed={data['vocab_size_observed']:,}  "
        f"total_tokens={data['total_tokens']:,})"
    )
    # Log y-scale makes Zipfian distributions readable.
    if counts and max(counts) / max(1, min(counts)) > 100:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.print(f"  [dim]→ {out_path.name}[/dim]")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))

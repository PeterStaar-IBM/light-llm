"""
Pydantic config for llm-analyse.

All fields are optional where sensible so that a YAML config can supply a
subset of fields and CLI flags can override individual ones.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GraphAnalyticsConfig(BaseModel):
    """Controls which graph-level metrics are computed on the token bigram graph."""

    pagerank: bool = True
    in_degree_centrality: bool = True
    out_degree_centrality: bool = True
    # The two metrics below are O(V·E) / O(V·(V+E)) — disable for large vocabs.
    betweenness_centrality: bool = False
    closeness_centrality: bool = False
    # Only the top-k nodes (by score) are written to the JSON output.
    top_k: int = 1000


class AnalyticsConfig(BaseModel):
    """Selects which analytics to compute."""

    length_histogram: bool = True   # distribution of document lengths in tokens
    token_frequency: bool = True    # per-token occurrence counts + Zipfian shape
    token_graph: bool = True        # directed weighted bigram graph

    # Graph analytics are only active when token_graph is True.
    graph: GraphAnalyticsConfig = Field(default_factory=GraphAnalyticsConfig)


class AnalysisConfig(BaseModel):
    # ---- Input / Output ----
    input: Optional[str] = None   # directory of parquet shards (required at runtime)
    output: str = "analysis"      # directory for JSON results

    # ---- Scope ----
    per_shard: bool = True    # write {shard}_analysis.json for every shard
    aggregate: bool = True    # write global_analysis.json across all shards

    # ---- Analytics ----
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)

    # ---- Histogram options ----
    # int → number of equal-width bins; list[int] → explicit bin edges.
    histogram_bins: int | list[int] = 50

    # ---- Token frequency options ----
    # None → include all observed tokens; set to e.g. 10_000 to cap the output.
    top_k_tokens: Optional[int] = None

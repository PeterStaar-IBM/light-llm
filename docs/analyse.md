# llm-analyse

Analyses pre-tokenised parquet shards produced by `llm-preprocess` and writes the results as JSON files.

Three families of analytics are available:

| analytic | description |
|---|---|
| `length_histogram` | distribution of document lengths in tokens |
| `token_frequency` | per-token occurrence counts + Zipfian frequency distribution |
| `token_graph` | directed weighted graph of consecutive token pairs, with optional graph metrics |

Results can be computed **per shard**, **globally** (aggregated across all shards), or both.

---

## Using a config file

All options can be stored in a YAML or JSON file and passed via `--config` / `-c`.
**CLI flags always override values from the file.**

```bash
# Run exactly as configured
llm-analyse --config configs/analyse/example.yaml

# Override input/output on the CLI
llm-analyse --config configs/analyse/example.yaml \
            --input data/wikipedia/train \
            --output analysis/wikipedia
```

A ready-to-use config lives at `configs/analyse/example.yaml`.

---

## Inline (no config file)

```bash
llm-analyse --input data/wikipedia/train --output analysis/wikipedia
```

This runs all three analytics with their defaults (per-shard + aggregate, graph analytics enabled, top-1000 nodes per metric).

---

## Output layout

```
analysis/wikipedia/
  shard_00000_analysis.json    # per-shard result  (per_shard: true)
  shard_00001_analysis.json
  …
  global_analysis.json         # aggregate result  (aggregate: true)
```

Each JSON file has up to three top-level keys depending on which analytics are enabled:

```json
{
  "length_histogram": { … },
  "token_frequency":  { … },
  "graph_summary":    { … },
  "graph_analytics":  { … }
}
```

### `length_histogram`

```json
{
  "bins":       [32, 132, 232, …],
  "counts":     [1234, 5678, …],
  "min":        32,
  "max":        8192,
  "mean":       874.3,
  "median":     612.0,
  "std":        620.5,
  "total_docs": 50000
}
```

### `token_frequency`

```json
{
  "top_tokens": [
    {"token_id": 198, "count": 1234567},
    {"token_id": 628, "count":  987654},
    …
  ],
  "total_tokens":        48320000,
  "vocab_size_observed": 50257,
  "frequency_distribution": {
    "bins":   [1, 1000, 2000, …],
    "counts": [32000, 4500, …]
  }
}
```

`frequency_distribution` shows the shape of the Zipfian curve: how many tokens occur 1–1000 times, 1000–2000 times, etc.

### `graph_summary`

```json
{
  "num_nodes":         50257,
  "num_edges":         2480193,
  "total_transitions": 48269743
}
```

### `graph_analytics`

```json
{
  "pagerank": [
    {"token_id": 198, "score": 0.00312},
    …
  ],
  "in_degree_centrality":  [{"token_id": 628, "score": 0.0021}, …],
  "out_degree_centrality": [{"token_id": 198, "score": 0.0031}, …]
}
```

Each list contains at most `top_k` entries (default 1000), sorted by score descending.

---

## Token bigram graph

The token graph is a directed weighted graph where an edge `(A → B, w)` means token `A` was immediately followed by token `B` exactly `w` times across the processed shards.

**Example**: the sequence `[3, 5, 9, 3, 5]` produces edges `(3→5, 2)`, `(5→9, 1)`, `(9→3, 1)`.

Graphs from individual shards are merged efficiently — the global graph is built incrementally as each shard is processed, with O(|edges|) merge cost.

### Available graph metrics

| metric | notes |
|---|---|
| `pagerank` | weighted PageRank; high score = token is a common "destination" |
| `in_degree_centrality` | fraction of tokens that point *to* this token |
| `out_degree_centrality` | fraction of tokens that this token points *to* |
| `betweenness_centrality` | how often the token lies on shortest paths; O(V·E) — **slow for large vocabs** |
| `closeness_centrality` | inverse mean distance to all other tokens; O(V·(V+E)) — **slow for large vocabs** |

Graph analytics require `networkx` and `scipy`:

```bash
uv pip install 'light-llm[analysis]'
```

---

## Config reference

```yaml
# ------ Input / Output -------------------------------------
input:  data/wikipedia/train     # directory of parquet shards  (required)
output: analysis/wikipedia       # directory for JSON results   [default: analysis]

# ------ Scope ----------------------------------------------
per_shard: true    # write {shard_name}_analysis.json per shard  [default: true]
aggregate: true    # write global_analysis.json across all shards [default: true]

# ------ Analytics ------------------------------------------
analytics:
  length_histogram: true    # [default: true]
  token_frequency:  true    # [default: true]
  token_graph:      true    # [default: true]

  graph:                        # only active when token_graph: true
    pagerank:               true   # [default: true]
    in_degree_centrality:   true   # [default: true]
    out_degree_centrality:  true   # [default: true]
    betweenness_centrality: false  # [default: false] — O(V·E), slow
    closeness_centrality:   false  # [default: false] — O(V·(V+E)), slow
    top_k: 1000                    # top-k nodes per metric in JSON [default: 1000]

# ------ Histogram options ----------------------------------
histogram_bins: 50    # bins for the length histogram [default: 50]

# ------ Token frequency options ----------------------------
top_k_tokens: null    # null = all tokens; or e.g. 10000  [default: null]
```

---

## All CLI options

| option | default | description |
|---|---|---|
| `--config` / `-c` | — | YAML/JSON config file; CLI flags override |
| `--input` / `-i` | — | Directory of parquet shards to analyse (required) |
| `--output` / `-o` | `analysis` | Output directory for JSON results |
| `--per-shard` / `--no-per-shard` | `true` | Write per-shard JSON files |
| `--aggregate` / `--no-aggregate` | `true` | Write global aggregate JSON file |

Analytics selection and all other options are config-file only.

---

## Using `TokenGraph` programmatically

```python
from light_llm.analysis.graph import TokenGraph

# Build from token sequences
g = TokenGraph()
g.add_sequence([3, 5, 9, 3, 5])   # O(n) per call

# Merge two graphs  (O(|other edges|))
g2 = TokenGraph()
g2.add_sequence([9, 3, 7])
g.merge(g2)

# Inspect
print(g)          # TokenGraph(nodes=4, edges=4, transitions=7)
print(g.summary())

# Iterate edges
for src, dst, weight in g.edges():
    print(src, "->", dst, ":", weight)

# Convert to networkx for custom analytics
G = g.to_networkx()   # nx.DiGraph with 'weight' edge attribute

# Serialise / deserialise
data = g.to_dict()    # JSON-compatible dict
g3   = TokenGraph.from_dict(data)
```

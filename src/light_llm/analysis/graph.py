"""
TokenGraph — directed weighted graph of token bigrams.

Design goals
------------
* O(1) amortised edge insertion via nested defaultdicts.
* O(|other edges|) merge — simply iterate and accumulate.
* Lazy conversion to networkx for analytics (networkx is an optional dep).
* JSON-serialisable via to_dict() / from_dict() for persistence.

Example
-------
>>> g = TokenGraph()
>>> g.add_sequence([3, 5, 9, 3, 5])
>>> g.summary()
{'num_nodes': 3, 'num_edges': 3, 'total_transitions': 4}
>>> list(g.edges())          # (src, dst, weight)
[(3, 5, 2), (5, 9, 1), (9, 3, 1)]
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterator, Sequence


class TokenGraph:
    """Directed weighted graph of consecutive token pairs (bigrams).

    Internally stored as an adjacency dict:
        _edges[src][dst] = count

    This layout gives O(1) lookups and insertions and O(|edges|) merge.
    """

    def __init__(self) -> None:
        self._edges: defaultdict[int, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._total_transitions: int = 0

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_sequence(self, tokens: Sequence[int]) -> None:
        """Register all consecutive token pairs from *tokens*.

        Parameters
        ----------
        tokens:
            Any integer sequence (list, numpy array, …).
        """
        n = len(tokens)
        if n < 2:
            return
        edges = self._edges
        for i in range(n - 1):
            edges[tokens[i]][tokens[i + 1]] += 1
        self._total_transitions += n - 1

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(self, other: TokenGraph) -> None:
        """Merge *other* into this graph **in-place** (O(|other edges|)).

        Safe to call with self == other (doubles weights).
        """
        for src, dsts in other._edges.items():
            target = self._edges[src]
            for dst, w in dsts.items():
                target[dst] += w
        self._total_transitions += other._total_transitions

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def num_nodes(self) -> int:
        """Number of distinct token ids that appear as source *or* target."""
        nodes: set[int] = set(self._edges.keys())
        for dsts in self._edges.values():
            nodes.update(dsts.keys())
        return len(nodes)

    def num_edges(self) -> int:
        """Number of distinct (src, dst) pairs (regardless of weight)."""
        return sum(len(dsts) for dsts in self._edges.values())

    def edges(self) -> Iterator[tuple[int, int, int]]:
        """Yield ``(src, dst, weight)`` triples for every edge."""
        for src, dsts in self._edges.items():
            for dst, w in dsts.items():
                yield src, dst, w

    def summary(self) -> dict[str, int]:
        """Return a compact summary dict (JSON-safe)."""
        return {
            "num_nodes": self.num_nodes(),
            "num_edges": self.num_edges(),
            "total_transitions": self._total_transitions,
        }

    # ------------------------------------------------------------------
    # Conversion — networkx
    # ------------------------------------------------------------------

    def to_networkx(self) -> Any:
        """Convert to a ``networkx.DiGraph`` with a ``weight`` edge attribute.

        Requires ``networkx`` (optional dependency):
            uv pip install networkx
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "networkx is required for graph analytics. "
                "Install it with: uv pip install networkx"
            ) from exc

        G: Any = nx.DiGraph()
        for src, dst, w in self.edges():
            G.add_edge(src, dst, weight=w)
        return G

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, int]]:
        """Serialise edges to a plain ``{str_src: {str_dst: weight}}`` dict."""
        return {
            str(src): {str(dst): w for dst, w in dsts.items()}
            for src, dsts in self._edges.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, int]]) -> TokenGraph:
        """Reconstruct a :class:`TokenGraph` from a serialised dict."""
        g = cls()
        for src_s, dsts in data.items():
            src = int(src_s)
            for dst_s, w in dsts.items():
                g._edges[src][int(dst_s)] = w
                g._total_transitions += w
        return g

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TokenGraph("
            f"nodes={self.num_nodes()}, "
            f"edges={self.num_edges()}, "
            f"transitions={self._total_transitions})"
        )

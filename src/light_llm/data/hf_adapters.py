"""
HuggingFace dataset adapters.

Each adapter knows how to extract `text` and a `source_id` from a single
row of a specific HuggingFace dataset.  The registry maps ``repo_id`` →
adapter instance so the preprocess tool can dispatch automatically.

Built-in adapters
-----------------
  wikimedia/wikipedia          → WikipediaAdapter
  HuggingFaceFW/fineweb        → FineWebAdapter
  HuggingFaceFW/fineweb-edu    → FineWebAdapter
  allenai/c4                   → C4Adapter
  togethercomputer/RedPajama-* → RedPajamaAdapter
  EleutherAI/pile              → PileAdapter
  openwebtext / bookcorpus     → TextColumnAdapter  (plain "text" column)
  tiiuae/falcon-refinedweb     → FalconRefinedWebAdapter

Default fallback
----------------
If no adapter is registered for a given repo_id, ``DefaultAdapter`` is
used: it tries common column names (``text``, ``content``, ``body``,
``article``, ``passage``) in order.

Adding custom adapters
----------------------
    from light_llm.data.hf_adapters import DatasetAdapter, register_adapter

    @register_adapter("my-org/my-dataset")
    class MyAdapter(DatasetAdapter):
        def get_text(self, row: dict, index: int) -> str | None:
            return row.get("article_body", "").strip() or None

        def get_source_id(self, row: dict, index: int) -> str:
            return row.get("doc_id", str(index))
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DatasetAdapter:
    """
    Interface for extracting a (text, source_id) pair from one dataset row.

    Override both methods to support a new dataset.
    """

    def get_text(self, row: dict, index: int) -> Optional[str]:
        """
        Return the main textual content of this row, or ``None`` to skip it.
        """
        raise NotImplementedError

    def get_source_id(self, row: dict, index: int) -> str:
        """Return a stable, human-readable identifier for this row."""
        return str(index)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, DatasetAdapter] = {}


def register_adapter(*repo_ids: str):
    """
    Class decorator that registers an adapter for one or more repo IDs.

    Usage::

        @register_adapter("my-org/my-dataset", "my-org/my-dataset-v2")
        class MyAdapter(DatasetAdapter):
            ...
    """
    def decorator(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        instance = cls()
        for repo_id in repo_ids:
            _REGISTRY[repo_id] = instance
        return cls

    return decorator


def get_adapter(repo_id: str) -> DatasetAdapter:
    """
    Return the registered adapter for ``repo_id``.

    Falls back to ``DefaultAdapter`` if none is registered.
    Matches are tried in order:
      1. Exact key match.
      2. Prefix match (useful for versioned datasets, e.g. ``wikimedia/wikipedia``).
      3. ``DefaultAdapter``.
    """
    if repo_id in _REGISTRY:
        return _REGISTRY[repo_id]
    for key, adapter in _REGISTRY.items():
        if repo_id.startswith(key):
            return adapter
    return DefaultAdapter()


def list_adapters() -> list[str]:
    """Return all registered repo IDs."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------

class DefaultAdapter(DatasetAdapter):
    """
    Fallback adapter.  Tries common column names in order:
    ``text``, ``content``, ``body``, ``article``, ``passage``.
    """

    _CANDIDATES = ("text", "content", "body", "article", "passage")

    def get_text(self, row: dict, index: int) -> Optional[str]:
        for col in self._CANDIDATES:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    def get_source_id(self, row: dict, index: int) -> str:
        for col in ("id", "doc_id", "url", "title"):
            if col in row and row[col]:
                return str(row[col])
        return str(index)


@register_adapter("wikimedia/wikipedia")
class WikipediaAdapter(DatasetAdapter):
    """
    wikimedia/wikipedia – columns: id, url, title, text.
    """

    def get_text(self, row: dict, index: int) -> Optional[str]:
        text = row.get("text", "").strip()
        return text or None

    def get_source_id(self, row: dict, index: int) -> str:
        return str(row.get("id", index))


@register_adapter(
    "HuggingFaceFW/fineweb",
    "HuggingFaceFW/fineweb-edu",
    "HuggingFaceFW/fineweb-edu-score-2",
)
class FineWebAdapter(DatasetAdapter):
    """
    FineWeb / FineWeb-Edu – columns: id, text, url, date, language, …
    """

    def get_text(self, row: dict, index: int) -> Optional[str]:
        text = row.get("text", "").strip()
        return text or None

    def get_source_id(self, row: dict, index: int) -> str:
        return str(row.get("id", index))


@register_adapter("allenai/c4")
class C4Adapter(DatasetAdapter):
    """
    allenai/c4 – columns: text, timestamp, url.
    """

    def get_text(self, row: dict, index: int) -> Optional[str]:
        text = row.get("text", "").strip()
        return text or None

    def get_source_id(self, row: dict, index: int) -> str:
        return str(row.get("url", index))


@register_adapter(
    "togethercomputer/RedPajama-Data-1T",
    "togethercomputer/RedPajama-Data-V2",
)
class RedPajamaAdapter(DatasetAdapter):
    """
    RedPajama – columns: text, meta (dict with source, timestamp, …).
    """

    def get_text(self, row: dict, index: int) -> Optional[str]:
        text = row.get("text", "").strip()
        return text or None

    def get_source_id(self, row: dict, index: int) -> str:
        meta = row.get("meta", {})
        if isinstance(meta, dict):
            return str(meta.get("url", meta.get("timestamp", index)))
        return str(index)


@register_adapter("EleutherAI/pile", "EleutherAI/pile-uncopyrighted")
class PileAdapter(DatasetAdapter):
    """
    The Pile – columns: text, meta.
    """

    def get_text(self, row: dict, index: int) -> Optional[str]:
        text = row.get("text", "").strip()
        return text or None

    def get_source_id(self, row: dict, index: int) -> str:
        meta = row.get("meta", {})
        if isinstance(meta, dict):
            return str(meta.get("pile_set_name", index))
        return str(index)


@register_adapter("tiiuae/falcon-refinedweb")
class FalconRefinedWebAdapter(DatasetAdapter):
    """
    Falcon RefinedWeb – columns: content, url, timestamp, dump, segment.
    """

    def get_text(self, row: dict, index: int) -> Optional[str]:
        text = row.get("content", "").strip()
        return text or None

    def get_source_id(self, row: dict, index: int) -> str:
        return str(row.get("url", index))


@register_adapter(
    "openwebtext",
    "Skylion007/openwebtext",
    "bookcorpus",
    "bookcorpusopenai/books1",
)
class TextColumnAdapter(DatasetAdapter):
    """Generic adapter for datasets with a plain ``text`` column."""

    def get_text(self, row: dict, index: int) -> Optional[str]:
        text = row.get("text", "").strip()
        return text or None

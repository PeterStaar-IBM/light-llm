"""
Pydantic config for llm-preprocess.

All fields are optional (with sensible defaults) so that a YAML config file
can supply a subset of fields and CLI flags can override individual ones.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PreprocessConfig(BaseModel):
    # ---- Source: HF Hub ----
    hf_dataset: Optional[str] = None
    # Single subset/config name (e.g. "20231101.en" for wikimedia/wikipedia).
    hf_config: Optional[str] = None
    # Multiple subset names – processed in sequence, each written to its own
    # subdirectory under `output/`.  Mutually exclusive with hf_config.
    hf_configs: Optional[list[str]] = None
    hf_split: str = "train"
    trust_remote_code: bool = False
    adapter_module: Optional[str] = None

    # ---- Source: local files ----
    input: Optional[str] = None       # file, directory, or glob
    text_column: Optional[str] = None  # column to read in parquet/jsonl input

    # ---- Output ----
    output: Optional[str] = None      # required at runtime (CLI or YAML)
    shard_size: int = 50_000

    # ---- Tokenizer ----
    tokenizer: str = "hf"
    tokenizer_path: Optional[str] = "gpt2"
    vocab_size: int = 8_000
    min_frequency: int = 2
    save_tokenizer: Optional[str] = None
    add_special_tokens: bool = True

    # ---- Filtering ----
    min_length: int = 32

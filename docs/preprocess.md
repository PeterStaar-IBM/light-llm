# llm-preprocess

Tokenises text into parquet shards ready for training.

Each output shard is a parquet file with four columns:

| column  | type       | description                                      |
|---------|------------|--------------------------------------------------|
| source  | str        | `"{dataset}/{split}:{id}"` or `"{file}:{row}"`  |
| length  | int        | number of tokens                                  |
| text    | str        | raw document text                                 |
| tokens  | list[int]  | token ids                                         |

---

## Streaming a HuggingFace dataset

```bash
llm-preprocess \
    --hf-dataset wikimedia/wikipedia \
    --hf-config  20231101.en \
    --hf-split   train \
    --output     data/wikipedia \
    --tokenizer  hf \
    --tokenizer-path gpt2
```

Any public HuggingFace dataset works. The tool ships with built-in adapters for common ones:

| repo id                              | adapter                  |
|--------------------------------------|--------------------------|
| `wikimedia/wikipedia`                | WikipediaAdapter         |
| `HuggingFaceFW/fineweb`              | FineWebAdapter           |
| `HuggingFaceFW/fineweb-edu`          | FineWebAdapter           |
| `allenai/c4`                         | C4Adapter                |
| `togethercomputer/RedPajama-Data-1T` | RedPajamaAdapter         |
| `EleutherAI/pile`                    | PileAdapter              |
| `tiiuae/falcon-refinedweb`           | FalconRefinedWebAdapter  |
| `openwebtext` / `bookcorpus`         | TextColumnAdapter        |
| anything else                        | DefaultAdapter (tries `text`, `content`, `body`, …) |

### Custom adapter for an unlisted dataset

Create a Python file that registers an adapter for your dataset, then pass it via `--adapter-module`:

```python
# my_adapters.py
from light_llm.data.hf_adapters import DatasetAdapter, register_adapter

@register_adapter("my-org/my-corpus")
class MyAdapter(DatasetAdapter):
    def get_text(self, row: dict, index: int) -> str | None:
        return row.get("article_body", "").strip() or None

    def get_source_id(self, row: dict, index: int) -> str:
        return str(row.get("doc_id", index))
```

```bash
llm-preprocess \
    --hf-dataset   my-org/my-corpus \
    --adapter-module my_adapters.py \
    --output       data/my-corpus \
    --tokenizer    hf --tokenizer-path gpt2
```

---

## Local text files

```bash
# Single file
llm-preprocess --input data/raw/train.txt --output data/tokens \
               --tokenizer hf --tokenizer-path gpt2

# Directory of .txt / .jsonl / .parquet files
llm-preprocess --input data/raw/ --output data/tokens \
               --tokenizer hf --tokenizer-path gpt2

# Glob pattern
llm-preprocess --input "data/raw/*.jsonl" --output data/tokens \
               --tokenizer hf --tokenizer-path gpt2
```

For jsonl/parquet input, use `--text-column` to specify which column holds the text (default: `text`).

---

## Training a custom BPE vocabulary

```bash
llm-preprocess \
    --input          data/raw/ \
    --output         data/tokens \
    --tokenizer      bpe \
    --vocab-size     16000 \
    --save-tokenizer tokenizer/bpe-16k
```

Load the same tokenizer in later runs:

```bash
llm-preprocess --input data/raw/ --output data/tokens \
               --tokenizer bpe --tokenizer-path tokenizer/bpe-16k
```

---

## All options

| option                | default   | description                                        |
|-----------------------|-----------|----------------------------------------------------|
| `--hf-dataset`        | —         | HuggingFace repo ID to stream                      |
| `--hf-config`         | —         | Dataset config name (e.g. `20231101.en`)           |
| `--hf-split`          | `train`   | Dataset split                                      |
| `--adapter-module`    | —         | Python file with custom `@register_adapter` calls  |
| `--input`             | —         | Local file, directory, or glob                     |
| `--text-column`       | `text`    | Column name for text in jsonl/parquet input        |
| `--output`            | required  | Output directory                                   |
| `--tokenizer`         | `hf`      | `hf` / `bpe` / `char`                             |
| `--tokenizer-path`    | `gpt2`    | HF model name or saved tokenizer path              |
| `--vocab-size`        | `8000`    | BPE target vocabulary size                         |
| `--save-tokenizer`    | —         | Where to save a newly trained bpe/char tokenizer   |
| `--add-special-tokens`| `true`    | Prepend BOS, append EOS to each document           |
| `--shard-size`        | `50000`   | Documents per parquet file                         |
| `--min-length`        | `32`      | Drop documents shorter than this many tokens       |

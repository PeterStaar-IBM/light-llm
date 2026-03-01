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

## Using a config file

All options can be stored in a YAML or JSON file and passed via `--config` / `-c`.
**CLI flags always override values from the file**, so you can keep a base config and vary individual parameters on the command line.

```bash
# Run exactly as configured
llm-preprocess --config configs/preprocess/wikipedia_en.yaml

# Override the output dir and split (e.g. to produce a validation set)
llm-preprocess --config configs/preprocess/wikipedia_en.yaml \
               --output data/wikipedia/val --hf-split test
```

Ready-to-use configs live in `configs/preprocess/`:

| file                                       | description                                  |
|--------------------------------------------|----------------------------------------------|
| `configs/preprocess/wikipedia_en.yaml`     | Wikipedia English (single subset)            |
| `configs/preprocess/wikipedia_multilang.yaml` | Wikipedia en + fr + de (three subsets)    |
| `configs/preprocess/fineweb.yaml`          | FineWeb full dump                            |

A minimal preprocess config looks like:

```yaml
hf_dataset:    wikimedia/wikipedia
hf_config:     20231101.en
hf_split:      train
output:        data/wikipedia/train
tokenizer:     hf
tokenizer_path: gpt2
```

---

## Streaming a HuggingFace dataset

```bash
llm-preprocess \
    --hf-dataset wikimedia/wikipedia \
    --hf-config  20231101.en \
    --hf-split   train \
    --output     data/wikipedia/train \
    --tokenizer  hf \
    --tokenizer-path gpt2
```

### Dataset subsets

Many HuggingFace datasets expose named **subsets** (also called configs).
Pass a single subset name with `--hf-config`:

| dataset                              | example subset            |
|--------------------------------------|---------------------------|
| `wikimedia/wikipedia`                | `20231101.en`, `20231101.fr`, `20231101.de` |
| `allenai/c4`                         | `en`, `realnewslike`      |
| `HuggingFaceFW/fineweb-edu`          | `CC-MAIN-2024-10`         |

```bash
# French Wikipedia
llm-preprocess \
    --hf-dataset wikimedia/wikipedia \
    --hf-config  20231101.fr \
    --hf-split   train \
    --output     data/wikipedia-fr/train \
    --tokenizer  hf --tokenizer-path gpt2
```

### Processing multiple subsets in one run

Use `hf_configs` (a YAML list) instead of `hf_config` to process several subsets
back-to-back. Each subset is written to its own subdirectory under `output/`:

```yaml
# configs/preprocess/wikipedia_multilang.yaml
hf_dataset:  wikimedia/wikipedia
hf_configs:
  - 20231101.en
  - 20231101.fr
  - 20231101.de
hf_split:    train
output:      data/wikipedia/train   # → train/20231101.en/, train/20231101.fr/, …
tokenizer:   hf
tokenizer_path: gpt2
```

```bash
llm-preprocess --config configs/preprocess/wikipedia_multilang.yaml
```

Produces:

```
data/wikipedia/train/
  20231101.en/  shard_00000.parquet  shard_00001.parquet  …
  20231101.fr/  shard_00000.parquet  …
  20231101.de/  shard_00000.parquet  …
```

`hf_config` and `hf_configs` are mutually exclusive.

---

## Output directory layout

`llm-preprocess` always writes shards directly into the `--output` directory
(or into per-subset subdirectories when `hf_configs` is used).

The conventional layout for a project is:

```
data/
  wikipedia/
    train/          ← llm-preprocess --output data/wikipedia/train --hf-split train
      shard_00000.parquet
      shard_00001.parquet
      …
    val/            ← llm-preprocess --output data/wikipedia/val   --hf-split test
      shard_00000.parquet
      …
  fineweb/
    train/
      …
```

Train and validation sets are produced by running the tool twice with different
`--hf-split` and `--output` values (or by overriding those on the CLI when a
config file provides the base settings).

---

## Built-in adapters

The tool ships with adapters that extract the text field from common datasets:

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
    --output       data/my-corpus/train \
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

| option                | default   | description                                                    |
|-----------------------|-----------|----------------------------------------------------------------|
| `--config` / `-c`     | —         | YAML/JSON preprocess config file; CLI flags override          |
| `--hf-dataset`        | —         | HuggingFace repo ID to stream                                  |
| `--hf-config`         | —         | Single dataset subset/config name (e.g. `20231101.en`)        |
| `--hf-split`          | `train`   | Dataset split                                                  |
| `--trust-remote-code` | `false`   | Trust remote code when loading HF datasets                     |
| `--adapter-module`    | —         | Python file with custom `@register_adapter` calls             |
| `--input`             | —         | Local file, directory, or glob                                 |
| `--text-column`       | `text`    | Column name for text in jsonl/parquet input                    |
| `--output`            | required  | Output directory (or base dir when `hf_configs` is used)      |
| `--tokenizer`         | `hf`      | `hf` / `bpe` / `char`                                        |
| `--tokenizer-path`    | `gpt2`    | HF model name or saved tokenizer path                          |
| `--vocab-size`        | `8000`    | BPE target vocabulary size                                     |
| `--min-frequency`     | `2`       | BPE minimum merge frequency                                    |
| `--save-tokenizer`    | —         | Where to save a newly trained bpe/char tokenizer               |
| `--add-special-tokens`| `true`    | Prepend BOS, append EOS to each document                       |
| `--shard-size`        | `50000`   | Documents per parquet file                                     |
| `--min-length`        | `32`      | Drop documents shorter than this many tokens                   |

The following key is **YAML/JSON config only** (no CLI equivalent):

| key           | description                                                                  |
|---------------|------------------------------------------------------------------------------|
| `hf_configs`  | List of subset names — processes each in turn, writing to `output/<subset>/` |

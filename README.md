# light-llm

A minimal framework for training and running language models from scratch, focused on experimentation — different attention maps, positional encodings, complex-valued layers, custom tokenizers, and more.

## Install

```bash
uv pip install -e .
```

## Quick start

```bash
# 1. Tokenise a dataset using a preprocess config file
llm-preprocess --config configs/preprocess/wikipedia_en.yaml

# Or fully inline (no config file needed):
llm-preprocess \
    --hf-dataset wikimedia/wikipedia \
    --hf-config  20231101.en \
    --hf-split   train \
    --output     data/wikipedia/train \
    --tokenizer  hf --tokenizer-path gpt2

# 2. Analyse the shards (length/token distributions, bigram graph)
llm-analyse --config configs/analyse/example.yaml
# or inline:
llm-analyse --input data/wikipedia/train --output analysis/wikipedia

# 3. Train
llm-train configs/small.yaml

# 4. Generate
llm-run --checkpoint checkpoints/small/best --tokenizer gpt2 \
        --prompt "The history of mathematics"
```

## Docs

- [preprocess](docs/preprocess.md) — tokenise text or HuggingFace datasets into parquet shards
- [analyse](docs/analyse.md) — analyse shard data: length/token histograms and token bigram graph
- [train](docs/train.md) — train a model from a config file
- [run](docs/run.md) — generate text from a trained checkpoint
- [export](docs/export.md) — export to ONNX, TorchScript, or SafeTensors
- [models](docs/models.md) — architecture options and how to configure them
- [tokenizers](docs/tokenizers.md) — supported tokenizers and how to add a custom one

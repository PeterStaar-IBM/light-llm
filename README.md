# light-llm

A minimal framework for training and running language models from scratch, focused on experimentation — different attention maps, positional encodings, complex-valued layers, custom tokenizers, and more.

## Install

```bash
uv pip install -e .
```

## Quick start

```bash
# 1. Tokenise a dataset (streams from HuggingFace)
llm-preprocess \
    --hf-dataset wikimedia/wikipedia \
    --hf-config  20231101.en \
    --output     data/wikipedia \
    --tokenizer  hf --tokenizer-path gpt2

# 2. Train
llm-train configs/small.yaml

# 3. Generate
llm-run --checkpoint checkpoints/small/best --tokenizer gpt2 \
        --prompt "The history of mathematics"
```

## Docs

- [preprocess](docs/preprocess.md) — tokenise text or HuggingFace datasets into parquet shards
- [train](docs/train.md) — train a model from a config file
- [run](docs/run.md) — generate text from a trained checkpoint
- [export](docs/export.md) — export to ONNX, TorchScript, or SafeTensors
- [models](docs/models.md) — architecture options and how to configure them
- [tokenizers](docs/tokenizers.md) — supported tokenizers and how to add a custom one

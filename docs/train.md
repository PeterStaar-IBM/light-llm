# llm-train

Train a Transformer language model from a YAML or JSON config file.

The config file is the **single source of truth** for a run — it is copied automatically into the checkpoint directory so every checkpoint folder is self-contained and reproducible.

---

## Usage

```bash
llm-train configs/small.yaml
```

Resume from a checkpoint:

```bash
llm-train configs/small.yaml --resume checkpoints/small/step0005000
```

---

## Config file format

Both YAML and JSON are accepted (detected by file extension).

A minimal example:

```yaml
train_data: data/wikipedia/train   # directory of parquet shards (from llm-preprocess)
val_data:   data/wikipedia/val
tokenizer_path: gpt2

model:
  vocab_size:  50257
  d_model:     512
  num_layers:  6

seq_len:    1024
batch_size: 32
num_steps:  10000

optimizer:
  lr: 3.0e-4

scheduler:
  type:         cosine
  warmup_steps: 500

dtype:          bfloat16
checkpoint_dir: checkpoints/my-run
```

Ready-to-use configs live in `configs/`:

| file                                    | description                              |
|-----------------------------------------|------------------------------------------|
| `configs/small.yaml`                    | ~25 M params, quick experiments          |
| `configs/medium.yaml`                   | ~120 M params, LLaMA-like recipe         |
| `configs/experiment_complex_attn.yaml`  | complex-valued attention experiment      |

---

## Data loading

Training reads directly from the parquet shards produced by `llm-preprocess`.
No pre-shuffling is done at the file level — instead:

- Each epoch the shard list is **randomly reordered** (seed changes per epoch).
- Within each shard, rows are read **linearly** in the order they appear.
- Tokens from consecutive rows are packed into `seq_len`-length windows with no padding waste.

### Specifying data paths

`train_data` and `val_data` each accept either a **single path** or a **list of paths**.
When a list is given, shards from all sources are pooled into one dataset.

```yaml
# Single directory (all *.parquet files inside are used)
train_data: data/wikipedia/train

# Single .parquet file
train_data: data/wikipedia/train/shard_00000.parquet

# Glob pattern
train_data: "data/wikipedia/train/shard_*.parquet"

# Multiple datasets — shards from all directories are pooled
train_data:
  - data/wikipedia/train/20231101.en
  - data/wikipedia/train/20231101.fr
  - data/wikipedia/train/20231101.de
  - data/fineweb/train

val_data:
  - data/wikipedia/val/20231101.en
```

This maps directly to the directory layout produced by `llm-preprocess`.
See [preprocess.md](preprocess.md) for the recommended layout.

---

## Full config reference

### Top-level

| key                          | default        | description                                        |
|------------------------------|----------------|----------------------------------------------------|
| `train_data`                 | required       | Path, glob, or list of paths to training shards    |
| `val_data`                   | —              | Path, glob, or list of paths to validation shards  |
| `tokenizer_path`             | —              | HF model name or saved tokenizer path              |
| `seq_len`                    | `1024`         | Training context length (tokens)                   |
| `batch_size`                 | `32`           | Sequences per gradient step                        |
| `gradient_accumulation_steps`| `1`            | Micro-batches before an optimizer step             |
| `num_steps`                  | `10000`        | Total optimizer steps                              |
| `dtype`                      | `bfloat16`     | `float32` / `float16` / `bfloat16`                |
| `checkpoint_dir`             | `checkpoints`  | Where to write checkpoints                         |
| `save_every`                 | `1000`         | Save a checkpoint every N steps                    |
| `keep_last_n`                | `3`            | How many step checkpoints to retain                |
| `log_every`                  | `10`           | Log metrics every N steps                          |
| `eval_every`                 | `500`          | Run validation every N steps                       |
| `eval_steps`                 | `50`           | Validation batches per eval run                    |
| `use_wandb`                  | `false`        | Enable Weights & Biases logging                    |
| `wandb_project`              | `light-llm`    | W&B project name                                   |
| `wandb_run_name`             | —              | W&B run name (auto-generated if omitted)           |
| `seed`                       | `42`           | Random seed                                        |
| `num_workers`                | `4`            | DataLoader worker processes                        |
| `compile_model`              | `false`        | Enable `torch.compile()` (PyTorch 2.x)             |

### `model`

| key                | default  | description                                   |
|--------------------|----------|-----------------------------------------------|
| `vocab_size`       | `32000`  | Vocabulary size (must match tokenizer)        |
| `d_model`          | `512`    | Hidden dimension                              |
| `num_layers`       | `6`      | Number of transformer layers                  |
| `max_seq_len`      | `2048`   | Maximum sequence length                       |
| `dropout`          | `0.0`    | Embedding / residual dropout                  |
| `tie_embeddings`   | `true`   | Share input and output embedding weights      |
| `parallel_layers`  | `false`  | PaLM-style parallel attention + FFN           |

See [models.md](models.md) for the `attention`, `positional`, `ffn`, and `norm` sub-keys.

### `optimizer`

| key            | default  | description                  |
|----------------|----------|------------------------------|
| `type`         | `adamw`  | `adamw` / `adam` / `sgd`    |
| `lr`           | `3e-4`   | Peak learning rate            |
| `weight_decay` | `0.1`    | L2 weight decay               |
| `beta1`        | `0.9`    | Adam β₁                      |
| `beta2`        | `0.95`   | Adam β₂                      |
| `eps`          | `1e-8`   | Adam ε                       |
| `grad_clip`    | `1.0`    | Gradient clipping norm        |

### `scheduler`

| key             | default    | description                                         |
|-----------------|------------|-----------------------------------------------------|
| `type`          | `cosine`   | `cosine` / `linear` / `constant` / `wsd`           |
| `warmup_steps`  | `100`      | Linear warmup steps                                 |
| `min_lr_ratio`  | `0.1`      | Final LR = `lr × min_lr_ratio`                     |

`wsd` (Warmup-Stable-Decay): 10 % warmup → 80 % stable → 10 % cosine decay.

---

## Checkpoints

Each checkpoint tag (`best`, `final`, `step<N>`) produces three files:

```
checkpoints/my-run/
  best.safetensors   ← model weights (lowest val loss)
  best.state.pt      ← optimizer + scheduler + scaler state
  best.config.json   ← full config snapshot
  small.yaml         ← copy of the original config file
  step0001000.safetensors
  step0001000.state.pt
  step0001000.config.json
  ...
```

Load weights independently of the trainer (e.g. for inference):

```python
from safetensors.torch import load_file
from light_llm.models.transformer import Transformer, TransformerConfig

cfg = TransformerConfig(...)
model = Transformer(cfg)
model.load_state_dict(load_file("checkpoints/my-run/best.safetensors"))
```

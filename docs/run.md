# llm-run

Generate text from a trained checkpoint.

---

## Usage

```bash
# Single prompt
llm-run \
    --checkpoint checkpoints/small/best \
    --tokenizer  gpt2 \
    --prompt     "The theory of relativity states that"

# Interactive REPL (keep hitting Enter for more)
llm-run \
    --checkpoint checkpoints/small/best \
    --tokenizer  gpt2 \
    --interactive
```

The checkpoint tag (e.g. `best`, `final`, `step0005000`) is resolved relative to the checkpoint directory.  Pass either the bare tag or the full `.safetensors` path:

```bash
# Both are equivalent
llm-run --checkpoint checkpoints/small/best ...
llm-run --checkpoint checkpoints/small/best.safetensors ...
```

The tool reads `best.config.json` alongside the weights to reconstruct the model architecture automatically — no need to specify model dimensions manually.

---

## Sampling options

| option                 | default     | description                                                  |
|------------------------|-------------|--------------------------------------------------------------|
| `--max-tokens`         | `200`       | Maximum new tokens to generate                               |
| `--temperature`        | `1.0`       | Sampling temperature — lower = more deterministic            |
| `--top-k`              | —           | Keep only the top-k most likely tokens before sampling       |
| `--top-p`              | —           | Nucleus sampling: keep tokens summing to probability ≥ p     |
| `--repetition-penalty` | `1.0`       | Values > 1 reduce repetition (1.0 = disabled)               |

Common recipe for coherent but creative output:

```bash
llm-run --checkpoint checkpoints/small/best --tokenizer gpt2 \
        --prompt "Once upon a time" \
        --temperature 0.8 --top-p 0.9 --repetition-penalty 1.1 \
        --max-tokens 300
```

Greedy decoding (always pick the most likely token):

```bash
llm-run --checkpoint checkpoints/small/best --tokenizer gpt2 \
        --prompt "The capital of France is" \
        --temperature 0.0
```

---

## All options

| option                 | default    | description                                     |
|------------------------|------------|-------------------------------------------------|
| `--checkpoint`         | required   | Checkpoint path or tag                          |
| `--tokenizer`          | required   | HF model name or saved tokenizer directory      |
| `--prompt`             | —          | Input text (required unless `--interactive`)    |
| `--interactive`        | `false`    | Start an interactive generation loop            |
| `--max-tokens`         | `200`      | Maximum new tokens                              |
| `--temperature`        | `1.0`      | Sampling temperature                            |
| `--top-k`              | —          | Top-k filtering                                 |
| `--top-p`              | —          | Nucleus sampling threshold                      |
| `--repetition-penalty` | `1.0`      | Repetition penalty                              |
| `--dtype`              | `bfloat16` | Inference dtype: `float32` / `float16` / `bfloat16` |
| `--device`             | `cuda`     | `cuda` or `cpu`                                 |

---

## Using the model directly in Python

```python
import torch
from safetensors.torch import load_file
from light_llm.models.transformer import Transformer
from light_llm.training.config import TrainingConfig
from light_llm.tokenizer import HFTokenizer

# Load config + weights
cfg = TrainingConfig.model_validate_json(
    open("checkpoints/small/best.config.json").read()
)
model = Transformer(cfg.model)
model.load_state_dict(load_file("checkpoints/small/best.safetensors"))
model = model.cuda().eval()

# Tokenize and generate
tok = HFTokenizer.from_pretrained("gpt2")
ids = tok.encode("Once upon a time", add_special_tokens=True)
prompt = torch.tensor([ids], dtype=torch.long, device="cuda")

with torch.inference_mode():
    out = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_p=0.9)

print(tok.decode(out[0].tolist()))
```
